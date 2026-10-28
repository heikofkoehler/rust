//! Concurrent rate limiters: token bucket and sliding window,
//! plus a sharded multi-tenant manager.
//!
//! Port of Go `pkg/ratelimiter`.
//!
//! Traps preserved:
//!
//! * Clock precision and drift: token refills are computed
//!   analytically on demand from elapsed monotonic time
//!   (`Instant`), never by a per-bucket ticker, so there is no
//!   cumulative drift.
//! * The sliding window keeps an exact log of request timestamps and
//!   prunes expired entries under the lock, eliminating the
//!   end-of-window burst a fixed window counter allows.
//! * The manager shards tenants (FNV-1a) to avoid one global lock,
//!   updates last-access with an atomic timestamp on the read path,
//!   and evicts idle tenants so the tenant map cannot leak.

use crate::cancel::CancelToken;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Rate limiter errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RateLimitError {
    LimitExceeded,
    ManagerClosed,
    ExceedsCapacity(String),
    Cancelled,
}

impl fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RateLimitError::LimitExceeded => write!(f, "rate limit exceeded"),
            RateLimitError::ManagerClosed => write!(f, "rate limiter manager closed"),
            RateLimitError::ExceedsCapacity(msg) => write!(f, "{msg}"),
            RateLimitError::Cancelled => write!(f, "wait cancelled"),
        }
    }
}

impl std::error::Error for RateLimitError {}

fn fnv1a_64(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// The standard limiter contract, mirroring Go's `Limiter` interface.
///
/// `allow_n` takes an explicit timestamp so tests can drive time
/// deterministically, exactly as Go's `AllowN(time.Time, n)`.
pub trait Limiter: Send + Sync {
    fn allow(&self) -> bool;
    fn allow_n(&self, now: Instant, n: usize) -> bool;
    fn wait(&self, cancel: &CancelToken) -> Result<(), RateLimitError>;
    fn wait_n(&self, cancel: &CancelToken, n: usize) -> Result<(), RateLimitError>;
}

/// Token bucket: bursts up to `capacity`, refilling at `refill_rate`
/// tokens per second. Starts full, as in Go.
#[derive(Debug)]
pub struct TokenBucket {
    inner: Mutex<TokenBucketState>,
    capacity: f64,
    refill_rate: f64,
}

#[derive(Debug)]
struct TokenBucketState {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Panics on non-positive capacity or rate, as Go does.
    pub fn new(capacity: f64, refill_rate: f64) -> TokenBucket {
        if capacity <= 0.0 || refill_rate <= 0.0 {
            panic!(
                "capacity and refillRate must be strictly positive: cap={capacity}, rate={refill_rate}"
            );
        }
        TokenBucket {
            inner: Mutex::new(TokenBucketState {
                tokens: capacity,
                last_refill: Instant::now(),
            }),
            capacity,
            refill_rate,
        }
    }

    fn refill_locked(state: &mut TokenBucketState, refill_rate: f64, capacity: f64, now: Instant) {
        let elapsed = now.saturating_duration_since(state.last_refill);
        if elapsed.is_zero() {
            return;
        }
        state.tokens = capacity.min(state.tokens + elapsed.as_secs_f64() * refill_rate);
        state.last_refill = now;
    }

    pub fn available_tokens(&self) -> f64 {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Self::refill_locked(&mut state, self.refill_rate, self.capacity, Instant::now());
        state.tokens
    }
}

impl Limiter for TokenBucket {
    fn allow(&self) -> bool {
        self.allow_n(Instant::now(), 1)
    }

    fn allow_n(&self, now: Instant, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        if n as f64 > self.capacity {
            return false;
        }
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Self::refill_locked(&mut state, self.refill_rate, self.capacity, now);
        if state.tokens >= n as f64 {
            state.tokens -= n as f64;
            true
        } else {
            false
        }
    }

    fn wait(&self, cancel: &CancelToken) -> Result<(), RateLimitError> {
        self.wait_n(cancel, 1)
    }

    fn wait_n(&self, cancel: &CancelToken, n: usize) -> Result<(), RateLimitError> {
        if n == 0 {
            return Ok(());
        }
        if n as f64 > self.capacity {
            return Err(RateLimitError::ExceedsCapacity(format!(
                "requested {n} tokens exceeds capacity of {}",
                self.capacity as i64
            )));
        }
        loop {
            let wait_duration = {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                let now = Instant::now();
                Self::refill_locked(&mut state, self.refill_rate, self.capacity, now);
                if state.tokens >= n as f64 {
                    state.tokens -= n as f64;
                    return Ok(());
                }
                let missing = n as f64 - state.tokens;
                // Go: ceil(missing / rate * 1s) in nanoseconds, i.e.
                // missing / rate seconds, not whole seconds.
                Duration::from_secs_f64(missing / self.refill_rate).max(Duration::from_millis(1))
            };
            if cancel.sleep(wait_duration) {
                return Err(RateLimitError::Cancelled);
            }
        }
    }
}

/// Exact sliding-window log limiter.
#[derive(Debug)]
pub struct SlidingWindow {
    inner: Mutex<SlidingWindowState>,
    limit: usize,
    window: Duration,
}

#[derive(Debug)]
struct SlidingWindowState {
    timestamps: Vec<Instant>,
}

impl SlidingWindow {
    /// Panics on non-positive limit or window, as Go does.
    pub fn new(limit: usize, window: Duration) -> SlidingWindow {
        if limit == 0 || window.is_zero() {
            panic!("limit and window must be strictly positive: limit={limit}, window={window:?}");
        }
        SlidingWindow {
            inner: Mutex::new(SlidingWindowState {
                timestamps: Vec::with_capacity(limit),
            }),
            limit,
            window,
        }
    }

    fn prune_locked(state: &mut SlidingWindowState, window: Duration, now: Instant) {
        let threshold = now.checked_sub(window).unwrap_or(now);
        // Go prunes timestamps strictly Before(threshold); a
        // timestamp exactly at the threshold survives.
        let start = state
            .timestamps
            .iter()
            .position(|t| *t >= threshold)
            .unwrap_or(state.timestamps.len());
        if start > 0 {
            state.timestamps.drain(..start);
        }
    }

    pub fn current_count(&self) -> usize {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Self::prune_locked(&mut state, self.window, Instant::now());
        state.timestamps.len()
    }
}

impl Limiter for SlidingWindow {
    fn allow(&self) -> bool {
        self.allow_n(Instant::now(), 1)
    }

    fn allow_n(&self, now: Instant, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        if n > self.limit {
            return false;
        }
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Self::prune_locked(&mut state, self.window, now);
        if state.timestamps.len() + n <= self.limit {
            for _ in 0..n {
                state.timestamps.push(now);
            }
            true
        } else {
            false
        }
    }

    fn wait(&self, cancel: &CancelToken) -> Result<(), RateLimitError> {
        self.wait_n(cancel, 1)
    }

    fn wait_n(&self, cancel: &CancelToken, n: usize) -> Result<(), RateLimitError> {
        if n == 0 {
            return Ok(());
        }
        if n > self.limit {
            return Err(RateLimitError::ExceedsCapacity(format!(
                "requested {n} exceeds sliding window limit {}",
                self.limit
            )));
        }
        loop {
            let wait_duration = {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                let now = Instant::now();
                Self::prune_locked(&mut state, self.window, now);
                if state.timestamps.len() + n <= self.limit {
                    for _ in 0..n {
                        state.timestamps.push(now);
                    }
                    return Ok(());
                }
                let idx_to_free = state.timestamps.len() + n - self.limit - 1;
                let earliest = state.timestamps[idx_to_free];
                earliest
                    .checked_add(self.window)
                    .and_then(|t| t.checked_duration_since(now))
                    .unwrap_or(Duration::from_millis(1))
                    .max(Duration::from_millis(1))
            };
            if cancel.sleep(wait_duration) {
                return Err(RateLimitError::Cancelled);
            }
        }
    }
}

type LimiterFactory = Arc<dyn Fn() -> Arc<dyn Limiter> + Send + Sync>;

struct TenantEntry {
    limiter: Arc<dyn Limiter>,
    last_access_nanos: AtomicI64,
}

struct TenantShard {
    entries: RwLock<HashMap<String, TenantEntry>>,
}

fn now_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Configuration for [`Manager`].
pub struct ManagerConfig {
    pub num_shards: usize,
    pub idle_ttl: Duration,
    pub cleanup_interval: Duration,
    pub limiter_factory: Option<LimiterFactory>,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        ManagerConfig {
            num_shards: 16,
            idle_ttl: Duration::from_secs(10 * 60),
            cleanup_interval: Duration::from_secs(60),
            limiter_factory: None,
        }
    }
}

#[derive(Debug)]
struct EvictionStop {
    stopped: Mutex<bool>,
    changed: Condvar,
}

impl EvictionStop {
    fn new() -> EvictionStop {
        EvictionStop {
            stopped: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn stop(&self) {
        let mut flag = self.stopped.lock().unwrap_or_else(|e| e.into_inner());
        *flag = true;
        self.changed.notify_all();
    }

    fn wait(&self, interval: Duration) -> bool {
        let flag = self.stopped.lock().unwrap_or_else(|e| e.into_inner());
        if *flag {
            return true;
        }
        let (flag, _) = self
            .changed
            .wait_timeout(flag, interval)
            .unwrap_or_else(|e| e.into_inner());
        *flag
    }
}

/// Sharded multi-tenant limiter manager with idle eviction.
pub struct Manager {
    shards: Vec<TenantShard>,
    factory: LimiterFactory,
    idle_ttl: Duration,
    closed: AtomicBool,
    stop: Arc<EvictionStop>,
    evictor: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Manager {
    pub fn new(config: ManagerConfig) -> Arc<Manager> {
        let num_shards = if config.num_shards > 0 {
            config.num_shards
        } else {
            16
        };
        let idle_ttl = if config.idle_ttl.is_zero() {
            Duration::from_secs(10 * 60)
        } else {
            config.idle_ttl
        };
        let cleanup_interval = if config.cleanup_interval.is_zero() {
            Duration::from_secs(60)
        } else {
            config.cleanup_interval
        };
        let factory: LimiterFactory = config.limiter_factory.unwrap_or_else(|| {
            Arc::new(|| Arc::new(TokenBucket::new(100.0, 20.0)) as Arc<dyn Limiter>)
        });
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(TenantShard {
                entries: RwLock::new(HashMap::new()),
            });
        }
        let manager = Arc::new(Manager {
            shards,
            factory,
            idle_ttl,
            closed: AtomicBool::new(false),
            stop: Arc::new(EvictionStop::new()),
            evictor: Mutex::new(None),
        });
        let evictor_manager = Arc::clone(&manager);
        let handle = std::thread::spawn(move || loop {
            if evictor_manager.stop.wait(cleanup_interval) {
                return;
            }
            evictor_manager.prune_idle_tenants();
        });
        *manager.evictor.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
        manager
    }

    fn shard(&self, tenant_id: &str) -> &TenantShard {
        let idx = (fnv1a_64(tenant_id) % self.shards.len() as u64) as usize;
        &self.shards[idx]
    }

    pub fn get_limiter(&self, tenant_id: &str) -> Result<Arc<dyn Limiter>, RateLimitError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(RateLimitError::ManagerClosed);
        }
        let shard = self.shard(tenant_id);
        let now = now_nanos();
        // Fast path: read lock, atomic timestamp store.
        {
            let entries = shard.entries.read().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = entries.get(tenant_id) {
                entry.last_access_nanos.store(now, Ordering::Relaxed);
                return Ok(Arc::clone(&entry.limiter));
            }
        }
        // Slow path: write lock, double-check, then instantiate.
        let mut entries = shard.entries.write().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get(tenant_id) {
            entry.last_access_nanos.store(now, Ordering::Relaxed);
            return Ok(Arc::clone(&entry.limiter));
        }
        let limiter = (self.factory)();
        entries.insert(
            tenant_id.to_string(),
            TenantEntry {
                limiter: Arc::clone(&limiter),
                last_access_nanos: AtomicI64::new(now),
            },
        );
        Ok(limiter)
    }

    pub fn allow(&self, tenant_id: &str) -> bool {
        match self.get_limiter(tenant_id) {
            Ok(limiter) => limiter.allow(),
            Err(_) => false,
        }
    }

    pub fn allow_n(&self, tenant_id: &str, n: usize) -> bool {
        match self.get_limiter(tenant_id) {
            Ok(limiter) => limiter.allow_n(Instant::now(), n),
            Err(_) => false,
        }
    }

    pub fn wait(&self, cancel: &CancelToken, tenant_id: &str) -> Result<(), RateLimitError> {
        let limiter = self.get_limiter(tenant_id)?;
        limiter.wait(cancel)
    }

    pub fn active_tenant_count(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.entries.read().unwrap_or_else(|e| e.into_inner()).len())
            .sum()
    }

    pub fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.stop.stop();
        if let Some(handle) = self
            .evictor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }

    fn prune_idle_tenants(&self) {
        let cutoff = now_nanos() - self.idle_ttl.as_nanos() as i64;
        for shard in &self.shards {
            let mut entries = shard.entries.write().unwrap_or_else(|e| e.into_inner());
            entries.retain(|_, entry| entry.last_access_nanos.load(Ordering::Relaxed) >= cutoff);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn token_bucket_burst_and_refill() {
        let bucket = TokenBucket::new(5.0, 10.0);
        for i in 0..5 {
            assert!(bucket.allow(), "expected token {i} to be allowed");
        }
        assert!(!bucket.allow(), "expected 6th request to be rejected");
        thread::sleep(Duration::from_millis(250));
        assert!(
            bucket.allow_n(Instant::now(), 2),
            "expected 2 tokens refilled"
        );
    }

    #[test]
    fn token_bucket_wait_context_cancel() {
        let bucket = TokenBucket::new(1.0, 1.0);
        assert!(bucket.allow()); // Deplete.
        let token = CancelToken::new();
        let canceller = token.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            canceller.cancel();
        });
        assert!(bucket.wait(&token).is_err());
    }

    #[test]
    fn sliding_window_edge_burst_prevention() {
        let window = SlidingWindow::new(5, Duration::from_millis(200));
        let now = Instant::now();
        for i in 0..5 {
            assert!(window.allow_n(now, 1), "expected request {i} to be allowed");
        }
        assert!(!window.allow_n(now, 1), "expected 6th request to fail");
        let future = now + Duration::from_millis(250);
        assert!(window.allow_n(future, 1), "expected request in new window");
    }

    #[test]
    fn sliding_window_wait() {
        let window = SlidingWindow::new(2, Duration::from_millis(100));
        assert!(window.allow());
        assert!(window.allow());
        let token = CancelToken::new();
        let canceller = token.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(500));
            canceller.cancel();
        });
        assert!(window.wait(&token).is_ok());
    }

    #[test]
    fn manager_multi_tenant_isolation() {
        let config = ManagerConfig {
            num_shards: 4,
            limiter_factory: Some(Arc::new(|| {
                Arc::new(TokenBucket::new(2.0, 1.0)) as Arc<dyn Limiter>
            })),
            ..Default::default()
        };
        let manager = Manager::new(config);
        assert!(manager.allow("tenant-A"));
        assert!(manager.allow("tenant-A"));
        assert!(!manager.allow("tenant-A"), "tenant-A should be throttled");
        assert!(manager.allow("tenant-B"), "tenant-B independent");
        manager.close();
    }

    #[test]
    fn manager_idle_tenant_eviction() {
        let config = ManagerConfig {
            num_shards: 2,
            idle_ttl: Duration::from_millis(50),
            cleanup_interval: Duration::from_millis(25),
            limiter_factory: Some(Arc::new(|| {
                Arc::new(TokenBucket::new(10.0, 10.0)) as Arc<dyn Limiter>
            })),
        };
        let manager = Manager::new(config);
        assert!(manager.allow("tenant-A"));
        assert_eq!(manager.active_tenant_count(), 1);
        thread::sleep(Duration::from_millis(120));
        assert_eq!(manager.active_tenant_count(), 0);
        manager.close();
    }

    #[test]
    fn manager_high_concurrency_stress() {
        let config = ManagerConfig {
            num_shards: 8,
            limiter_factory: Some(Arc::new(|| {
                Arc::new(TokenBucket::new(100.0, 1000.0)) as Arc<dyn Limiter>
            })),
            ..Default::default()
        };
        let manager = Manager::new(config);
        thread::scope(|s| {
            for w in 0..20 {
                let manager = Arc::clone(&manager);
                s.spawn(move || {
                    for i in 0..100 {
                        let tenant = format!("tenant-{}", (w + i) % 10);
                        let _ = manager.allow(&tenant);
                        let _ = manager.get_limiter(&tenant);
                    }
                });
            }
        });
        manager.close();
    }
}
