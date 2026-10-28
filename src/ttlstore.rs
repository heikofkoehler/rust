//! Thread-safe in-memory store with TTL.
//!
//! Port of Go `pkg/ttlstore`.
//!
//! Traps preserved:
//!
//! * Read-write contention is cut by partitioning keys across
//!   independent shards (FNV-1a 64-bit hashing, as in Go), each with
//!   its own `RwLock`.
//! * Lazy cleanup: an expired key met on `get` is deleted on the fly.
//! * Active cleanup: a background reaper sweeps one shard at a time,
//!   in bounded batches, so it never starves live operations.
//! * CAS and CAD evaluate and mutate under the shard write lock, so
//!   there is no read-then-write race window.
//!
//! One deliberate divergence: Go's store holds `any` values and
//! compares them with `reflect.DeepEqual`. A Rust store holding
//! `Box<dyn Any>` cannot compare values without downcasting, so this
//! store is generic over `V: Clone + PartialEq + Send + Sync`. A
//! `Store<String>` and a `Store<i64>` are separate types, which is
//! the idiomatic Rust shape of the same idea.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

const DEFAULT_SHARDS: usize = 32;
const DEFAULT_CLEANUP_INTERVAL: Duration = Duration::from_millis(100);
const DEFAULT_MAX_SWEEP_BATCH: usize = 100;

/// Store errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreError {
    KeyNotFound,
    StoreClosed,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::KeyNotFound => write!(f, "key not found or expired"),
            StoreError::StoreClosed => write!(f, "store is closed"),
        }
    }
}

impl std::error::Error for StoreError {}

fn fnv1a_64(key: &str) -> u64 {
    // FNV-1a 64, matching Go's hash/fnv New64a.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Debug)]
struct Item<V> {
    value: V,
    expires_at: Option<Instant>,
}

impl<V> Item<V> {
    fn is_expired(&self, now: Instant) -> bool {
        match self.expires_at {
            None => false,
            Some(t) => now > t,
        }
    }

    /// Remaining TTL: `None` means no expiration (Go returns -1).
    fn remaining_ttl(&self, now: Instant) -> Option<Duration> {
        self.expires_at.map(|t| t.saturating_duration_since(now))
    }
}

#[derive(Debug)]
struct Shard<V> {
    items: RwLock<HashMap<String, Item<V>>>,
}

impl<V: Clone + PartialEq> Shard<V> {
    fn new() -> Shard<V> {
        Shard {
            items: RwLock::new(HashMap::new()),
        }
    }

    fn get(&self, key: &str, now: Instant) -> Option<V> {
        {
            let items = self.items.read().unwrap_or_else(|e| e.into_inner());
            match items.get(key) {
                Some(item) if !item.is_expired(now) => return Some(item.value.clone()),
                Some(_) => {} // Expired: fall through to lazy cleanup.
                None => return None,
            }
        }
        // Lazy cleanup: promote to a write lock and remove if still expired.
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        if let Some(item) = items.get(key) {
            if item.is_expired(now) {
                items.remove(key);
            }
        }
        None
    }

    fn set(&self, key: &str, value: V, expires_at: Option<Instant>) {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        items.insert(key.to_string(), Item { value, expires_at });
    }

    fn set_nx(&self, key: &str, value: V, expires_at: Option<Instant>, now: Instant) -> bool {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        if let Some(item) = items.get(key) {
            if !item.is_expired(now) {
                return false;
            }
        }
        items.insert(key.to_string(), Item { value, expires_at });
        true
    }

    fn delete(&self, key: &str, now: Instant) -> bool {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        match items.remove(key) {
            Some(item) => !item.is_expired(now),
            None => false,
        }
    }

    fn cas(&self, key: &str, expected: &V, new_value: V, now: Instant) -> Result<bool, StoreError> {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        let item = match items.get_mut(key) {
            Some(item) if !item.is_expired(now) => item,
            Some(_) => {
                items.remove(key);
                return Err(StoreError::KeyNotFound);
            }
            None => return Err(StoreError::KeyNotFound),
        };
        if item.value != *expected {
            return Ok(false);
        }
        item.value = new_value;
        Ok(true)
    }

    fn cad(&self, key: &str, expected: &V, now: Instant) -> Result<bool, StoreError> {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        let present = match items.get(key) {
            Some(item) if !item.is_expired(now) => item.value == *expected,
            Some(_) => {
                items.remove(key);
                return Err(StoreError::KeyNotFound);
            }
            None => return Err(StoreError::KeyNotFound),
        };
        if present {
            items.remove(key);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn ttl(&self, key: &str, now: Instant) -> Option<Option<Duration>> {
        let items = self.items.read().unwrap_or_else(|e| e.into_inner());
        match items.get(key) {
            Some(item) if !item.is_expired(now) => Some(item.remaining_ttl(now)),
            _ => None,
        }
    }

    fn sweep(&self, max_per_shard: usize, now: Instant) -> usize {
        let mut items = self.items.write().unwrap_or_else(|e| e.into_inner());
        let mut checked = 0;
        let mut deleted = 0;
        // Snapshot keys first: Go ranges the map directly, but
        // deleting while iterating a Rust HashMap needs the key list.
        let keys: Vec<String> = items.keys().cloned().collect();
        for key in keys {
            if let Some(item) = items.get(&key) {
                if item.is_expired(now) {
                    items.remove(&key);
                    deleted += 1;
                }
            }
            checked += 1;
            if max_per_shard > 0 && checked >= max_per_shard {
                break;
            }
        }
        deleted
    }

    /// Count of stored items, including ones not yet reaped — as in Go.
    fn size(&self) -> usize {
        self.items.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// A stop signal for a background thread: a flag plus a condvar so
/// `close` wakes the reaper immediately instead of waiting out the
/// full cleanup interval (which Go's ticker-based loop also avoids
/// only via its stop channel — same idea, std primitives).
#[derive(Debug)]
struct StopSignal {
    stopped: Mutex<bool>,
    changed: Condvar,
}

impl StopSignal {
    fn new() -> StopSignal {
        StopSignal {
            stopped: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn stop(&self) {
        let mut flag = self.stopped.lock().unwrap_or_else(|e| e.into_inner());
        *flag = true;
        self.changed.notify_all();
    }

    fn is_stopped(&self) -> bool {
        *self.stopped.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait up to `interval`, returning `true` if stopped.
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

/// Builder options, mirroring Go's functional `Option` pattern with
/// a plain struct (the idiomatic Rust equivalent).
#[derive(Clone, Debug)]
pub struct StoreOptions {
    pub shards: usize,
    pub cleanup_interval: Duration,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            shards: DEFAULT_SHARDS,
            cleanup_interval: DEFAULT_CLEANUP_INTERVAL,
        }
    }
}

/// A sharded, thread-safe in-memory key-value store with TTL.
#[derive(Debug)]
pub struct Store<V> {
    shards: Vec<Shard<V>>,
    cleanup_interval: Duration,
    closed: AtomicBool,
    stop: Arc<StopSignal>,
    reaper: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl<V: Clone + PartialEq + Send + Sync + 'static> Store<V> {
    pub fn new(options: StoreOptions) -> Arc<Store<V>> {
        let shards_count = if options.shards > 0 {
            options.shards
        } else {
            DEFAULT_SHARDS
        };
        let cleanup_interval = if options.cleanup_interval.as_nanos() > 0 {
            options.cleanup_interval
        } else {
            DEFAULT_CLEANUP_INTERVAL
        };
        let mut shards = Vec::with_capacity(shards_count);
        for _ in 0..shards_count {
            shards.push(Shard::new());
        }
        let store = Arc::new(Store {
            shards,
            cleanup_interval,
            closed: AtomicBool::new(false),
            stop: Arc::new(StopSignal::new()),
            reaper: Mutex::new(None),
        });
        let reaper_store = Arc::clone(&store);
        let handle = std::thread::spawn(move || loop {
            if reaper_store.stop.wait(reaper_store.cleanup_interval) {
                return;
            }
            reaper_store.sweep_shards();
        });
        *store.reaper.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
        store
    }

    /// Convenience constructor with default options.
    pub fn with_defaults() -> Arc<Store<V>> {
        Self::new(StoreOptions::default())
    }

    fn shard(&self, key: &str) -> &Shard<V> {
        let idx = (fnv1a_64(key) % self.shards.len() as u64) as usize;
        &self.shards[idx]
    }

    fn check_open(&self) -> Result<(), StoreError> {
        if self.closed.load(Ordering::SeqCst) {
            Err(StoreError::StoreClosed)
        } else {
            Ok(())
        }
    }

    fn expiry_for(ttl: Duration) -> Option<Instant> {
        if ttl.as_nanos() > 0 {
            Some(Instant::now() + ttl)
        } else {
            None
        }
    }

    pub fn set(&self, key: &str, value: V) -> Result<(), StoreError> {
        self.check_open()?;
        self.shard(key).set(key, value, None);
        Ok(())
    }

    pub fn set_with_ttl(&self, key: &str, value: V, ttl: Duration) -> Result<(), StoreError> {
        self.check_open()?;
        self.shard(key).set(key, value, Self::expiry_for(ttl));
        Ok(())
    }

    pub fn set_nx(&self, key: &str, value: V, ttl: Duration) -> Result<bool, StoreError> {
        self.check_open()?;
        Ok(self
            .shard(key)
            .set_nx(key, value, Self::expiry_for(ttl), Instant::now()))
    }

    pub fn get(&self, key: &str) -> Option<V> {
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        self.shard(key).get(key, Instant::now())
    }

    pub fn delete(&self, key: &str) -> Result<bool, StoreError> {
        self.check_open()?;
        Ok(self.shard(key).delete(key, Instant::now()))
    }

    pub fn cas(&self, key: &str, expected: &V, new_value: V) -> Result<bool, StoreError> {
        self.check_open()?;
        self.shard(key)
            .cas(key, expected, new_value, Instant::now())
    }

    pub fn cad(&self, key: &str, expected: &V) -> Result<bool, StoreError> {
        self.check_open()?;
        self.shard(key).cad(key, expected, Instant::now())
    }

    /// Remaining TTL.
    ///
    /// Outer `None` = missing or expired. Inner `None` = no
    /// expiration (Go returns -1 for this case).
    pub fn ttl(&self, key: &str) -> Option<Option<Duration>> {
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        self.shard(key).ttl(key, Instant::now())
    }

    pub fn size(&self) -> usize {
        self.shards.iter().map(|s| s.size()).sum()
    }

    /// Stop the background reaper. Idempotent, as in Go.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.stop.stop();
        if let Some(handle) = self.reaper.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = handle.join();
        }
    }

    fn sweep_shards(&self) {
        let now = Instant::now();
        for shard in &self.shards {
            if self.stop.is_stopped() {
                return;
            }
            shard.sweep(DEFAULT_MAX_SWEEP_BATCH, now);
        }
    }
}

impl<V> Drop for Store<V> {
    fn drop(&mut self) {
        // Closing here only signals the reaper; joining from Drop
        // would risk blocking on a thread that holds a shard lock
        // this thread is waiting on. Explicit `close()` joins.
        self.stop.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;
    use std::thread;

    fn opts(shards: usize, interval: Duration) -> StoreOptions {
        StoreOptions {
            shards,
            cleanup_interval: interval,
        }
    }

    #[test]
    fn basic_crud() {
        let store = Store::<String>::new(opts(8, Duration::from_millis(100)));
        store.set("key1", "value1".to_string()).unwrap();
        assert_eq!(store.get("key1").as_deref(), Some("value1"));
        assert!(store.delete("key1").unwrap());
        assert_eq!(store.get("key1"), None);
        store.close();
    }

    #[test]
    fn lazy_ttl_expiration() {
        // Long reaper interval, so expiry must come from lazy cleanup.
        let store = Store::<String>::new(opts(4, Duration::from_secs(10)));
        store
            .set_with_ttl(
                "temp-key",
                "temp-val".to_string(),
                Duration::from_millis(50),
            )
            .unwrap();
        assert_eq!(store.get("temp-key").as_deref(), Some("temp-val"));
        thread::sleep(Duration::from_millis(80));
        assert_eq!(store.get("temp-key"), None);
        store.close();
    }

    #[test]
    fn active_background_reaper() {
        let store = Store::<String>::new(opts(4, Duration::from_millis(25)));
        store
            .set_with_ttl(
                "active-exp-1",
                "val1".to_string(),
                Duration::from_millis(30),
            )
            .unwrap();
        store
            .set_with_ttl(
                "active-exp-2",
                "val2".to_string(),
                Duration::from_millis(30),
            )
            .unwrap();
        assert_eq!(store.size(), 2);
        thread::sleep(Duration::from_millis(120));
        assert_eq!(store.size(), 0);
        store.close();
    }

    #[test]
    fn set_nx() {
        let store = Store::<String>::new(opts(4, Duration::from_millis(100)));
        assert!(store
            .set_nx(
                "lock-key",
                "holder-1".to_string(),
                Duration::from_millis(100)
            )
            .unwrap());
        assert!(!store
            .set_nx(
                "lock-key",
                "holder-2".to_string(),
                Duration::from_millis(100)
            )
            .unwrap());
        assert_eq!(store.get("lock-key").as_deref(), Some("holder-1"));
        store.close();
    }

    #[test]
    fn atomic_cas_contention() {
        let store = Store::<i64>::new(opts(4, Duration::from_millis(100)));
        store.set("counter", 0).unwrap();
        let failures = AtomicI64::new(0);
        let increments = 100;
        thread::scope(|s| {
            for _ in 0..increments {
                s.spawn(|| loop {
                    let current = store.get("counter").expect("key vanished");
                    match store.cas("counter", &current, current + 1) {
                        Ok(true) => break,
                        Ok(false) => {
                            failures.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(e) => panic!("CAS error: {e}"),
                    }
                });
            }
        });
        assert_eq!(store.get("counter"), Some(increments));
        store.close();
    }

    #[test]
    fn cad() {
        let store = Store::<String>::new(opts(4, Duration::from_millis(100)));
        store.set("cad-key", "secret-token".to_string()).unwrap();
        assert!(!store.cad("cad-key", &"wrong-token".to_string()).unwrap());
        assert!(store.cad("cad-key", &"secret-token".to_string()).unwrap());
        assert_eq!(store.get("cad-key"), None);
        store.close();
    }

    #[test]
    fn heavy_read_write_contention() {
        let store = Store::<i64>::new(opts(16, Duration::from_millis(15)));
        thread::scope(|s| {
            for worker in 0..40 {
                let store = Arc::clone(&store);
                s.spawn(move || {
                    for i in 0..200 {
                        let key = format!("k-{}", i % 20);
                        match (worker + i) % 4 {
                            0 => {
                                let _ =
                                    store.set_with_ttl(&key, i as i64, Duration::from_millis(50));
                            }
                            1 => {
                                let _ = store.get(&key);
                            }
                            2 => {
                                let _ = store.cas(&key, &(i as i64 - 1), i as i64);
                            }
                            _ => {
                                let _ = store.delete(&key);
                            }
                        }
                    }
                });
            }
        });
        store.close();
    }
}
