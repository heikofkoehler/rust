//! Concurrent web crawler.
//!
//! Port of Go `pkg/crawler`.
//!
//! Traps preserved:
//!
//! * Visited-set races: `VisitedSet::try_add` is one atomic
//!   check-and-insert under a single lock, so two workers can never
//!   both decide an unseen URL is theirs.
//! * Termination: an empty queue alone does *not* end the crawl.
//!   The crawl ends only when the queue is empty *and* no worker is
//!   in flight, because an in-flight fetch may still discover links.
//! * Bounded concurrency: exactly `max_workers` worker threads.
//!
//! Two divergences, both forced by the standard library:
//!
//! * Go's coordinator uses `select` over a work channel and a result
//!   channel. Rust's std channels have no `select`, so the workers
//!   share one queue guarded by a `Mutex` + `Condvar`, with an
//!   in-flight counter. The invariant — and the termination
//!   condition — is identical; only the waiting mechanism differs.
//! * Go's `net/url` is replaced by a small parser for the URL shapes
//!   this crawler deals with (`scheme://host/path`, absolute and
//!   root-relative links). It lowercases scheme and host, strips
//!   fragments, and trims a trailing slash on non-root paths,
//!   matching Go's `NormalizeURL` for those shapes.
//! * Cancellation uses [`CancelToken`][crate::cancel::CancelToken]
//!   in place of Go's `context.Context`.

use crate::cancel::CancelToken;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Crawler errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CrawlError {
    InvalidUrl(String),
    Cancelled,
}

impl fmt::Display for CrawlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrawlError::InvalidUrl(msg) => write!(f, "{msg}"),
            CrawlError::Cancelled => write!(f, "crawl cancelled"),
        }
    }
}

impl std::error::Error for CrawlError {}

/// Thread-safe check-and-add visited set.
#[derive(Debug, Default)]
pub struct VisitedSet {
    seen: RwLock<HashSet<String>>,
}

impl VisitedSet {
    pub fn new() -> VisitedSet {
        VisitedSet::default()
    }

    /// Atomically record `url` if absent. Returns `true` only for
    /// the first caller to claim the URL.
    pub fn try_add(&self, url: &str) -> bool {
        let mut seen = self.seen.write().unwrap_or_else(|e| e.into_inner());
        seen.insert(url.to_string())
    }

    pub fn has(&self, url: &str) -> bool {
        self.seen
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(url)
    }

    pub fn size(&self) -> usize {
        self.seen.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn elements(&self) -> Vec<String> {
        self.seen
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

/// Crawler configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Maximum concurrent fetches.
    pub max_workers: usize,
    /// Maximum depth from the root (0 = root only).
    pub max_depth: usize,
    /// Only follow links on the root's host.
    pub same_host_only: bool,
    /// Optional throttle between requests.
    pub delay: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_workers: 8,
            max_depth: 3,
            same_host_only: true,
            delay: Duration::ZERO,
        }
    }
}

/// The crawl data discovered for a single URL.
#[derive(Clone, Debug)]
pub struct PageResult {
    pub url: String,
    pub depth: usize,
    pub body: String,
    pub links: Vec<String>,
    pub error: Option<String>,
    pub discovered: SystemTime,
}

/// The complete outcome of a crawl.
#[derive(Clone, Debug)]
pub struct CrawlResult {
    pub root_url: String,
    pub visited: Vec<String>,
    pub pages: HashMap<String, PageResult>,
    pub errors: HashMap<String, String>,
    pub duration: Duration,
}

/// The fetching contract, mirroring Go's `Fetcher` interface.
pub trait Fetcher: Send + Sync {
    fn fetch(&self, cancel: &CancelToken, url: &str) -> Result<(String, Vec<String>), String>;
}

/// A mock page for tests.
#[derive(Clone, Debug)]
pub struct MockPage {
    pub body: String,
    pub urls: Vec<String>,
    pub error: Option<String>,
    pub delay: Duration,
}

impl MockPage {
    pub fn new(body: &str, urls: &[&str]) -> MockPage {
        MockPage {
            body: body.to_string(),
            urls: urls.iter().map(|s| s.to_string()).collect(),
            error: None,
            delay: Duration::ZERO,
        }
    }
}

#[derive(Debug, Default)]
struct MockState {
    call_counts: HashMap<String, usize>,
    max_in_flight: usize,
    cur_in_flight: usize,
}

/// A thread-safe mock fetcher, like Go's `MockFetcher`.
#[derive(Debug)]
pub struct MockFetcher {
    pages: HashMap<String, MockPage>,
    state: Mutex<MockState>,
}

impl MockFetcher {
    pub fn new(pages: HashMap<String, MockPage>) -> MockFetcher {
        MockFetcher {
            pages,
            state: Mutex::new(MockState::default()),
        }
    }

    pub fn call_count(&self, url: &str) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .call_counts
            .get(url)
            .copied()
            .unwrap_or(0)
    }

    pub fn total_calls(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .call_counts
            .values()
            .sum()
    }

    pub fn max_concurrent_fetches(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .max_in_flight
    }
}

impl Fetcher for MockFetcher {
    fn fetch(&self, cancel: &CancelToken, url: &str) -> Result<(String, Vec<String>), String> {
        let page = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.cur_in_flight += 1;
            state.max_in_flight = state.max_in_flight.max(state.cur_in_flight);
            *state.call_counts.entry(url.to_string()).or_insert(0) += 1;
            self.pages.get(url).cloned()
        };
        // Decrement on every exit path, like Go's deferred unlock.
        let result = (|| {
            let page = match page {
                Some(page) => page,
                None => return Err(format!("404 not found: {url}")),
            };
            if page.delay > Duration::ZERO && cancel.sleep(page.delay) {
                return Err("cancelled".to_string());
            }
            if let Some(err) = page.error {
                return Err(err);
            }
            Ok((page.body, page.urls))
        })();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.cur_in_flight -= 1;
        result
    }
}

// --- Minimal URL handling (see module docs) ---

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParsedUrl {
    scheme: String,
    host: String,
    path: String,
}

fn parse_url(raw: &str) -> Option<ParsedUrl> {
    let idx = raw.find("://")?;
    let (scheme, rest) = (raw[..idx].to_lowercase(), &raw[idx + 3..]);
    let (host, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };
    // Strip any fragment or query handling is out of scope for the
    // shapes used here; fragments are stripped by the caller path
    // via normalize_url before parsing the result.
    let path = path.split('#').next().unwrap_or(path);
    Some(ParsedUrl {
        scheme,
        host: host.to_lowercase(),
        path: path.to_string(),
    })
}

fn host_of(url: &str) -> String {
    parse_url(url).map(|p| p.host).unwrap_or_default()
}

fn render(parsed: &ParsedUrl) -> String {
    format!("{}://{}{}", parsed.scheme, parsed.host, parsed.path)
}

/// Clean and resolve a discovered URL against the base URL.
///
/// Strips fragments, lowercases scheme and host, resolves
/// root-relative links against the base, and trims a trailing slash
/// on non-root paths — Go's `NormalizeURL` behaviour for the URL
/// shapes this crawler handles.
pub fn normalize_url(base: &str, raw_url: &str) -> Result<String, CrawlError> {
    let raw_url = raw_url.trim();
    if raw_url.is_empty() {
        return Err(CrawlError::InvalidUrl("empty url".to_string()));
    }
    let raw_no_fragment = raw_url.split('#').next().unwrap_or(raw_url);
    let base_parsed = parse_url(base)
        .ok_or_else(|| CrawlError::InvalidUrl(format!("failed to parse base url {base:?}")))?;
    let mut parsed = if raw_no_fragment.contains("://") {
        parse_url(raw_no_fragment)
            .ok_or_else(|| CrawlError::InvalidUrl(format!("failed to parse url {raw_url:?}")))?
    } else if let Some(path) = raw_no_fragment.strip_prefix('/') {
        ParsedUrl {
            scheme: base_parsed.scheme.clone(),
            host: base_parsed.host.clone(),
            path: format!("/{path}"),
        }
    } else {
        // Relative link: resolve against the base path's directory.
        let dir = match base_parsed.path.rfind('/') {
            Some(idx) => &base_parsed.path[..=idx],
            None => "/",
        };
        ParsedUrl {
            scheme: base_parsed.scheme.clone(),
            host: base_parsed.host.clone(),
            path: format!("{dir}{raw_no_fragment}"),
        }
    };
    if parsed.path.len() > 1 && parsed.path.ends_with('/') {
        while parsed.path.len() > 1 && parsed.path.ends_with('/') {
            parsed.path.pop();
        }
    }
    if parsed.path.is_empty() {
        parsed.path = "/".to_string();
    }
    Ok(render(&parsed))
}

#[derive(Clone, Debug)]
struct QueueTask {
    url: String,
    depth: usize,
}

#[derive(Debug)]
struct SharedQueue {
    queue: Mutex<VecDeque<QueueTask>>,
    changed: Condvar,
    active: Mutex<usize>,
    done: Mutex<bool>,
}

impl SharedQueue {
    fn new() -> SharedQueue {
        SharedQueue {
            queue: Mutex::new(VecDeque::new()),
            changed: Condvar::new(),
            active: Mutex::new(0),
            done: Mutex::new(false),
        }
    }
}

/// A concurrent web crawler.
pub struct Crawler {
    fetcher: Arc<dyn Fetcher>,
    config: Config,
}

impl Crawler {
    pub fn new(fetcher: Arc<dyn Fetcher>, config: Config) -> Crawler {
        let max_workers = if config.max_workers == 0 {
            1
        } else {
            config.max_workers
        };
        Crawler {
            fetcher,
            config: Config {
                max_workers,
                ..config
            },
        }
    }

    /// Crawl from `start_url` until the queue is empty and no worker
    /// is in flight, or until `cancel` fires.
    pub fn crawl(&self, cancel: &CancelToken, start_url: &str) -> Result<CrawlResult, CrawlError> {
        let started = Instant::now();
        let normalized_start = normalize_url(start_url, start_url)
            .map_err(|_| CrawlError::InvalidUrl(format!("invalid root URL {start_url:?}")))?;
        let base_host = host_of(&normalized_start);

        let visited = Arc::new(VisitedSet::new());
        if !visited.try_add(&normalized_start) {
            return Err(CrawlError::InvalidUrl(
                "failed to seed root URL into visited set".to_string(),
            ));
        }
        let pages: Arc<Mutex<HashMap<String, PageResult>>> = Arc::new(Mutex::new(HashMap::new()));
        let errors: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let shared = Arc::new(SharedQueue::new());
        {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.push_back(QueueTask {
                url: normalized_start.clone(),
                depth: 0,
            });
        }

        let mut handles = Vec::with_capacity(self.config.max_workers);
        for _ in 0..self.config.max_workers {
            let fetcher = Arc::clone(&self.fetcher);
            let visited = Arc::clone(&visited);
            let pages = Arc::clone(&pages);
            let errors = Arc::clone(&errors);
            let shared = Arc::clone(&shared);
            let cancel = cancel.clone();
            let config = self.config;
            let base_host = base_host.clone();
            handles.push(std::thread::spawn(move || {
                worker(
                    fetcher, visited, pages, errors, shared, cancel, config, &base_host,
                )
            }));
        }

        // Coordinator: wait for queue-empty AND active == 0.
        let cancelled = loop {
            if cancel.is_cancelled() {
                break true;
            }
            let queue_empty = shared
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty();
            let active = *shared.active.lock().unwrap_or_else(|e| e.into_inner());
            if queue_empty && active == 0 {
                break false;
            }
            // Wake periodically so cancellation is noticed promptly
            // even if no worker finishes (e.g. a hung fetch that
            // itself honours the token will finish and notify; the
            // timeout is the backstop).
            let queue_guard = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            let (guard, _) = shared
                .changed
                .wait_timeout(queue_guard, Duration::from_millis(5))
                .unwrap_or_else(|e| e.into_inner());
            drop(guard);
        };
        // Tell workers to stop, then join them.
        *shared.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
        shared.changed.notify_all();
        for handle in handles {
            let _ = handle.join();
        }

        let mut result = CrawlResult {
            root_url: normalized_start,
            visited: visited.elements(),
            pages: pages.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            errors: errors.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            duration: started.elapsed(),
        };
        // Keep visited deterministic for callers that compare sets:
        // the order itself is unspecified in Go too (map iteration).
        result.visited.sort();
        if cancelled {
            return Err(CrawlError::Cancelled);
        }
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)]
fn worker(
    fetcher: Arc<dyn Fetcher>,
    visited: Arc<VisitedSet>,
    pages: Arc<Mutex<HashMap<String, PageResult>>>,
    errors: Arc<Mutex<HashMap<String, String>>>,
    shared: Arc<SharedQueue>,
    cancel: CancelToken,
    config: Config,
    base_host: &str,
) {
    loop {
        let task = {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(task) = queue.pop_front() {
                    break task;
                }
                if *shared.done.lock().unwrap_or_else(|e| e.into_inner()) {
                    return;
                }
                // Wait for work or shutdown. Holding the queue lock
                // here is what makes pop_front atomic with the wait.
                // The coordinator's notify_all wakes us on new work
                // and on shutdown; the timeout is the backstop.
                let (guard, _) = shared
                    .changed
                    .wait_timeout(queue, Duration::from_millis(5))
                    .unwrap_or_else(|e| e.into_inner());
                queue = guard;
                if *shared.done.lock().unwrap_or_else(|e| e.into_inner()) {
                    return;
                }
                if cancel.is_cancelled() {
                    return;
                }
            }
        };
        {
            let mut active = shared.active.lock().unwrap_or_else(|e| e.into_inner());
            *active += 1;
        }
        if config.delay > Duration::ZERO {
            cancel.sleep(config.delay);
        }
        let outcome = if cancel.is_cancelled() {
            Err("cancelled".to_string())
        } else {
            fetcher.fetch(&cancel, &task.url)
        };
        // Process the outcome under the queue lock so enqueueing
        // discovered links and decrementing active are atomic with
        // respect to the coordinator's termination check.
        let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        match outcome {
            Err(err) => {
                errors
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(task.url.clone(), err.clone());
                pages.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    task.url.clone(),
                    PageResult {
                        url: task.url.clone(),
                        depth: task.depth,
                        body: String::new(),
                        links: Vec::new(),
                        error: Some(err),
                        discovered: SystemTime::now(),
                    },
                );
            }
            Ok((body, raw_links)) => {
                let mut valid_links = Vec::new();
                for raw_link in raw_links {
                    let normalized = match normalize_url(&task.url, &raw_link) {
                        Ok(url) => url,
                        Err(_) => continue,
                    };
                    if config.same_host_only && host_of(&normalized) != base_host {
                        continue;
                    }
                    valid_links.push(normalized.clone());
                    if task.depth < config.max_depth && visited.try_add(&normalized) {
                        queue.push_back(QueueTask {
                            url: normalized,
                            depth: task.depth + 1,
                        });
                    }
                }
                pages.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    task.url.clone(),
                    PageResult {
                        url: task.url.clone(),
                        depth: task.depth,
                        body,
                        links: valid_links,
                        error: None,
                        discovered: SystemTime::now(),
                    },
                );
            }
        }
        {
            let mut active = shared.active.lock().unwrap_or_else(|e| e.into_inner());
            *active -= 1;
        }
        shared.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::thread;

    fn pages_map(entries: &[(&str, MockPage)]) -> HashMap<String, MockPage> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn basic_and_cycle_handling() {
        let fetcher = Arc::new(MockFetcher::new(pages_map(&[
            (
                "http://example.com/",
                MockPage::new("root", &["http://example.com/b", "http://example.com/c"]),
            ),
            (
                "http://example.com/b",
                MockPage::new("page b", &["http://example.com/", "http://example.com/d"]),
            ),
            (
                "http://example.com/c",
                MockPage::new("page c", &["http://example.com/b"]),
            ),
            (
                "http://example.com/d",
                MockPage::new("page d", &["http://example.com/c"]),
            ),
        ])));
        let crawler = Crawler::new(
            fetcher.clone(),
            Config {
                max_workers: 4,
                max_depth: 10,
                same_host_only: true,
                delay: Duration::ZERO,
            },
        );
        let result = crawler
            .crawl(&CancelToken::new(), "http://example.com/")
            .unwrap();
        assert_eq!(result.pages.len(), 4);
        for url in [
            "http://example.com/",
            "http://example.com/b",
            "http://example.com/c",
            "http://example.com/d",
        ] {
            assert_eq!(fetcher.call_count(url), 1, "url {url} fetched exactly once");
        }
    }

    #[test]
    fn termination_with_lagging_workers() {
        let mut slow = MockPage::new(
            "slow page",
            &[
                "http://test.com/delayed-child-1",
                "http://test.com/delayed-child-2",
            ],
        );
        slow.delay = Duration::from_millis(150);
        let mut root = MockPage::new("root", &["http://test.com/slow"]);
        root.delay = Duration::from_millis(10);
        let mut child1 = MockPage::new("child 1", &[]);
        child1.delay = Duration::from_millis(20);
        let mut child2 = MockPage::new("child 2", &[]);
        child2.delay = Duration::from_millis(20);
        let fetcher = Arc::new(MockFetcher::new(pages_map(&[
            ("http://test.com/", root),
            ("http://test.com/slow", slow),
            ("http://test.com/delayed-child-1", child1),
            ("http://test.com/delayed-child-2", child2),
        ])));
        let crawler = Crawler::new(
            fetcher.clone(),
            Config {
                max_workers: 4,
                max_depth: 5,
                same_host_only: true,
                delay: Duration::ZERO,
            },
        );
        let result = crawler
            .crawl(&CancelToken::new(), "http://test.com/")
            .unwrap();
        assert_eq!(result.pages.len(), 4);
        assert_eq!(fetcher.call_count("http://test.com/delayed-child-1"), 1);
        assert_eq!(fetcher.call_count("http://test.com/delayed-child-2"), 1);
    }

    #[test]
    fn bounded_concurrency() {
        let mut entries: Vec<(String, MockPage)> = Vec::new();
        let mut root_links: Vec<String> = Vec::new();
        for i in 0..50 {
            let url = format!("http://bounded.com/page/{i}");
            root_links.push(url.clone());
            let mut page = MockPage::new(&format!("content {i}"), &[]);
            page.delay = Duration::from_millis(25);
            entries.push((url, page));
        }
        entries.push((
            "http://bounded.com/".to_string(),
            MockPage::new(
                "root",
                &root_links.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ),
        ));
        let fetcher = Arc::new(MockFetcher::new(entries.into_iter().collect()));
        let crawler = Crawler::new(
            fetcher.clone(),
            Config {
                max_workers: 5,
                max_depth: 2,
                same_host_only: true,
                delay: Duration::ZERO,
            },
        );
        let result = crawler
            .crawl(&CancelToken::new(), "http://bounded.com/")
            .unwrap();
        assert_eq!(result.pages.len(), 51);
        assert!(
            fetcher.max_concurrent_fetches() <= 5,
            "concurrency bound violated: {} > 5",
            fetcher.max_concurrent_fetches()
        );
    }

    #[test]
    fn context_cancellation() {
        let mut root = MockPage::new("root", &["http://slow.com/hang"]);
        root.delay = Duration::from_millis(10);
        let mut hang = MockPage::new("hanging", &[]);
        hang.delay = Duration::from_secs(5);
        let fetcher = Arc::new(MockFetcher::new(pages_map(&[
            ("http://slow.com/", root),
            ("http://slow.com/hang", hang),
        ])));
        let crawler = Crawler::new(
            fetcher,
            Config {
                max_workers: 2,
                max_depth: 3,
                same_host_only: true,
                delay: Duration::ZERO,
            },
        );
        let token = CancelToken::new();
        let canceller = token.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            canceller.cancel();
        });
        assert!(crawler.crawl(&token, "http://slow.com/").is_err());
    }

    #[test]
    fn visited_set_concurrent_safety() {
        let set = Arc::new(VisitedSet::new());
        // Borrowed as references (which are Copy) so scoped threads
        // can share the counters without moving them.
        let successes = &AtomicI64::new(0);
        let duplicates = &AtomicI64::new(0);
        thread::scope(|s| {
            for _ in 0..100 {
                let set = Arc::clone(&set);
                s.spawn(move || {
                    if set.try_add("http://race.com/target") {
                        successes.fetch_add(1, Ordering::SeqCst);
                    } else {
                        duplicates.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(successes.load(Ordering::SeqCst), 1);
        assert_eq!(duplicates.load(Ordering::SeqCst), 99);
        assert_eq!(set.size(), 1);
    }

    #[test]
    fn normalize_url_shapes() {
        assert_eq!(
            normalize_url("http://example.com/a/b", "http://EXAMPLE.com/c/").unwrap(),
            "http://example.com/c"
        );
        assert_eq!(
            normalize_url("http://example.com/a/b", "/c#frag").unwrap(),
            "http://example.com/c"
        );
        assert_eq!(
            normalize_url("http://example.com/", "http://example.com/").unwrap(),
            "http://example.com/"
        );
        assert!(normalize_url("http://example.com/", "").is_err());
    }
}
