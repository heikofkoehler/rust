# Production Rust Concurrency Suite: A Set of Interesting Concurrency Problems

The Rust counterpart to
[github.com/heikofkoehler/golang](https://github.com/heikofkoehler/golang):
a production-grade, race-free Rust implementation of a set of
interesting concurrency problems commonly encountered in
high-throughput distributed systems and systems engineering.

Every module addresses the exact concurrency traps, failure modes,
race conditions, and architectural criteria outlined in the problem
specifications, and is verified by the ported test suite
(`cargo test`). Rust has no `-race` flag to match Go's
`go test -race ./...`: instead, the ownership model and `Send`/`Sync`
checks rule out data races at compile time, and the tests stress the
same interleavings Go's race detector was run against.

Standard library only — no external crates.

---

## Architecture & Concurrency Traps Overview

### 1. The Concurrent Web Crawler (`src/crawler.rs`)
- **Core Problem:** Given a root URL and a high-latency fetch API, crawl unique internal links concurrently without infinite loops, deadlocks, or redundant network calls.
- **Trap 1: Termination Condition:** An empty task queue does **not** signify crawl completion, because in-flight workers may discover new links. Solved via a shared queue plus an in-flight counter: the crawl only terminates when the queue is empty **and** no worker is in flight.
- **Trap 2: Visited Set Race Conditions:** Naive `if !visited[url]` checks allow concurrent workers to fetch duplicate pages. Solved with atomic check-and-insert (`VisitedSet::try_add`) under a single lock.
- **Trap 3: Bounded Concurrency:** A fixed pool of `N` worker threads (`Config::max_workers`) prevents unconstrained thread creation.

### 2. The Multi-Worker Job / Image Processor (`src/jobprocessor.rs`)
- **Core Problem:** Asynchronous batch execution engine supporting task submissions (`submit`), inspection (`get_status`, `get_job`), and cooperative cancellation (`cancel_job`).
- **Trap 1: Lifecycle State Races:** Atomic transitions between `QUEUED`, `RUNNING`, `CANCELLED`, `COMPLETED`, and `FAILED`. When a worker pops a task, it validates that the status is `QUEUED` under the record lock; if cancelled while queued, it immediately aborts. If cancelled while running, the task's cancellation token is cancelled.
- **Trap 2: Failure Isolation:** Panics in jobs (e.g. corrupt images) are captured via `catch_unwind` (Go's `defer recover()`), marking the job `FAILED` without terminating the worker thread or stalling the pool.
- **Trap 3: Graceful Teardown:** `shutdown(wait=true)` stops accepting submissions and drains the queue; `shutdown(wait=false)` cancels queued and in-flight jobs, and workers exit cleanly.

### 3. Banking Ledger & Concurrent Account Transfers (`src/ledger.rs`)
- **Core Problem:** Multi-account banking ledger supporting deposits, withdrawals, and multi-party transfers under high concurrent load.
- **Trap 1: Deadlock Prevention via Lock Ordering:** Circular wait (`A → B` vs `B → A`) is eliminated by acquiring per-account locks in strict lexicographic account-ID order. Self-transfers are detected and rejected.
- **Trap 2: Balance Invariant Violations:** Fine-grained per-account locks prevent global bottlenecks while atomically verifying `balance >= amount` before debiting.
- **Trap 3: Atomic Rollback on Partial Failure:** In multi-leg transfers, net deltas are pre-validated before committing. If any balance invariant is violated, all modifications are aborted and the record is `ROLLED_BACK`.
- **Auditability:** Append-only immutable transaction log and `total_system_balance()` for conservation-of-money verification. Failed operations still return their record (as a `TxOutcome`), since Rust has no nil record pointer to return alongside an error the way Go returns `(*TransactionRecord, error)`.

### 4. Thread-Safe In-Memory Store with TTL (`src/ttlstore.rs`)
- **Core Problem:** In-memory key-value database with TTL expiration, CRUD, and atomic conditional writes.
- **Trap 1: Read-Write Contention:** Keys are partitioned across 32 (default) independent shards using 64-bit FNV-1a hashing. Each shard has an isolated `RwLock`, allowing concurrent reads and writes across different keys.
- **Trap 2: Active vs. Lazy Cleanup:**
  - *Lazy Cleanup:* Expired keys encountered on `get` are deleted on the fly.
  - *Active Cleanup:* A background reaper periodically sweeps shards one-by-one, inspecting bounded batches (`100` per shard) to eliminate lock starvation for active operations.
- **Trap 3: Atomic Operations:** Compare-And-Swap (`cas`) and Compare-And-Delete (`cad`) evaluate and mutate values inside the shard write lock, preventing race windows between read and write.

### 5. Distributed / Concurrent Rate Limiter (`src/ratelimiter.rs`)
- **Core Problem:** Token bucket and sliding window rate limiters to throttle API requests per tenant/user.
- **Trap 1: Clock Precision & Drift:** Refills are calculated analytically on-demand using Rust's monotonic clock (`Instant`). This prevents cumulative drift without requiring per-limiter background ticker threads.
- **Trap 2: Lock Contention & Burst Handling:** Token Bucket allows bursts up to capacity `C` and refills at rate `R`. Sliding Window provides an exact sliding log to eliminate end-of-window burst exploitation.
- **Trap 3: Multi-Tenant Sharding & Idle Eviction:** A sharded `Manager` partitions tenants across multiple shards with lock-free atomic timestamp updates on read paths, and safely evicts idle tenants to prevent memory leaks.

---

## Go Concurrency vs. Rust Concurrency

| Aspect | Go Runtime | Rust (this crate, std only) |
|---|---|---|
| **Execution Model** | M:N scheduler: goroutines multiplexed onto OS threads. | OS threads (`std::thread`), one per worker; no async runtime. |
| **Data races** | Detected at test time (`-race`). | Ruled out at compile time by ownership, borrowing, `Send`/`Sync`. |
| **Synchronization** | Channels, `select`, `sync.Mutex`, `sync.RWMutex`, `sync.WaitGroup`, `sync/atomic`, `context.Context`. | `mpsc` channels, `Mutex`, `RwLock`, `Condvar`, atomics, and the `CancelToken` in `src/cancel.rs` standing in for `context.Context`. |
| **Panic isolation** | `defer recover()` in the worker. | `catch_unwind` around the job execution. |
| **Coordination without `select`** | Coordinator `select`s over work and result channels. | Crawler workers share a queue + `Condvar` with an in-flight counter; the termination invariant (queue empty **and** idle) is identical. |
| **Memory footprint** | Goroutines start with a 2 KB resizable stack. | OS threads carry OS stacks; the worker pools are bounded, so thread count stays at `max_workers`. |

## Porting divergences (deliberate, documented in code)

- **Cancellation:** Go threads `context.Context` everywhere. This crate uses an explicit `CancelToken` (`src/cancel.rs`): a cloneable flag + condvar, so sleeps wake immediately on cancel. `jobprocessor::Engine::shutdown(wait)` has no context timeout parameter — it joins cooperative workers, as Go's does within its timeout.
- **Generic store values:** Go's TTL store holds `any` and compares with `reflect.DeepEqual`. Rust cannot compare `Box<dyn Any>` values, so `ttlstore::Store<V>` is generic over `V: Clone + PartialEq + Send + Sync`. A `Store<String>` and a `Store<i64>` are separate types.
- **Transaction IDs:** Go mints IDs from crypto-random bytes. The standard library has no RNG, so IDs combine a process-wide counter with the current time in nanoseconds — unique within a process, which is all an audit-log ID needs.
- **URL parsing:** Go's `net/url` is replaced by a small parser for the URL shapes this crawler handles (absolute and root-relative links). It matches Go's `NormalizeURL` on those shapes: lowercase scheme/host, fragment stripped, trailing slash trimmed on non-root paths.
- **Job results:** Go jobs return `any`; Rust jobs return `String`, which keeps snapshots cloneable. The tests only need a result payload.
- **Crawler link resolution:** Go resolves discovered links against the root URL; this port resolves relative links against the page that linked them (standard web behaviour). All ported tests use absolute links, so behaviour on the test suite is identical.

## Running the Verification Suite

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets
```

33 tests, all passing, clippy clean, no warnings.
