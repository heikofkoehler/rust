//! Production Rust concurrency suite: a set of interesting
//! concurrency problems.
//!
//! The Rust counterpart to
//! [github.com/heikofkoehler/golang](https://github.com/heikofkoehler/golang),
//! which implements the same five problems in Go. Every module
//! addresses the same concurrency traps, failure modes and race
//! conditions, verified by the ported test suite
//! (`cargo test` — Rust has no `-race` flag; its ownership model
//! and `Send`/`Sync` checks rule out data races at compile time,
//! and the tests stress the same interleavings Go's race detector
//! was run against).
//!
//! * [`crawler`] — concurrent web crawler with atomic visited set
//!   and queue-empty-plus-idle termination.
//! * [`jobprocessor`] — multi-worker job engine with atomic
//!   lifecycle transitions, panic isolation and graceful teardown.
//! * [`ledger`] — banking ledger with lock-ordered, atomic
//!   transfers and multi-leg rollback.
//! * [`ttlstore`] — sharded in-memory store with TTL, lazy and
//!   active expiry, and atomic CAS/CAD.
//! * [`ratelimiter`] — token bucket, sliding window, and a sharded
//!   multi-tenant manager with idle eviction.
//!
//! The [`cancel::CancelToken`] is the stand-in for Go's
//! `context.Context` cancellation throughout the crate.

pub mod cancel;
pub mod crawler;
pub mod jobprocessor;
pub mod ledger;
pub mod ratelimiter;
pub mod ttlstore;
