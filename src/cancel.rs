//! A minimal cancellation token, the Rust counterpart to Go's
//! `context.Context` cancellation for this crate.
//!
//! Go code threads a `context.Context` through fetchers, jobs and
//! limiter waits so a timeout or an explicit cancel can interrupt a
//! blocking operation. Rust's standard library has no context type, so
//! this crate uses an explicit, cheaply cloneable token instead:
//! one `Arc` holding a flag plus a condition variable, so waits can
//! wake the moment cancellation lands instead of polling on a timer.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A cloneable cancellation signal.
///
/// Clones share the same flag: cancelling any clone cancels all of
/// them. This mirrors a Go context derived from a parent — cancelling
/// the engine cancels every job derived from it.
#[derive(Clone, Debug)]
pub struct CancelToken {
    inner: Arc<CancelInner>,
}

#[derive(Debug)]
struct CancelInner {
    cancelled: Mutex<bool>,
    changed: Condvar,
}

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        CancelToken {
            inner: Arc::new(CancelInner {
                cancelled: Mutex::new(false),
                changed: Condvar::new(),
            }),
        }
    }

    /// A token that is already cancelled.
    pub fn cancelled() -> Self {
        let token = Self::new();
        token.cancel();
        token
    }

    /// Signal cancellation to every holder of this token.
    pub fn cancel(&self) {
        let mut flag = self
            .inner
            .cancelled
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !*flag {
            *flag = true;
            self.inner.changed.notify_all();
        }
    }

    /// Whether cancellation has been signalled.
    pub fn is_cancelled(&self) -> bool {
        *self
            .inner
            .cancelled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Wait up to `duration` for cancellation.
    ///
    /// Returns `true` if cancellation arrived during the wait,
    /// `false` if the full duration elapsed without it. This is the
    /// building block for interruptible sleeps (`time.After` +
    /// `<-ctx.Done()` in Go).
    pub fn wait_timeout(&self, duration: Duration) -> bool {
        let flag = self
            .inner
            .cancelled
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *flag {
            return true;
        }
        let (flag, _timeout) = self
            .inner
            .changed
            .wait_timeout(flag, duration)
            .unwrap_or_else(|e| e.into_inner());
        *flag
    }

    /// Sleep for `duration`, returning early on cancellation.
    ///
    /// Returns `true` if cancelled, `false` if the sleep completed.
    pub fn sleep(&self, duration: Duration) -> bool {
        self.wait_timeout(duration)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn cancel_wakes_a_waiter() {
        let token = CancelToken::new();
        let waiter = token.clone();
        let handle = thread::spawn(move || waiter.wait_timeout(Duration::from_secs(5)));
        thread::sleep(Duration::from_millis(20));
        token.cancel();
        assert!(handle.join().unwrap());
        assert!(token.is_cancelled());
    }

    #[test]
    fn sleep_completes_without_cancel() {
        let token = CancelToken::new();
        assert!(!token.sleep(Duration::from_millis(10)));
    }
}
