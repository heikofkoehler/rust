//! Asynchronous batch job execution engine.
//!
//! Port of Go `pkg/jobprocessor`.
//!
//! Traps preserved:
//!
//! * Lifecycle state races: a worker only runs a job it atomically
//!   moved from `QUEUED` to `RUNNING`; a job cancelled while queued
//!   is skipped, and one cancelled while running has its token
//!   cancelled so cooperative jobs abort.
//! * Failure isolation: a panicking job is caught (Go's
//!   `defer recover()` becomes Rust's `catch_unwind`), marked
//!   `FAILED`, and the worker survives for the next job.
//! * Graceful teardown: `shutdown(wait=true)` drains the queue,
//!   `shutdown(wait=false)` cancels queued and in-flight jobs.
//!
//! Divergences, all forced by the standard library:
//!
//! * Go's `context.Context` is the [`CancelToken`][crate::cancel::CancelToken]
//!   here; jobs receive `&CancelToken` instead of a context.
//! * Go's `Shutdown(ctx, wait)` takes a context for a join timeout.
//!   This `shutdown(wait)` has no timeout and joins cooperative
//!   workers until they finish; a job that ignores cancellation
//!   would block it, exactly as it would block Go's workers.
//! * Go jobs return `any`; here a job returns `String`. The tests
//!   only need a result payload, and a string keeps snapshots
//!   cloneable without a `Box<dyn Any>` downcast dance.

use crate::cancel::CancelToken;
use std::collections::HashMap;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

/// Engine and job errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobError {
    EngineClosed,
    JobNotFound,
    JobAlreadyExists,
    JobCancelled,
    InvalidState(String),
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobError::EngineClosed => write!(f, "job engine is shut down"),
            JobError::JobNotFound => write!(f, "job not found"),
            JobError::JobAlreadyExists => write!(f, "job ID already exists"),
            JobError::JobCancelled => write!(f, "job was cancelled"),
            JobError::InvalidState(msg) => write!(f, "invalid job state transition: {msg}"),
        }
    }
}

impl std::error::Error for JobError {}

/// Lifecycle states of a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::Queued => "QUEUED",
            Status::Running => "RUNNING",
            Status::Completed => "COMPLETED",
            Status::Failed => "FAILED",
            Status::Cancelled => "CANCELLED",
        };
        write!(f, "{s}")
    }
}

/// The executable workload.
pub trait Job: Send + Sync {
    fn id(&self) -> &str;
    fn execute(&self, cancel: &CancelToken) -> Result<String, String>;
}

/// The closure type behind [`FuncJob`], factored out so the field
/// type stays readable.
pub type JobFn = Arc<dyn Fn(&CancelToken) -> Result<String, String> + Send + Sync>;

/// Adapter letting a plain closure act as a job, like Go's `FuncJob`.
pub struct FuncJob {
    id: String,
    func: JobFn,
}

impl FuncJob {
    pub fn new<F>(id: &str, func: F) -> FuncJob
    where
        F: Fn(&CancelToken) -> Result<String, String> + Send + Sync + 'static,
    {
        FuncJob {
            id: id.to_string(),
            func: Arc::new(func),
        }
    }
}

impl Job for FuncJob {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(&self, cancel: &CancelToken) -> Result<String, String> {
        (self.func)(cancel)
    }
}

/// Image transform types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageOperation {
    Resize,
    Grayscale,
    Blur,
}

impl fmt::Display for ImageOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ImageOperation::Resize => "RESIZE",
            ImageOperation::Grayscale => "GRAYSCALE",
            ImageOperation::Blur => "BLUR",
        };
        write!(f, "{s}")
    }
}

/// A simulated batch image-processing job.
pub struct ImageJob {
    id: String,
    pub image_path: String,
    pub operation: ImageOperation,
    pub simulated_ms: Duration,
    should_panic: bool,
}

impl ImageJob {
    pub fn new(id: &str, image_path: &str, operation: ImageOperation, delay: Duration) -> ImageJob {
        ImageJob {
            id: id.to_string(),
            image_path: image_path.to_string(),
            operation,
            simulated_ms: delay,
            should_panic: false,
        }
    }

    /// Test helper mirroring Go's same-package struct literal with
    /// `ShouldPanic: true`.
    pub fn with_panic(mut self) -> ImageJob {
        self.should_panic = true;
        self
    }
}

impl Job for ImageJob {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(&self, cancel: &CancelToken) -> Result<String, String> {
        if self.should_panic {
            panic!(
                "fatal segmentation fault processing image {}",
                self.image_path
            );
        }
        if cancel.sleep(self.simulated_ms) {
            return Err("job was cancelled".to_string());
        }
        Ok(format!(
            "processed {} with op {}",
            self.image_path, self.operation
        ))
    }
}

#[derive(Debug)]
struct RecordState {
    status: Status,
    result: Option<String>,
    error: Option<String>,
    created_at: SystemTime,
    started_at: Option<SystemTime>,
    completed_at: Option<SystemTime>,
    started_instant: Option<Instant>,
    completed_instant: Option<Instant>,
}

/// Runtime metadata and state for one submitted job.
pub struct JobRecord {
    job: Arc<dyn Job>,
    state: Mutex<RecordState>,
    cancel: CancelToken,
}

impl JobRecord {
    fn new(job: Arc<dyn Job>) -> JobRecord {
        JobRecord {
            job,
            state: Mutex::new(RecordState {
                status: Status::Queued,
                result: None,
                error: None,
                created_at: SystemTime::now(),
                started_at: None,
                completed_at: None,
                started_instant: None,
                completed_instant: None,
            }),
            cancel: CancelToken::new(),
        }
    }

    /// A point-in-time copy of the record, like Go's `Snapshot()`.
    pub fn snapshot(&self) -> JobSnapshot {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let duration = match (state.started_instant, state.completed_instant) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        };
        JobSnapshot {
            id: self.job.id().to_string(),
            status: state.status,
            result: state.result.clone(),
            error: state.error.clone(),
            created_at: state.created_at,
            started_at: state.started_at,
            completed_at: state.completed_at,
            duration,
        }
    }
}

/// A snapshot of a job's state and execution metrics.
#[derive(Clone, Debug)]
pub struct JobSnapshot {
    pub id: String,
    pub status: Status,
    pub result: Option<String>,
    pub error: Option<String>,
    pub created_at: SystemTime,
    pub started_at: Option<SystemTime>,
    pub completed_at: Option<SystemTime>,
    pub duration: Duration,
}

/// Engine configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_workers: usize,
    pub queue_capacity: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_workers: 4,
            queue_capacity: 128,
        }
    }
}

/// A multi-worker job execution engine.
pub struct Engine {
    jobs: RwLock<HashMap<String, Arc<JobRecord>>>,
    queue: Mutex<Option<SyncSender<Arc<JobRecord>>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    closed: AtomicBool,
}

impl Engine {
    pub fn new(config: Config) -> Arc<Engine> {
        let num_workers = if config.num_workers > 0 {
            config.num_workers
        } else {
            4
        };
        let queue_capacity = if config.queue_capacity > 0 {
            config.queue_capacity
        } else {
            128
        };
        let (sender, receiver) = sync_channel::<Arc<JobRecord>>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let engine = Arc::new(Engine {
            jobs: RwLock::new(HashMap::new()),
            queue: Mutex::new(Some(sender)),
            workers: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        });
        let mut workers = engine.workers.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..num_workers {
            let receiver = Arc::clone(&receiver);
            workers.push(std::thread::spawn(move || worker_loop(receiver)));
        }
        drop(workers);
        engine
    }

    /// Enqueue a job. Rejects duplicates and submissions after
    /// shutdown, as in Go.
    pub fn submit(&self, job: Arc<dyn Job>) -> Result<(), JobError> {
        if job.id().is_empty() {
            return Err(JobError::InvalidState(
                "job and job ID must not be empty".to_string(),
            ));
        }
        let record = {
            let mut jobs = self.jobs.write().unwrap_or_else(|e| e.into_inner());
            if self.closed.load(Ordering::SeqCst) {
                return Err(JobError::EngineClosed);
            }
            if jobs.contains_key(job.id()) {
                return Err(JobError::JobAlreadyExists);
            }
            let record = Arc::new(JobRecord::new(job));
            jobs.insert(record.job.id().to_string(), Arc::clone(&record));
            record
        };
        // Hold the queue lock across the send so shutdown cannot
        // close the channel underneath us: either we send first, or
        // we observe the closed sender and report EngineClosed.
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        match queue.as_ref() {
            Some(sender) => sender.send(record).map_err(|_| JobError::EngineClosed),
            None => Err(JobError::EngineClosed),
        }
    }

    /// Attempt to cancel a job by ID.
    ///
    /// Returns `Ok(true)` if the job was queued or running and is now
    /// cancelled, `Ok(false)` if it was already terminal, and an
    /// error if the job does not exist.
    pub fn cancel_job(&self, id: &str) -> Result<bool, JobError> {
        let record = self
            .jobs
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
            .ok_or(JobError::JobNotFound)?;
        let mut state = record.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.status {
            Status::Completed | Status::Failed | Status::Cancelled => Ok(false),
            Status::Queued => {
                state.status = Status::Cancelled;
                state.error = Some(JobError::JobCancelled.to_string());
                state.completed_at = Some(SystemTime::now());
                state.completed_instant = Some(Instant::now());
                record.cancel.cancel();
                Ok(true)
            }
            Status::Running => {
                state.status = Status::Cancelled;
                state.error = Some(JobError::JobCancelled.to_string());
                record.cancel.cancel();
                Ok(true)
            }
        }
    }

    pub fn get_status(&self, id: &str) -> Result<Status, JobError> {
        let record = self
            .jobs
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
            .ok_or(JobError::JobNotFound)?;
        let status = record
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .status;
        Ok(status)
    }

    pub fn get_job(&self, id: &str) -> Result<JobSnapshot, JobError> {
        let record = self
            .jobs
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
            .ok_or(JobError::JobNotFound)?;
        Ok(record.snapshot())
    }

    /// Graceful teardown.
    ///
    /// With `wait=true`, queued jobs drain and running jobs finish.
    /// With `wait=false`, queued jobs are marked cancelled, in-flight
    /// jobs get their tokens cancelled, and workers exit as jobs
    /// abort. Submitting after shutdown returns `EngineClosed`.
    pub fn shutdown(&self, wait: bool) -> Result<(), JobError> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        if !wait {
            let jobs = self.jobs.read().unwrap_or_else(|e| e.into_inner());
            for record in jobs.values() {
                record.cancel.cancel();
                let mut state = record.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.status == Status::Queued {
                    state.status = Status::Cancelled;
                    state.error = Some(JobError::JobCancelled.to_string());
                    state.completed_at = Some(SystemTime::now());
                    state.completed_instant = Some(Instant::now());
                }
            }
        }
        // Close the queue: workers exit once it drains.
        *self.queue.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let handles = std::mem::take(&mut *self.workers.lock().unwrap_or_else(|e| e.into_inner()));
        for handle in handles {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Let workers exit if shutdown was never called; do not join
        // from Drop, which could block on a job this thread owns.
        *self.queue.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

fn worker_loop(receiver: Arc<Mutex<Receiver<Arc<JobRecord>>>>) {
    loop {
        let record = {
            let guard = receiver.lock().unwrap_or_else(|e| e.into_inner());
            guard.recv()
        };
        let record = match record {
            Ok(record) => record,
            Err(_) => return, // Channel closed: engine is shutting down.
        };
        // Atomic pop-vs-cancel check, as in Go.
        let should_run = {
            let mut state = record.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.status == Status::Queued {
                state.status = Status::Running;
                state.started_at = Some(SystemTime::now());
                state.started_instant = Some(Instant::now());
                true
            } else {
                false
            }
        };
        if !should_run {
            continue; // Cancelled while waiting in the queue.
        }
        execute_job_safely(&record);
    }
}

fn execute_job_safely(record: &Arc<JobRecord>) {
    let outcome = catch_unwind(AssertUnwindSafe(|| record.job.execute(&record.cancel)));
    let mut state = record.state.lock().unwrap_or_else(|e| e.into_inner());
    // If cancelled during execution, keep CANCELLED, as in Go.
    if state.status == Status::Cancelled {
        state.completed_at = Some(SystemTime::now());
        state.completed_instant = Some(Instant::now());
        return;
    }
    state.completed_at = Some(SystemTime::now());
    state.completed_instant = Some(Instant::now());
    match outcome {
        Ok(Ok(result)) => {
            state.status = Status::Completed;
            state.result = Some(result);
        }
        Ok(Err(err)) => {
            if record.cancel.is_cancelled() {
                state.status = Status::Cancelled;
                state.error = Some(JobError::JobCancelled.to_string());
            } else {
                state.status = Status::Failed;
                state.error = Some(err);
            }
        }
        Err(panic) => {
            let message = if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            state.status = Status::Failed;
            state.error = Some(format!("job crashed with panic: {message}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicI32};
    use std::thread;

    fn wait_for_status(engine: &Engine, id: &str, want: Status) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match engine.get_status(id) {
                Ok(status) if status == want => return,
                Ok(_) => {}
                Err(e) => panic!("status error for {id}: {e}"),
            }
            if Instant::now() > deadline {
                panic!("job {id} did not reach {want} in time");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn basic_job_execution() {
        let engine = Engine::new(Config {
            num_workers: 3,
            queue_capacity: 10,
        });
        let job = Arc::new(ImageJob::new(
            "img-1",
            "path/to/cat.png",
            ImageOperation::Resize,
            Duration::from_millis(20),
        ));
        engine.submit(job).unwrap();
        wait_for_status(&engine, "img-1", Status::Completed);
        let snapshot = engine.get_job("img-1").unwrap();
        assert_eq!(snapshot.status, Status::Completed);
        assert!(snapshot.result.is_some());
        engine.shutdown(true).unwrap();
    }

    #[test]
    fn lifecycle_cancel_queued() {
        let engine = Engine::new(Config {
            num_workers: 1,
            queue_capacity: 10,
        });
        let slow = Arc::new(FuncJob::new("slow", |cancel| {
            if cancel.sleep(Duration::from_millis(300)) {
                return Err("cancelled".to_string());
            }
            Ok("done".to_string())
        }));
        engine.submit(slow).unwrap();
        thread::sleep(Duration::from_millis(20));
        let executed = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&executed);
        let queued = Arc::new(FuncJob::new("queued-to-cancel", move |_| {
            flag.store(true, Ordering::SeqCst);
            Ok("should not run".to_string())
        }));
        engine.submit(queued).unwrap();
        assert_eq!(
            engine.get_status("queued-to-cancel").unwrap(),
            Status::Queued
        );
        assert!(engine.cancel_job("queued-to-cancel").unwrap());
        assert_eq!(
            engine.get_status("queued-to-cancel").unwrap(),
            Status::Cancelled
        );
        thread::sleep(Duration::from_millis(350));
        assert!(
            !executed.load(Ordering::SeqCst),
            "cancelled job must never run"
        );
        engine.shutdown(false).unwrap();
    }

    #[test]
    fn lifecycle_cancel_running() {
        let engine = Engine::new(Config {
            num_workers: 2,
            queue_capacity: 10,
        });
        let started = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&started);
        let running = Arc::new(FuncJob::new("running-cancel", move |cancel| {
            flag.store(true, Ordering::SeqCst);
            if cancel.sleep(Duration::from_secs(2)) {
                return Err("cancelled".to_string());
            }
            Ok("finished".to_string())
        }));
        engine.submit(running).unwrap();
        while !started.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            engine.get_status("running-cancel").unwrap(),
            Status::Running
        );
        assert!(engine.cancel_job("running-cancel").unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            engine.get_status("running-cancel").unwrap(),
            Status::Cancelled
        );
        assert!(!engine.cancel_job("running-cancel").unwrap());
        engine.shutdown(false).unwrap();
    }

    #[test]
    fn failure_isolation() {
        let engine = Engine::new(Config {
            num_workers: 1,
            queue_capacity: 10,
        });
        let panic_job = Arc::new(
            ImageJob::new(
                "panic-image",
                "corrupt.raw",
                ImageOperation::Resize,
                Duration::ZERO,
            )
            .with_panic(),
        );
        engine.submit(panic_job).unwrap();
        wait_for_status(&engine, "panic-image", Status::Failed);
        let snapshot = engine.get_job("panic-image").unwrap();
        assert!(snapshot.error.is_some());
        // The worker must still be alive for the next job.
        let healthy = Arc::new(ImageJob::new(
            "healthy-image",
            "normal.png",
            ImageOperation::Grayscale,
            Duration::from_millis(10),
        ));
        engine.submit(healthy).unwrap();
        wait_for_status(&engine, "healthy-image", Status::Completed);
        engine.shutdown(true).unwrap();
    }

    #[test]
    fn graceful_teardown_wait() {
        let engine = Engine::new(Config {
            num_workers: 2,
            queue_capacity: 10,
        });
        let count = Arc::new(AtomicI32::new(0));
        for i in 0..5 {
            let count = Arc::clone(&count);
            let job = Arc::new(FuncJob::new(&format!("drain-{i}"), move |_| {
                thread::sleep(Duration::from_millis(30));
                count.fetch_add(1, Ordering::SeqCst);
                Ok("ok".to_string())
            }));
            engine.submit(job).unwrap();
        }
        engine.shutdown(true).unwrap();
        let after = Arc::new(FuncJob::new("after-shutdown", |_| Ok(String::new())));
        assert_eq!(engine.submit(after).unwrap_err(), JobError::EngineClosed);
        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn high_concurrency_stress() {
        let engine = Engine::new(Config {
            num_workers: 8,
            queue_capacity: 200,
        });
        thread::scope(|s| {
            for i in 0..100 {
                let engine = Arc::clone(&engine);
                s.spawn(move || {
                    let id = format!("stress-{i}");
                    let job = Arc::new(FuncJob::new(&id, |_| {
                        thread::sleep(Duration::from_millis(5));
                        Ok("done".to_string())
                    }));
                    let _ = engine.submit(job);
                    if id == "stress-10" || id == "stress-25" || id == "stress-50" {
                        let _ = engine.cancel_job(&id);
                    }
                    let _ = engine.get_status(&id);
                    let _ = engine.get_job(&id);
                });
            }
        });
        engine.shutdown(false).unwrap();
    }
}
