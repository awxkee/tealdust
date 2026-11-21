//! A small, dependency-free persistent worker pool for the frame decoder.
//!
//! Motivation: the parallel decode used to spawn a fresh set of OS threads with
//! `std::thread::scope` for every frame (twice — once for the decode phase and
//! once for the filter phase). For a still-image / repeated-decode workload that
//! per-frame `pthread_create` cost sits right on the critical path. A persistent
//! pool starts workers as dispatches need them and reuses them across frames.
//!
//! Like `std::thread::scope`, [`ThreadPool::run`] is a *scoped* dispatch: it
//! broadcasts a borrowed job to the workers, runs it on the caller too, and does
//! not return until every worker has finished. Because the borrow is therefore
//! live for the whole dispatch, the job may reference per-frame stack data
//! directly — no owning, cloning, or copying of the frame buffers is needed. The
//! single `unsafe` is the lifetime erasure required to hand a `'env` reference to
//! the `'static` worker threads; it is sound precisely because `run` joins all
//! workers before returning, so no reference can outlive `'env` (the same
//! argument that makes `std::thread::scope` sound).

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

/// A type-erased, lifetime-erased pointer to the caller's job. The pointer is
/// only ever dereferenced by a worker while the dispatching `run` call is blocked
/// on that worker's completion, so the referent (which lives on the caller's
/// stack for at least that long) is always valid. `Send` is sound for the same
/// reason: the job is `Sync`, and the pointer never escapes the `run` call.
struct JobPtr(*const (dyn Fn() + Sync));
unsafe impl Send for JobPtr {}

enum Msg {
    Run(JobPtr),
    Stop,
}

/// A pool of persistent workers that grows up to its configured capacity.
///
/// The dispatch state sits behind a `Mutex`, which serializes dispatches (at most
/// one `run` in flight per pool, so completion signals never cross) and makes the
/// pool `Send + Sync` so a decoder can own one across its frames.
pub(crate) struct ThreadPool {
    inner: Mutex<Inner>,
    capacity: usize,
}

struct Inner {
    workers: Vec<Worker>,
    done_tx: Sender<bool>,
    done_rx: Receiver<bool>,
}

struct Worker {
    tx: Sender<Msg>,
    handle: JoinHandle<()>,
}

impl Worker {
    fn spawn(done_tx: Sender<bool>) -> std::io::Result<Self> {
        let (tx, rx) = channel::<Msg>();
        let handle = std::thread::Builder::new().spawn(move || {
            while let Ok(Msg::Run(JobPtr(job))) = rx.recv() {
                // SAFETY: the dispatcher waits for our completion below, so
                // `job` remains valid throughout this call.
                let job: &(dyn Fn() + Sync) = unsafe { &*job };
                let ok = match catch_unwind(AssertUnwindSafe(job)) {
                    Ok(()) => true,
                    Err(payload) => {
                        // A custom panic payload can panic on drop. Signal the
                        // failed job even in that case and keep the worker alive.
                        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
                            std::mem::forget(payload);
                        }
                        false
                    }
                };
                let _ = done_tx.send(ok);
            }
        })?;
        Ok(Self { tx, handle })
    }
}

impl ThreadPool {
    /// Configures up to `n` (>= 1) persistent workers.
    pub(crate) fn new(n: usize) -> Self {
        let (done_tx, done_rx) = channel::<bool>();
        Self {
            inner: Mutex::new(Inner {
                workers: Vec::new(),
                done_tx,
                done_rx,
            }),
            capacity: n.max(1),
        }
    }

    /// Maximum number of helpers, excluding the calling thread.
    pub(crate) fn workers(&self) -> usize {
        self.capacity
    }

    /// Runs `job` on `n_active` threads total: `n_active - 1` pool workers plus
    /// the calling thread, then blocks until the workers have finished. `n_active`
    /// is clamped to `[1, workers() + 1]`.
    ///
    /// This is a scoped dispatch: `job` may borrow `'env` stack data, and that
    /// borrow is guaranteed live until `run` returns.
    pub(crate) fn run<'env>(&self, n_active: usize, job: &(dyn Fn() + Sync + 'env)) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let helpers = n_active.saturating_sub(1).min(self.capacity);
        // Start every required helper before sending any borrowed jobs. If a
        // spawn fails, the workers already started are idle and can be reused.
        while inner.workers.len() < helpers {
            let worker = Worker::spawn(inner.done_tx.clone())
                .expect("failed to spawn a tealdust pool worker");
            inner.workers.push(worker);
        }
        // SAFETY: erase `'env` to hand the reference to the `'static` workers. We
        // wait for every submitted helper before returning or propagating a
        // panic. Every worker signals completion after it stops using `job`.
        let job_erased: *const (dyn Fn() + Sync) =
            unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + 'env), _>(job) };
        let mut submitted = 0;
        for worker in inner.workers.iter().take(helpers) {
            if worker.tx.send(Msg::Run(JobPtr(job_erased))).is_err() {
                break;
            }
            submitted += 1;
        }
        let caller_result = if submitted == helpers {
            catch_unwind(AssertUnwindSafe(job))
        } else {
            Ok(())
        };
        let mut worker_failed = submitted != helpers;
        for _ in 0..submitted {
            worker_failed |= !matches!(inner.done_rx.recv(), Ok(true));
        }
        drop(inner);
        if let Err(payload) = caller_result {
            resume_unwind(payload);
        }
        assert!(!worker_failed, "a tealdust pool worker stopped or panicked");
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        for worker in &inner.workers {
            let _ = worker.tx.send(Msg::Stop);
        }
        for worker in inner.workers.drain(..) {
            let _ = worker.handle.join();
        }
    }
}

/// Dispatches `job` across `n_active` threads, using `pool` when one is provided
/// and falling back to a per-call `std::thread::scope` otherwise. The job is the
/// same borrowed closure either way; only the thread source differs.
pub(crate) fn dispatch<'env>(
    pool: Option<&ThreadPool>,
    n_active: usize,
    job: &(dyn Fn() + Sync + 'env),
) {
    match pool {
        Some(p) => p.run(n_active, job),
        None => {
            std::thread::scope(|scope| {
                for _ in 1..n_active {
                    scope.spawn(job);
                }
                job();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // Workers + caller cooperatively drain a shared atomic cursor; every index is
    // handled exactly once and the pool is reusable across dispatches.
    #[test]
    fn pool_dynamic_fill_is_correct_and_reusable() {
        let pool = ThreadPool::new(4);
        for _ in 0..3 {
            let n = 1000usize;
            let out: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let cursor = AtomicUsize::new(0);
            let job = || loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                out[i].store(i * i, Ordering::Relaxed);
            };
            pool.run(pool.workers() + 1, &job);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(v.load(Ordering::Relaxed), i * i);
            }
        }
    }

    // n_active == 1 runs entirely on the caller (no workers engaged).
    #[test]
    fn pool_single_active_runs_on_caller() {
        let pool = ThreadPool::new(4);
        let hit = AtomicUsize::new(0);
        let job = || {
            hit.fetch_add(1, Ordering::Relaxed);
        };
        pool.run(1, &job);
        assert_eq!(hit.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn pool_starts_and_reuses_only_needed_helpers() {
        let pool = ThreadPool::new(4);
        assert!(pool.inner.lock().unwrap().workers.is_empty());
        pool.run(1, &|| {});
        assert!(pool.inner.lock().unwrap().workers.is_empty());

        pool.run(2, &|| {});
        let first = {
            let inner = pool.inner.lock().unwrap();
            assert_eq!(inner.workers.len(), 1);
            inner.workers[0].handle.thread().id()
        };
        pool.run(4, &|| {});
        pool.run(2, &|| {});
        let inner = pool.inner.lock().unwrap();
        assert_eq!(inner.workers.len(), 3);
        assert_eq!(inner.workers[0].handle.thread().id(), first);
    }

    #[test]
    fn pool_waits_for_helpers_before_propagating_a_caller_panic() {
        let pool = ThreadPool::new(1);
        let caller = std::thread::current().id();
        let started = Barrier::new(2);
        let finished = AtomicBool::new(false);
        let job = || {
            started.wait();
            assert_ne!(std::thread::current().id(), caller, "caller job panicked");
            std::thread::sleep(std::time::Duration::from_millis(10));
            finished.store(true, Ordering::Relaxed);
        };
        let result = catch_unwind(AssertUnwindSafe(|| pool.run(2, &job)));
        assert!(result.is_err());
        assert!(finished.load(Ordering::Relaxed));
        pool.run(2, &|| {});
    }

    // `dispatch` with no pool falls back to thread::scope and is still correct.
    #[test]
    fn dispatch_scope_fallback() {
        let n = 500usize;
        let out: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let cursor = AtomicUsize::new(0);
        let job = || loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            out[i].store(i + 1, Ordering::Relaxed);
        };
        dispatch(None, 4, &job);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.load(Ordering::Relaxed), i + 1);
        }
    }
}
