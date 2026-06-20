//! A small, dependency-free persistent worker pool for the frame decoder.
//!
//! Motivation: the parallel decode used to spawn a fresh set of OS threads with
//! `std::thread::scope` for every frame (twice — once for the decode phase and
//! once for the filter phase). For a still-image / repeated-decode workload that
//! per-frame `pthread_create` cost sits right on the critical path. A persistent
//! pool spawns the workers once and reuses them across every frame, the way a
//! C decoder's task pool does.
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

/// A fixed set of worker threads spawned once and reused for every dispatch.
///
/// The dispatch state sits behind a `Mutex`, which serializes dispatches (at most
/// one `run` in flight per pool, so completion signals never cross) and makes the
/// pool `Send + Sync` so a decoder can own one across its frames.
pub(crate) struct ThreadPool {
    inner: Mutex<Inner>,
    n: usize,
    handles: Vec<Option<JoinHandle<()>>>,
}

struct Inner {
    txs: Vec<Sender<Msg>>,
    done_rx: Receiver<bool>,
}

impl ThreadPool {
    /// Spawns `n` (>= 1) persistent worker threads.
    pub(crate) fn new(n: usize) -> Self {
        let n = n.max(1);
        let (done_tx, done_rx) = channel::<bool>();
        let mut txs = Vec::with_capacity(n);
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = channel::<Msg>();
            let done_tx = done_tx.clone();
            let handle = std::thread::spawn(move || {
                loop {
                    match rx.recv() {
                        Ok(Msg::Run(JobPtr(job))) => {
                            // SAFETY: the dispatcher blocks until it has received our
                            // completion signal below, so `job` is valid for this call.
                            let job: &(dyn Fn() + Sync) = unsafe { &*job };
                            let ok =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_ok();
                            // Signal completion last. Report whether we panicked so the
                            // dispatcher can propagate it (rather than deadlock) while
                            // this worker thread stays alive for future dispatches.
                            let _ = done_tx.send(ok);
                        }
                        Ok(Msg::Stop) | Err(_) => break,
                    }
                }
            });
            txs.push(tx);
            handles.push(Some(handle));
        }
        Self {
            inner: Mutex::new(Inner { txs, done_rx }),
            n,
            handles,
        }
    }

    /// Number of worker threads (the maximum `n_active` that adds parallelism).
    pub(crate) fn workers(&self) -> usize {
        self.n
    }

    /// Runs `job` on `n_active` threads total: `n_active - 1` pool workers plus
    /// the calling thread, then blocks until the workers have finished. `n_active`
    /// is clamped to `[1, workers() + 1]`.
    ///
    /// This is a scoped dispatch: `job` may borrow `'env` stack data, and that
    /// borrow is guaranteed live until `run` returns.
    pub(crate) fn run<'env>(&self, n_active: usize, job: &(dyn Fn() + Sync + 'env)) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let helpers = n_active.min(self.n + 1).saturating_sub(1);
        // SAFETY: erase `'env` to hand the reference to the `'static` workers. We
        // block on `done_rx` for every helper below before returning, and a worker
        // only signals after it has finished calling `job`; therefore no worker
        // retains the reference past this call, so it never outlives `'env`.
        let job_erased: *const (dyn Fn() + Sync) =
            unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + 'env), _>(job) };
        for tx in inner.txs.iter().take(helpers) {
            let _ = tx.send(Msg::Run(JobPtr(job_erased)));
        }
        // The caller participates so it is never idle while helpers run.
        job();
        let mut panicked = false;
        for _ in 0..helpers {
            if let Ok(false) = inner.done_rx.recv() { panicked = true }
        }
        if panicked {
            panic!("a tealdust pool worker panicked");
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if let Ok(inner) = self.inner.lock() {
            for tx in &inner.txs {
                let _ = tx.send(Msg::Stop);
            }
        }
        for h in &mut self.handles {
            if let Some(h) = h.take() {
                let _ = h.join();
            }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
