//! The worker pool shared by the crate's CPU parallelism, and a scheduler for independent tasks.
//!
//! # Meaning
//!
//! [`worker_pool`] returns the process-wide rayon pool, built on first use. The training and
//! inference code (batched solving, destroy operators, solution checks, building the diagrams of
//! a batch) runs its parallel loops on it with `worker_pool().install(|| ...)`, so the number of
//! threads these tasks may use is decided in one place. Compiling a single diagram is
//! single-threaded and does not use the pool; several diagrams can be compiled at the same time,
//! one per task.
//!
//! The size of the pool is the setting [`num_threads`]. Unless [`set_num_threads`] has been
//! called it is [`default_num_threads`], a quarter of the available threads and at least one.
//! `set_num_threads(n)` can be called at any time, before or after the pool has been used.
//!
//! [`run_pulled`]`(pool, total, max_workers, task)` calls `task(i)` once for every
//! $i \in 0..total$ and returns the results in index order. At most `max_workers` tasks run at
//! the same time (never more than the pool has threads, and always at least one).
//!
//! # How it works
//!
//! The workers share an atomic counter. Each one takes the next unstarted index as soon as it
//! finishes its current task, and writes the result into the slot of that index. The call returns
//! once all workers are done.
//!
//! # Design notes
//!
//! - **The default is a quarter of the available threads, and at least one.** The library is run
//!   on shared servers, where taking every core would slow down other users, so using more has
//!   to be asked for. The setting is the size of the pool and nothing else: how many chunks of a
//!   batch are in flight at once is bounded separately (by device memory, not by cores).
//! - **A new size replaces the pool, it does not resize it.** Changing the size drops the
//!   current pool and the next [`worker_pool`] call builds another. Work that already holds the
//!   old pool (it is shared, `Arc`) runs to completion on it, so setting the number of threads
//!   is safe from any thread at any moment, and the call order of a script does not matter.
//! - **Zero threads is refused.** rayon reads 0 as "one thread per core", the opposite of the
//!   cautious default, so a 0 that slipped through (an unset option, say) would take the whole
//!   machine. [`set_num_threads`] panics with a message instead; the Python function checks it
//!   first and raises `ValueError`.
//! - **Tasks are pulled, not split in advance.** Tasks can take very different times (one
//!   hard instance among easy ones), so a worker that finishes early takes the next task instead
//!   of waiting for a share decided up front.
//! - **Results are returned by index, not by completion order.** Callers derive seeds from the
//!   index, so a run gives the same results whatever the scheduler does.
//!
//! # Example
//!
//! ```
//! assert!(aicad::num_threads() >= 1);
//!
//! aicad::set_num_threads(2);
//! assert_eq!(aicad::num_threads(), 2);
//! ```

use rayon::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// The setting and the pool built from it.
struct PoolState {
    /// The size asked for with `set_num_threads`; `None` means `default_num_threads`.
    requested: Option<usize>,
    /// The pool of the current size, built by the first `worker_pool` call that needs it. Empty
    /// again whenever the size changes.
    pool: Option<Arc<ThreadPool>>,
}

/// The state behind `num_threads` and `worker_pool`.
static POOL_STATE: Mutex<PoolState> = Mutex::new(PoolState {
    requested: None,
    pool: None,
});

/// The state, usable even if a thread panicked while holding it: it is always left consistent.
fn lock_state() -> MutexGuard<'static, PoolState> {
    POOL_STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The number of threads the pool has unless `set_num_threads` is called: a quarter of the
/// available threads, and at least one.
pub fn default_num_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    available.saturating_div(4).max(1)
}

/// The number of threads of the worker pool: the last value given to `set_num_threads`, or
/// `default_num_threads` if there is none.
pub fn num_threads() -> usize {
    lock_state().requested.unwrap_or_else(default_num_threads)
}

/// Sets the number of threads of the worker pool, which runs the parallel parts of batched
/// solving, training and dataset building. It applies from the next parallel loop on; loops
/// already running finish with the threads they had.
///
/// # Panics
///
/// If `n` is 0, which rayon would read as "one thread per core".
pub fn set_num_threads(n: usize) {
    assert!(
        n >= 1,
        "invalid number of threads: 0 was given, but the worker pool needs at least one thread"
    );
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if n > available {
        log::warn!(
            "{n} threads were requested but only {available} are available; the extra threads will compete for the same cores"
        );
    }
    let mut state = lock_state();
    state.requested = Some(n);
    if state
        .pool
        .as_ref()
        .is_some_and(|pool| pool.current_num_threads() != n)
    {
        state.pool = None;
    }
}

/// The process-wide pool, of `num_threads()` threads, built on the first call that needs it. The
/// caller keeps the pool it got for as long as it holds it, even if `set_num_threads` replaces
/// it meanwhile.
pub(crate) fn worker_pool() -> Arc<ThreadPool> {
    let mut state = lock_state();
    let num_threads = state.requested.unwrap_or_else(default_num_threads);
    let pool = state.pool.get_or_insert_with(|| {
        log::info!("Creating thread pool with {num_threads} threads");
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build the rayon worker pool"),
        )
    });
    Arc::clone(pool)
}

/// Calls `task(i)` for every `i` in `0..total` on `pool` and returns the results in index order.
/// At most `max_workers` run at once (never more than the pool has threads, and at least one);
/// each worker takes the next unstarted index as soon as it is free.
pub(crate) fn run_pulled<T: Send>(
    pool: &ThreadPool,
    total: usize,
    max_workers: usize,
    task: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let workers = max_workers
        .min(total)
        .min(pool.current_num_threads())
        .max(1);
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..total).map(|_| Mutex::new(None)).collect();
    pool.scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|_| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    let result = task(idx);
                    *slots[idx].lock().expect("result slot lock poisoned") = Some(result);
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("result slot lock poisoned")
                .expect("every index below total is run exactly once")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    fn pool(threads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn results_come_back_in_index_order() {
        let out = run_pulled(&pool(4), 50, 4, |i| i * 2);
        assert_eq!(out, (0..50).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn every_index_runs_exactly_once() {
        let counter = AtomicUsize::new(0);
        let out = run_pulled(&pool(3), 40, 8, |i| {
            counter.fetch_add(1, Ordering::SeqCst);
            i
        });
        assert_eq!(counter.load(Ordering::SeqCst), 40);
        assert_eq!(out.len(), 40);
    }

    #[test]
    fn a_slow_task_does_not_hold_back_later_ones() {
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let done_rx = Mutex::new(done_rx);
        let done_tx = Mutex::new(done_tx);
        let out = run_pulled(&pool(2), 4, 2, |i| {
            if i == 0 {
                for _ in 0..3 {
                    done_rx.lock().unwrap().recv().unwrap();
                }
            } else {
                done_tx.lock().unwrap().send(()).unwrap();
            }
            i
        });
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn worker_count_is_capped_by_max_workers() {
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        run_pulled(&pool(4), 20, 2, |_| {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(5));
            live.fetch_sub(1, Ordering::SeqCst);
        });
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn the_pool_follows_the_setting() {
        let before = num_threads();
        let default = default_num_threads();
        assert!(default >= 1);

        set_num_threads(3);
        assert_eq!(num_threads(), 3);
        let old = worker_pool();
        assert_eq!(old.current_num_threads(), 3);

        set_num_threads(2);
        let new = worker_pool();
        assert_eq!(new.current_num_threads(), 2);
        // The pool taken before the change is still alive and usable.
        assert_eq!(old.current_num_threads(), 3);
        assert_eq!(old.install(|| 7), 7);

        set_num_threads(before);
    }

    #[test]
    #[should_panic(expected = "invalid number of threads")]
    fn zero_threads_is_refused() {
        set_num_threads(0);
    }
}
