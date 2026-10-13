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
//! - **The pool has a quarter of the available threads, and at least one.** The library is run
//!   on shared servers, where taking every core would slow down other users; the pool is the one
//!   place that limits it. The size is fixed when the pool is first used and cannot be set by the
//!   caller.
//! - **Tasks are pulled, not split in advance.** Tasks can take very different times (one
//!   hard instance among easy ones), so a worker that finishes early takes the next task instead
//!   of waiting for a share decided up front.
//! - **Results are returned by index, not by completion order.** Callers derive seeds from the
//!   index, so a run gives the same results whatever the scheduler does.

use rayon::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

static WORKER_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub(crate) fn worker_pool() -> &'static ThreadPool {
    WORKER_POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_threads = available.saturating_div(4).max(1);
        log::info!("Creating thread pool with  {} threads", num_threads);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build capped-size rayon worker pool")
    })
}

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
}
