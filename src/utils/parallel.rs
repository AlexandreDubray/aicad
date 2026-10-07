use rayon::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

static WORKER_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn worker_pool() -> &'static ThreadPool {
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

pub fn run_pulled<T: Send>(
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
