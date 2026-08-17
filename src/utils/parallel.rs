use rayon::ThreadPool;
use std::sync::OnceLock;

static WORKER_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn worker_pool() -> &'static ThreadPool {
    WORKER_POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_threads = available.saturating_div(4).max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build capped-size rayon worker pool")
    })
}
