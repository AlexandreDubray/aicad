//! Shared helper for CPU-bound work that wants rayon parallelism without claiming every core.
//!
//! Using rayon's default global thread pool (one worker per core) is the fastest option, but it's
//! not good behaviour here: this crate is often run on a shared benchmarking server, where
//! saturating every core is inconsiderate of other jobs, and even on a personal machine, leaving
//! zero headroom means the process (and, in the middle of a training loop, the operator's ability
//! to Ctrl+C it) can end up starved by the pool it just spawned. `worker_pool` reserves at least
//! one core for the rest of the system instead.

/// Builds a dedicated rayon thread pool sized to one less than the machine's available
/// parallelism (minimum one thread), rather than using rayon's all-cores-by-default global pool.
/// Callers should run their `par_iter`/`into_par_iter` work inside `.install(...)` on the
/// returned pool; any parallel work spawned from within that closure (including nested
/// `par_iter` calls in functions it calls) automatically stays on the same, capped pool.
pub fn worker_pool() -> rayon::ThreadPool {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let num_threads = available.saturating_div(4).max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build capped-size rayon worker pool")
}
