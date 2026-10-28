//! The worker pool shared by the crate's CPU parallelism.
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
//! # Design notes
//!
//! - **The default is a quarter of the available threads, and at least one.** The library is run
//!   on shared servers, where taking every core would slow down other users, so using more has
//!   to be asked for.
//! - **A new size replaces the pool, it does not resize it.** Changing the size drops the
//!   current pool and the next [`worker_pool`] call builds another. Work that already holds the
//!   old pool (it is shared, `Arc`) runs to completion on it, so setting the number of threads
//!   is safe from any thread at any moment, and the call order of a script does not matter.
//! - **Zero threads is refused.** rayon reads 0 as "one thread per core", the opposite of the
//!   cautious default, so a 0 that slipped through (an unset option, say) would take the whole
//!   machine. [`set_num_threads`] panics with a message instead; the Python function checks it
//!   first and raises `ValueError`.
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

#[cfg(test)]
mod tests {
    use super::*;

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
