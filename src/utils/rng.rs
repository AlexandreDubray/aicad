//! Process-wide randomness that can be seeded to make a whole run reproducible.
//!
//! # Meaning
//!
//! [`with_rng`] runs a closure against a random number generator. Until [`set_seed`] has been
//! called it is a fresh, entropy-seeded `rand::rng()`; after `set_seed(seed)`, every call on any
//! thread draws from one shared generator started from `seed`. Calling `set_seed` again restarts
//! the sequence, whatever was drawn before.
//!
//! Call sites that do not receive an explicit seed (the shuffle of the training set, the random
//! masks of the classical training recipe, the initial assignment drawn by `Variable::sample`)
//! use `with_rng` instead of `rand::rng()`. Everything that takes a `seed` argument, such as the
//! neural local search, keeps using its own generator.
//!
//! The seed does not reach the network's backend (the Python-facing `set_seed` seeds it next to
//! this generator) nor the tie-breaking generator of the MDD compiler, which each thread seeds
//! from the operating system the first time it compiles.
//!
//! # How it works
//!
//! The shared generator is a `StdRng` behind a mutex in a `OnceLock`. Before the first
//! `set_seed` the cell is empty and `with_rng` takes no lock.
//!
//! # Design notes
//!
//! - **Reproducible only if the order of draws is.** Threads that draw from the shared generator
//!   concurrently get values in the order they happen to lock it. Parallel code therefore draws
//!   what it needs sequentially first (for example one seed per task) and hands it to the tasks.
//! - **Unseeded behaviour is unchanged.** A process that never calls `set_seed` behaves as if
//!   this module did not exist.

use std::sync::{Mutex, OnceLock};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The shared generator, empty until `set_seed` is first called.
static GLOBAL_RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

/// (Re)seeds the process-wide RNG that `with_rng` draws from. Safe to call more than once (e.g.
/// between successive runs in the same process): each call replaces the shared state with a
/// fresh `StdRng::seed_from_u64(seed)`, so the next `with_rng` call after `set_seed` always starts
/// a new, deterministic sequence regardless of how much randomness a previous run consumed.
pub(crate) fn set_seed(seed: u64) {
    let seeded = StdRng::seed_from_u64(seed);
    match GLOBAL_RNG.get() {
        Some(existing) => *existing.lock().unwrap() = seeded,
        None => {
            // If another thread's `with_rng` call raced us to initialize the cell first (with
            // its own entropy-seeded fallback -- see below), `set()` loses the race but that's
            // fine: `existing` above would then be `Some`, so a concurrent `set_seed` and
            // `with_rng` at process start can only ever land on the `None` branch once.
            let _ = GLOBAL_RNG.set(Mutex::new(seeded));
        }
    }
}

/// Runs `f` against the process-wide RNG if `set_seed` has been called at least once, or a fresh
/// entropy-seeded `rand::rng()` otherwise -- i.e. this crate's behaviour before `set_seed`
/// existed. Every call site that wants its randomness to become reproducible under `set_seed`
/// should draw from here instead of calling `rand::rng()` directly.
pub(crate) fn with_rng<T>(f: impl FnOnce(&mut dyn Rng) -> T) -> T {
    match GLOBAL_RNG.get() {
        Some(shared) => f(&mut *shared.lock().unwrap()),
        None => f(&mut rand::rng()),
    }
}
