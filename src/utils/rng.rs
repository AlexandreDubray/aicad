//! Process-wide, optionally-seeded randomness.
//!
//! Most of this crate's randomness is already explicitly seeded where it matters for a single
//! call (e.g. `neural_local_search`'s `seed` parameter feeds a local `StdRng`). What's left --
//! the classical training recipe's per-batch variable masking, the train/validation shuffle, and
//! `Variable::sample`'s initial-assignment draw -- goes through plain `rand::rng()` (a
//! non-reseedable, OS-entropy-seeded thread-local), so there was previously no way to make a
//! whole training/search run reproduce bit-for-bit.
//!
//! `with_rng` is a drop-in replacement for `rand::rng()` at those call sites: before `set_seed` is
//! ever called it behaves identically (fresh OS entropy, no locking), but once `set_seed(seed)`
//! runs, every subsequent `with_rng` call -- on any thread -- draws from the same seeded `StdRng`
//! instead. Combined with seeding `burn`'s own backend RNG (see `pyaicad::learn::set_seed`, which
//! calls both this and `Backend::seed`), this covers the two independently-seedable randomness
//! sources in the crate. It does not cover `mdd`'s internal tie-breaking RNG, which is seeded
//! lazily, once per thread, from OS entropy the first time that thread compiles an MDD -- see that
//! module's `RNG` thread-local for why threading a global seed through it isn't as
//! straightforward.

use std::sync::{Mutex, OnceLock};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

static GLOBAL_RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

/// (Re)seeds the process-wide RNG that `with_rng` draws from. Safe to call more than once (e.g.
/// between successive runs in the same process): each call replaces the shared state with a
/// fresh `StdRng::seed_from_u64(seed)`, so the next `with_rng` call after `set_seed` always starts
/// a new, deterministic sequence regardless of how much randomness a previous run consumed.
pub fn set_seed(seed: u64) {
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
pub fn with_rng<T>(f: impl FnOnce(&mut dyn Rng) -> T) -> T {
    match GLOBAL_RNG.get() {
        Some(shared) => f(&mut *shared.lock().unwrap()),
        None => f(&mut rand::rng()),
    }
}
