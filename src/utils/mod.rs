//! Crate-private helpers shared by the layers above.
//!
//! This module imports nothing else from the crate, so any layer may use it. It holds a compact
//! bit set for the per-node state of constraints ([`Bitset`]), the shared worker pool
//! ([`worker_pool`], [`run_pulled`]), the process-wide seedable randomness ([`with_rng`]) and
//! the conversions between assignments and integer tensors.

pub(crate) mod bitset;
pub(crate) mod parallel;
pub(crate) mod rng;
pub(crate) mod tensor;

pub(crate) use bitset::Bitset;
pub(crate) use parallel::{run_pulled, worker_pool};
pub(crate) use rng::with_rng;
