//! Crate-private helpers shared by the layers above.
//!
//! This module imports nothing else from the crate, so any layer may use it. It holds a compact
//! bit set for the per-node state of constraints ([`Bitset`]), the shared worker pool
//! ([`worker_pool`] and its public size setting [`set_num_threads`]), the process-wide seedable
//! randomness ([`with_rng`]) and the conversions between assignments and integer tensors.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub(crate) mod bitset;
pub(crate) mod parallel;
pub(crate) mod rng;
#[cfg(feature = "learning")]
pub(crate) mod tensor;

pub(crate) use bitset::Bitset;
pub(crate) use parallel::worker_pool;
pub use parallel::{default_num_threads, num_threads, set_num_threads};
pub(crate) use rng::set_seed;
pub(crate) use rng::with_rng;
