//! # aicad
//!
//! MDD compilation for constraint satisfaction problems, with strategies built on top of it:
//! belief propagation, local search, and local search guided by a trained neural network.
//!
//! Everything the Python package `pyaicad` does is available here too, for applications that
//! cannot afford the interpreter. The Rust API is **experimental** and may change between minor
//! versions until its users have shaped it.
//!
//! # Modules
//!
//! From the bottom up, each module only depends on the ones listed before it:
//!
//! - [`modelling`]: variables, the [`Problem`](modelling::Problem), and the traits that a
//!   constraint implements.
//! - [`constraints`]: the constraints, with the contract that the compiler relies on.
//! - [`mdd`]: the compiler and the compiled diagram.
//! - [`sampling`]: sampling solutions from a compiled diagram.
//! - [`bp`]: belief propagation over the diagrams of the constraints.
//! - [`destroy`]: the operators that choose which variables a local search frees.
//! - [`local_search`]: the local search engine, its architectures and [`local_search::solve`].
//! - [`diagnostics`]: an optional log of structured events for offline analysis.
//!
//! # Features
//!
//! - `learning` (off by default): the neural networks (`neural`), their training (`learning`)
//!   and the local searches that use them. It pulls in [burn](https://burn.dev), which is heavy to
//!   build; the compiler, sampling, belief propagation and the belief-propagation local search do
//!   not need it.
//! - `python` (off by default): the bindings built by maturin into `pyaicad`, and never needed from
//!   Rust. Combined with `learning` they give the full wheel; alone, the slim one.
//!
//! # Settings
//!
//! A few settings are global: the number of threads of the worker pool that runs the parallel parts
//! of the library ([`set_num_threads`], [`num_threads`]), a quarter of the available threads
//! unless set, and the seed of the randomness ([`set_seed`]).
//!
//! # Example
//!
//! ```
//! use aicad::constraints::all_different;
//! use aicad::mdd::Compiler;
//! use aicad::mdd::heuristics::CompilationHeuristic;
//! use aicad::modelling::Problem;
//! use std::sync::Arc;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2]);
//! let constraint = all_different(&mut problem, vars);
//!
//! let mut compiler = Compiler::new(
//!     Arc::new(problem),
//!     CompilationHeuristic::default(),
//!     &[constraint],
//! );
//! compiler.refine(usize::MAX, usize::MAX);
//! assert_eq!(compiler.into_mdd().count_solutions().to_string(), "6");
//! ```

#![cfg_attr(not(test), warn(unreachable_pub))]

pub mod bp;
pub mod build_info;
pub mod constraints;
pub mod destroy;
pub mod diagnostics;
pub mod local_search;
pub mod mdd;
pub mod modelling;
pub mod sampling;

#[cfg(feature = "learning")]
pub mod learning;
#[cfg(feature = "learning")]
pub mod neural;

#[cfg(feature = "python")]
mod pyaicad;

mod utils;

pub use build_info::{BuildInfo, version};
pub use utils::{default_num_threads, num_threads, set_num_threads};

/// Seeds every source of randomness of the crate, so that a later run of a compilation, a
/// training or a search reproduces itself given the same seed: the process-wide generator (the
/// sampling of diagrams, the destroy operators, the train and validation split, the masking of
/// training samples) and, with the `learning` feature, the generators of the burn backends (the
/// initial weights, the sampling of logits).
///
/// Compilation uses no randomness and local search is reproducible given its own seed; whether
/// a seeded training run reproduces itself bit for bit is not guaranteed.
pub fn set_seed(seed: u64) {
    utils::set_seed(seed);
    #[cfg(feature = "learning")]
    neural::seed_backends(seed);
}
