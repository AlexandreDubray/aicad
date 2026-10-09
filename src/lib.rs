//! # aicad
//!
//! MDD compilation for constraint satisfaction and optimisation problems, with strategies built
//! on top of it (belief propagation, neural local search, learned priors).
//!
//! The supported interface of aicad is its Python package, `pyaicad`. The Rust API below is the
//! *core*: it is public so that other solvers can embed the compiler, but it is **experimental**
//! and may change between minor versions until its users have shaped it.
//!
//! The core is, from the bottom up:
//!
//! - [`modelling`]: variables, the [`Problem`](modelling::Problem), and the traits that a
//!   constraint implements.
//! - [`constraints`]: the constraints, with the contract that the compiler relies on.
//! - [`mdd`]: the compiler and the compiled diagram.
//! - [`sampling`]: belief propagation on a compiled diagram.
//!
//! Everything above the core (`learning`, `nls`, the Python bindings) and the helpers below it
//! are private. Modules only depend on the ones listed before them, which `scripts/check_layering.py`
//! checks in CI.

pub mod constraints;
pub mod mdd;
pub mod modelling;
pub mod sampling;

mod diagnostics;
mod learning;
mod nls;
mod pyaicad;
mod utils;
