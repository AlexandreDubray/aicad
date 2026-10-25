//! The choices that steer MDD compilation: the order of the layers, which node to split, and how
//! to merge nodes.
//!
//! # Purpose
//!
//! Compiling a relaxed MDD involves three decisions that change its shape, and so its quality, but
//! never its soundness (every solution stays a path):
//!
//! - [`OrderingHeuristic`] chooses the variable branched on at each layer.
//! - [`SelectHeuristic`] chooses, during refinement, the node to split next.
//! - [`MergeHeuristic`] chooses which nodes are merged when a layer is wider than allowed.
//!
//! [`CompilationHeuristic`] groups the three, and is what `Compiler::new` takes.
//!
//! # Design notes
//!
//! - **Heuristics never see constraint semantics.** An order is repaired afterwards so that
//!   order-sensitive constraints such as `Regular` are satisfied; see `Compiler::new`.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::all_different;
//! use aicad::mdd::Compiler;
//! use aicad::mdd::heuristics::{
//!     CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
//! };
//! use aicad::modelling::Problem;
//! use std::sync::Arc;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
//! let constraint = all_different(&mut problem, vars.clone());
//!
//! // Branch on the variables in reverse, and keep at most two nodes per layer.
//! let mut compiler = Compiler::new(
//!     Arc::new(problem),
//!     CompilationHeuristic {
//!         ordering: OrderingHeuristic::Custom(vec![3, 2, 1, 0]),
//!         merge: MergeHeuristic::LessRelaxed,
//!         selection: SelectHeuristic::Greedy,
//!     },
//!     &[constraint],
//! );
//! compiler.refine(2);
//! let mdd = compiler.into_mdd();
//!
//! assert_eq!(mdd.decision_at_layer(0), vars[3]);
//! assert!((0..4).all(|layer| mdd.number_nodes_in_layer(layer) <= 2));
//! ```

pub mod merge;
pub mod ordering;
pub mod select;

pub use merge::MergeHeuristic;
pub use ordering::OrderingHeuristic;
pub use select::SelectHeuristic;

/// The three choices that steer the compilation of one MDD. See the [module documentation](self).
#[derive(Clone, Debug)]
pub struct CompilationHeuristic {
    /// The variable branched on at each layer. Only used by `Compiler::new`, which fixes the order.
    pub ordering: OrderingHeuristic,
    /// Which nodes are merged when a layer is wider than allowed, during `Compiler::refine`.
    pub merge: MergeHeuristic,
    /// Which node is split next, during `Compiler::refine`.
    pub selection: SelectHeuristic,
}
