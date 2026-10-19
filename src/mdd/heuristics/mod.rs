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
//! # Design notes
//!
//! - **Heuristics never see constraint semantics.** An order is repaired afterwards so that
//!   order-sensitive constraints such as `Regular` are satisfied; see `Mdd::new`.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::all_different;
//! use aicad::mdd::Mdd;
//! use aicad::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
//! use aicad::modelling::Problem;
//! use std::sync::Arc;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
//! let constraint = all_different(&mut problem, vars.clone());
//!
//! // Branch on the variables in reverse, and keep at most two nodes per layer.
//! let mut mdd = Mdd::new(
//!     Arc::new(problem),
//!     OrderingHeuristic::Custom(vec![3, 2, 1, 0]),
//!     MergeHeuristic::LessRelaxed,
//!     SelectHeuristic::Greedy,
//!     &[constraint],
//! );
//! mdd.refine(2);
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
