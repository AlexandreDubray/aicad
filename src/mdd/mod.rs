//! Multi-valued decision diagrams (MDDs) of constraints: compilation, relaxation and queries.
//!
//! # Purpose
//!
//! An MDD stores the solutions of a constraint problem as the paths of a layered graph: one layer
//! per variable, an edge for each value a variable may take, and a single root and sink. This
//! module compiles the constraints of a [`Problem`](crate::modelling::Problem) into such a
//! diagram, with a limit on the width of the layers if wanted, and answers questions on it:
//! counting and sampling solutions, and weighted model counting (`wmc`) for the learning and
//! sampling code.
//!
//! - [`Mdd`] is one diagram, compiled from any subset of the constraints of a problem.
//! - [`MddArena`] caches the diagram of each constraint shape so that identical constraints on
//!   different variables share it.
//! - [`heuristics`] holds the choices that steer the compilation.
//! - [`MddView`] and [`MddViewWithOrder`] are the read-only surface that `wmc` and the sampling
//!   code are written against, so that they work on either representation.
//!
//! # How it works
//!
//! `Mdd::new` creates one node per layer and propagates the constraints: every constraint keeps a
//! summary (a `ConstraintProperty`) of the paths that reach a node from the root and of the paths
//! that leave it towards the sink, and an edge is removed when the two summaries show that no
//! solution goes through it. `Mdd::refine` then splits relaxed nodes into more precise ones, and
//! merges nodes again whenever a layer is wider than the limit. Merging keeps every path of both
//! nodes, so a relaxed diagram contains every solution: the number of its paths is an upper bound
//! on the number of solutions, and a diagram without a width limit is as tight as the filtering
//! of its constraints allows.
//!
//! Edges and nodes are never deleted while the diagram changes: they are deactivated, and
//! `clean` drops them and renumbers the rest, so that an `EdgeIndex` or a `NodeIndex` stays valid
//! between two cleanings.
//!
//! # Design notes
//!
//! - **The branching order is repaired, not trusted.** Heuristics know nothing of constraints that
//!   need a specific sequence of variables (`Regular`); `Mdd::new` reorders for them and panics
//!   if two constraints ask for contradictory sequences.
//! - **A diagram is only as exact as its constraints' filtering.** Each constraint documents
//!   what its properties can and cannot detect.
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
//! let vars = problem.add_variables(3, vec![0, 1, 2], None);
//! let constraint = all_different(&mut problem, vars);
//!
//! let mut mdd = Mdd::new(
//!     Arc::new(problem),
//!     OrderingHeuristic::MinDomMaxLinked,
//!     MergeHeuristic::LessRelaxed,
//!     SelectHeuristic::Greedy,
//!     &[constraint],
//! );
//! mdd.refine(usize::MAX);
//!
//! // The permutations of three values.
//! assert_eq!(mdd.count_solutions().to_string(), "6");
//! let mut sample = mdd.sample();
//! sample.sort();
//! assert_eq!(sample, vec![0, 1, 2]);
//! ```

pub mod heuristics;
pub mod wmc;

pub(crate) mod arena;
pub(crate) mod edge;
#[allow(clippy::module_inception)]
pub(crate) mod mdd;
mod memory;
pub(crate) mod node;
pub(crate) mod structure;
pub(crate) mod view;

pub use arena::MddArena;
pub use mdd::Mdd;
pub use view::{MddView, MddViewWithOrder};

pub(crate) use arena::CompiledConstraint;
pub(crate) use edge::Edge;
pub(crate) use node::Node;
pub(crate) use structure::MddStructure;

use crate::constraints::*;
use std::hash::{Hash, Hasher};

/// Address of a node: its layer, then its position in the layer.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize, pub usize);

/// Address of an edge: the layer it leaves, then its position among the edges of that layer.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize, pub usize);

/// What makes two nodes of a layer interchangeable: all their top-down and bottom-up properties
/// are equal. `Mdd::collapse` merges the nodes that share a key, which does not relax the diagram.
struct MergeKey<'a> {
    /// The node's top-down property for each compiled constraint.
    td_properties: &'a [Box<dyn ConstraintProperty>],
    /// The node's bottom-up property for each compiled constraint.
    bu_properties: &'a [Box<dyn ConstraintProperty>],
}

impl<'a> Hash for MergeKey<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for td_property in self.td_properties {
            td_property.hash(state);
        }
        for bu_property in self.bu_properties {
            bu_property.hash(state);
        }
    }
}

impl<'a> PartialEq for MergeKey<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.td_properties
            .iter()
            .zip(other.td_properties.iter())
            .all(|(p1, p2)| p1.eq(p2))
            && self
                .bu_properties
                .iter()
                .zip(other.bu_properties.iter())
                .all(|(p1, p2)| p1.eq(p2))
    }
}

impl<'a> Eq for MergeKey<'a> {}
