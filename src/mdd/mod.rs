//! Multi-valued decision diagrams (MDDs) of constraints: compilation, relaxation and queries.
//!
//! An MDD stores the solutions of a constraint problem as the paths of a layered graph: one layer
//! per variable, an edge for each value a variable may take, and a single root and sink. This
//! module compiles the constraints of a [`Problem`](crate::modelling::Problem) into such a
//! diagram, with a limit on the width of the layers if wanted, and allow tractable queries on it:
//! counting its solutions and reading one off it, and weighted model counting (`wmc`) for the
//! learning and sampling code.
//!
//! - [`Compiler`] builds a diagram from any subset of the constraints of a problem and refines it,
//!   as many times as wanted and with growing widths if wanted.
//! - [`Mdd`] is the diagram itself, which a [`Compiler`] owns and lends out between refinements.
//! - [`MddArena`] caches the diagram of each constraint shape so that identical constraints on
//!   different variables share it.
//! - [`heuristics`] holds the choices that steer the compilation.
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
//! let vars = problem.add_variables(3, vec![0, 1, 2], None);
//! let constraint = all_different(&mut problem, vars);
//!
//! let mut compiler = Compiler::new(
//!     Arc::new(problem),
//!     CompilationHeuristic {
//!         ordering: OrderingHeuristic::MinDomMaxLinked,
//!         merge: MergeHeuristic::LessRelaxed,
//!         selection: SelectHeuristic::Greedy,
//!     },
//!     &[constraint],
//! );
//! compiler.refine(usize::MAX);
//! let mdd = compiler.into_mdd();
//!
//! // The permutations of three values.
//! assert_eq!(mdd.count_solutions().to_string(), "6");
//! let (mut values, variables) = mdd.get_solution().expect("the diagram is satisfiable");
//! assert_eq!(variables.len(), 3);
//! values.sort();
//! assert_eq!(values, vec![0, 1, 2]);
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod heuristics;
pub mod wmc;

pub(crate) mod arena;
pub(crate) mod compiler;
pub(crate) mod diagram;
pub(crate) mod edge;
mod inspect;
mod memory;
pub(crate) mod node;

pub use arena::MddArena;
pub use compiler::Compiler;
pub use diagram::Mdd;

pub(crate) use arena::CompiledConstraint;
pub(crate) use edge::Edge;
pub(crate) use node::Node;

use crate::constraints::*;
use std::hash::{Hash, Hasher};

/// Address of a node: its layer, then its position in the layer.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize, pub usize);

/// Address of an edge: the layer it leaves, then its position among the edges of that layer.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize, pub usize);

/// What makes two nodes of a layer interchangeable: all their top-down and bottom-up properties
/// are equal. `Compiler::collapse` merges the nodes that share a key, which does not relax the diagram.
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
