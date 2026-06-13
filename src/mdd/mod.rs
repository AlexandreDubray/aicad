pub mod mdd;
pub mod node;
pub mod layer;
pub mod edge;
pub mod heuristics;

// re-export modules
pub use mdd::Mdd;
pub use node::Node;
pub use layer::Layer;
pub use edge::Edge;

use crate::constraints::Constraint;
use std::hash::{Hash, Hasher};

/// Represents the index of a node in a layer of the MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize, pub usize);

/// Represents the index of an edge in a layer of a MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize, pub usize);

struct MergeKey<'a> {
    node: NodeIndex,
    constraints: &'a [Box<dyn Constraint + Send + Sync>],
}

impl<'a> Hash for MergeKey<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for constraint in self.constraints {
            constraint.hash_node_state(self.node, state);
        }
    }
}

impl<'a> PartialEq for MergeKey<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.constraints.iter().all(|constraint| constraint.eq_node_state(self.node, other.node))
    }
}

impl<'a> Eq for MergeKey<'a> {}
