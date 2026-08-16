pub mod edge;
pub mod heuristics;
pub mod layer;
pub mod mdd;
pub mod node;

// re-export modules
pub use edge::Edge;
pub use layer::Layer;
pub use mdd::Mdd;
pub use node::Node;

use crate::constraints::*;
use std::hash::{Hash, Hasher};

/// Represents the index of a node in a layer of the MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize, pub usize);

/// Represents the index of an edge in a layer of a MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize, pub usize);

struct MergeKey<'a> {
    td_properties: &'a [Box<dyn ConstraintProperty>],
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
