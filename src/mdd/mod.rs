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

/// Represents the index of a layer in the MDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct LayerIndex(pub usize);

impl std::ops::Deref for LayerIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Represents the index of a node in a layer of the MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize);

impl std::ops::Deref for NodeIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Represents the index of an edge in a layer of a MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize);

impl std::ops::Deref for EdgeIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
