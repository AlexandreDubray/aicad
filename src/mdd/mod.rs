pub mod mdd;
pub mod node;
pub mod layer;
pub mod edge;

/// Represents the index of a layer in the MDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct LayerIndex(pub usize);

/// Represents the index of a node in a layer of the MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize);

/// Represents the index of an edge in a layer of a MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize);

// re-export modules
pub use mdd::Mdd;
pub use node::Node;
pub use layer::Layer;
pub use edge::Edge;
