//! `MddStructure` is the purely structural part of a compiled `Mdd`: its nodes, edges, root and
//! sink, and whether it's unsat -- everything that determines the diagram's shape, and nothing
//! that ties it to a particular real problem's `VariableIndex`s.

use super::{Edge, EdgeIndex, Node, NodeIndex};

/// The compiled shape of an MDD: nodes, edges, root and sink, with no reference to which real
/// problem or which `VariableIndex`s were compiled. Built from an already-compiled (and normally
/// already-`refine`d) `Mdd` via `Mdd::into_structure`.
pub struct MddStructure {
    pub(super) nodes: Vec<Vec<Node>>,
    pub(super) edges: Vec<Vec<Edge>>,
    pub(super) root: NodeIndex,
    pub(super) sink: NodeIndex,
    pub(super) unsat: bool,
}

impl MddStructure {
    pub fn number_nodes(&self) -> usize {
        self.nodes.iter().map(|layer| layer.len()).sum::<usize>()
    }

    pub fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.nodes[layer].len()
    }

    pub fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes[layer].len()).map(move |index| NodeIndex(layer, index))
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn number_layers(&self) -> usize {
        self.nodes.len()
    }

    pub fn root(&self) -> NodeIndex {
        self.root
    }

    pub fn sink(&self) -> NodeIndex {
        self.sink
    }

    pub fn is_unsat(&self) -> bool {
        self.unsat
    }
}

impl std::ops::Index<NodeIndex> for MddStructure {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0][index.1]
    }
}

impl std::ops::Index<EdgeIndex> for MddStructure {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0][index.1]
    }
}
