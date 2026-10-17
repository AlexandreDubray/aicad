//! The shape of a compiled MDD, without the problem it was compiled for.
//!
//! # Meaning
//!
//! `MddStructure` keeps the nodes, edges, root, sink and satisfiability flag of a compiled `Mdd`,
//! and nothing that refers to the variables of a particular `Problem`. Two constraints of the
//! same shape on different variables therefore compile to one structure, which `MddArena` shares.
//!
//! # How it works
//!
//! A structure is obtained by consuming a compiled (normally refined) `Mdd` with
//! `Mdd::into_structure`; it is read-only afterwards. It implements `MddView`, so the algorithms
//! written for `Mdd` (weighted counting, belief propagation) run on it unchanged. Layer `l` of
//! `nodes` holds the nodes at depth `l` and `edges[l]` the edges from layer `l` to `l + 1`.

use super::{Edge, EdgeIndex, Node, NodeIndex};

/// The compiled shape of an MDD, with no reference to the problem it came from.
pub(crate) struct MddStructure {
    /// Nodes, one vector per layer (root layer first, sink layer last).
    pub(super) nodes: Vec<Vec<Node>>,
    /// Edges, one vector per pair of consecutive layers.
    pub(super) edges: Vec<Vec<Edge>>,
    /// The only node of the first layer.
    pub(super) root: NodeIndex,
    /// The only node of the last layer.
    pub(super) sink: NodeIndex,
    /// `true` if the constraint has no solution (the diagram then has no root-to-sink path).
    pub(super) unsat: bool,
}

impl MddStructure {
    /// Total number of nodes, in every layer.
    #[cfg(test)]
    pub(crate) fn number_nodes(&self) -> usize {
        self.nodes.iter().map(|layer| layer.len()).sum::<usize>()
    }

    /// Number of nodes in `layer`.
    pub(crate) fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.nodes[layer].len()
    }

    /// Iterates over the nodes of `layer`.
    pub(crate) fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes[layer].len()).map(move |index| NodeIndex(layer, index))
    }

    #[cfg(test)]
    pub(crate) fn number_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of layers, the sink's included (one more than the number of variables).
    pub(crate) fn number_layers(&self) -> usize {
        self.nodes.len()
    }

    /// The root node.
    pub(crate) fn root(&self) -> NodeIndex {
        self.root
    }

    /// The sink node.
    pub(crate) fn sink(&self) -> NodeIndex {
        self.sink
    }

    /// `true` if the diagram accepts no assignment.
    pub(crate) fn is_unsat(&self) -> bool {
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
