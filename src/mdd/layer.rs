use super::*;
use crate::modelling::VariableIndex;

/// A layer of the MDD
pub struct Layer {
    /// Nodes of the layers
    nodes: Vec<NodeIndex>,
    /// Edges from this layer to the next
    edges: Vec<EdgeIndex>,
    /// Decision varaible associated with the layer
    decision: VariableIndex,
}

impl Layer {

    pub fn new() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            decision: VariableIndex(0),
        }
    }

    /// Adds a node the the layer and returns its index
    pub fn add_node(&mut self, node: NodeIndex) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: EdgeIndex) {
        self.edges.push(edge);
    }

    pub fn decision(&self) -> VariableIndex {
        self.decision
    }

    pub fn set_decision(&mut self, decision: VariableIndex) {
        self.decision = decision
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_at(&self, index: usize) -> NodeIndex {
        self.nodes[index]
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn edge_at(&self, index: usize) -> EdgeIndex {
        self.edges[index]
    }

    pub fn swap_remove_edge(&mut self, index: usize) {
        self.edges.swap_remove(index);
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        self.nodes.iter().copied()
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = EdgeIndex> {
        self.edges.iter().copied()
    }
}
