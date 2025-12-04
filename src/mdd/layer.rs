use super::*;
use crate::modelling::VariableIndex;

#[derive(Default, Clone)]
/// A layer of the MDD
pub struct Layer {
    /// Nodes of the layers
    nodes: Vec<NodeIndex>,
    /// Decision varaible associated with the layer
    decision: VariableIndex,
}

impl Layer {

    /// Adds a node the the layer and returns its index
    pub fn add_node(&mut self, node: NodeIndex) {
        self.nodes.push(node);
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

    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        self.nodes.iter().copied()
    }
}
