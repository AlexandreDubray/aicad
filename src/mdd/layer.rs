use super::*;
use crate::modelling::VariableIndex;

#[derive(Default, Clone)]
/// A layer of the MDD
pub struct Layer {
    /// Nodes of the layers
    nodes: Vec<NodeIndex>,
    /// Decision varaible associated with the layer
    decision: VariableIndex,
    /// Number of active nodes in the layer
    number_active_node: usize,
}

impl Layer {

    /// Adds a node the the layer and returns its index
    pub fn add_node(&mut self, node: NodeIndex) {
        self.nodes.push(node);
        self.number_active_node += 1;
    }

    pub fn decision(&self) -> VariableIndex {
        self.decision
    }

    pub fn set_decision(&mut self, decision: VariableIndex) {
        self.decision = decision
    }

    pub fn number_nodes(&self) -> usize {
        self.number_active_node
    }

    pub fn node_at(&self, index: usize) -> NodeIndex {
        self.nodes[index]
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        self.nodes.iter().copied()
    }

    pub fn decrease_active_node(&mut self) {
        self.number_active_node -= 1;
    }
}
