use super::*;
use crate::core::VariableIndex;

/// A layer of the MDD
pub struct Layer {
    /// Nodes of the layers
    nodes: Vec<Node>,
    /// Outgoing edges to the next layer. If the index of the layer is i, then there are edges only
    /// to i+1
    edges: Vec<Edge>,
    /// Decision varaible associated with the layer
    decision: VariableIndex,
    /// Vector of size k, with k the number of constraints, such that is_in_constraint_scope[j] = true
    /// iff decision is in the scope of the j-th constraint.
    is_in_constraint_scope: Vec<bool>,
}

impl Layer {

    pub fn new(is_in_constraint_scope: Vec<bool>) -> Self {
        Self {
            nodes: vec![Node::default()],
            edges: vec![],
            decision: VariableIndex(0),
            is_in_constraint_scope,
        }
    }

    /// Adds a node the the layer and returns its index
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        let index = NodeIndex(self.nodes.len());
        self.nodes.push(node);
        index
    }

    /// Adds an edge to the layer and returns its index
    pub fn add_edge(&mut self, edge: Edge) -> EdgeIndex {
        let index = EdgeIndex(self.edges.len());
        self.edges.push(edge);
        index
    }

    pub fn decision(&self) -> VariableIndex {
        self.decision
    }

    pub fn set_decision(&mut self, decision: VariableIndex) {
        self.decision = decision
    }

    pub fn is_in_constraint_scope(&self, constraint_index: usize) -> bool {
        self.is_in_constraint_scope[constraint_index]
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).map(NodeIndex)
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = EdgeIndex> {
        (0..self.edges.len()).map(EdgeIndex)
    }
}

impl std::ops::Index<NodeIndex> for Layer {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<NodeIndex> for Layer {

    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

impl std::ops::Index<EdgeIndex> for Layer {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0]
    }
}
