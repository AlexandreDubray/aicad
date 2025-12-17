use super::*;

/// A decision node of the MDD
#[derive(Default)]
pub struct Node {
    /// Layer containing the node
    layer: LayerIndex,
    /// What is the index of the node in the layer
    index_in_layer: usize,
    /// Edges to from the parent of the nodes
    parents_edges: Vec<EdgeIndex>,
    /// Edges to the children of the nodes
    children_edges: Vec<EdgeIndex>,
    /// Is the node active
    active: bool,
}

impl Node {

    pub fn new(layer: LayerIndex, index_in_layer: usize) -> Self {
        Self {
            layer,
            index_in_layer,
            parents_edges: vec![],
            children_edges: vec![],
            active: true,
        }
    }

    pub fn layer(&self) -> LayerIndex {
        self.layer
    }

    pub fn index_in_layer(&self) -> usize {
        self.index_in_layer
    }

    pub fn number_children(&self) -> usize {
        self.children_edges.len()
    }

    pub fn add_child_edge(&mut self, edge: EdgeIndex) {
        self.children_edges.push(edge);
    }

    pub fn child_edge_at(&self, index: usize) -> EdgeIndex {
        self.children_edges[index]
    }

    pub fn swap_remove_child_edge(&mut self, index: usize) {
        self.children_edges.swap_remove(index);
    }

    pub fn remove_child_edge(&mut self, edge: EdgeIndex) {
        for i in (0..self.children_edges.len()).rev() {
            if self.children_edges[i] == edge {
                self.children_edges.swap_remove(i);
                break;
            }
        }
    }

    pub fn number_parents(&self) -> usize {
        self.parents_edges.len()
    }

    pub fn add_parent_edge(&mut self, edge: EdgeIndex) {
        self.parents_edges.push(edge);
    }

    pub fn parent_edge_at(&self, index: usize) -> EdgeIndex {
        self.parents_edges[index]
    }

    pub fn swap_remove_parent_edge(&mut self, index: usize) {
        self.parents_edges.swap_remove(index);
    }

    pub fn remove_parent_edge(&mut self, edge: EdgeIndex) {
        for i in (0..self.parents_edges.len()).rev() {
            if self.parents_edges[i] == edge {
                self.parents_edges.swap_remove(i);
                break;
            }
        }
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}
