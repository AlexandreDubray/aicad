use super::*;
use rustc_hash::FxHashMap;

/// A decision node of the MDD
#[derive(Default, Clone)]
pub struct Node {
    /// Layer containing the node
    layer: usize,
    /// What is the index of the node in the layer
    index_in_layer: usize,
    /// Edges to from the parent of the nodes
    parents_edges: Vec<EdgeIndex>,
    /// Edges to the children of the nodes
    children_edges: Vec<EdgeIndex>,
    /// Is the node active
    active: bool,
    /// Is the node relaxed
    relaxed: bool,
    /// Is the node flaged for property update
    property_flag: bool,
}

impl Node {
    pub fn new(layer: usize, index_in_layer: usize, relaxed: bool) -> Self {
        Self {
            layer,
            index_in_layer,
            parents_edges: vec![],
            children_edges: vec![],
            active: true,
            relaxed,
            property_flag: false,
        }
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn index_in_layer(&self) -> usize {
        self.index_in_layer
    }

    pub(crate) fn number_children(&self) -> usize {
        self.children_edges.len()
    }

    pub(crate) fn add_child_edge(&mut self, edge: EdgeIndex) {
        self.children_edges.push(edge);
    }

    pub fn set_child_edges(&mut self, edges: &[EdgeIndex]) {
        self.children_edges.copy_from_slice(edges);
    }

    pub(crate) fn child_edge_at(&self, index: usize) -> EdgeIndex {
        self.children_edges[index]
    }

    pub fn child_edges(&self) -> &[EdgeIndex] {
        &self.children_edges
    }

    pub(crate) fn swap_remove_child_edge(&mut self, index: usize) {
        self.children_edges.swap_remove(index);
    }

    pub(crate) fn remove_child_edge(&mut self, edge: EdgeIndex) {
        for i in (0..self.children_edges.len()).rev() {
            if self.children_edges[i] == edge {
                self.children_edges.swap_remove(i);
                break;
            }
        }
    }

    pub(crate) fn number_parents(&self) -> usize {
        self.parents_edges.len()
    }

    pub(crate) fn add_parent_edge(&mut self, edge: EdgeIndex) {
        self.parents_edges.push(edge);
    }

    pub(crate) fn parent_edge_at(&self, index: usize) -> EdgeIndex {
        self.parents_edges[index]
    }

    pub(crate) fn swap_remove_parent_edge(&mut self, index: usize) {
        self.parents_edges.swap_remove(index);
    }

    pub(crate) fn remove_parent_edge(&mut self, edge: EdgeIndex) {
        for i in (0..self.parents_edges.len()).rev() {
            if self.parents_edges[i] == edge {
                self.parents_edges.swap_remove(i);
                break;
            }
        }
    }

    pub(crate) fn deactivate(&mut self) {
        self.active = false;
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    pub(crate) fn iter_parents(&self) -> impl Iterator<Item = EdgeIndex> {
        self.parents_edges.iter().copied()
    }

    pub(crate) fn iter_children(&self) -> impl Iterator<Item = EdgeIndex> {
        self.children_edges.iter().copied()
    }

    pub(crate) fn is_relaxed(&self) -> bool {
        self.relaxed
    }

    pub(crate) fn set_relaxed(&mut self, relaxed: bool) {
        self.relaxed = relaxed
    }

    pub(crate) fn is_property_flag(&self) -> bool {
        self.property_flag
    }

    pub(crate) fn set_property_flag(&mut self) {
        self.property_flag = true;
    }

    pub(crate) fn unset_property_flag(&mut self) {
        self.property_flag = false;
    }

    pub(crate) fn update_edge_indices(&mut self, map: &FxHashMap<EdgeIndex, EdgeIndex>) {
        for i in (0..self.parents_edges.len()).rev() {
            match map.get(&self.parents_edges[i]) {
                Some(&new_index) => self.parents_edges[i] = new_index,
                None => _ = self.parents_edges.swap_remove(i),
            };
        }
        for i in (0..self.children_edges.len()).rev() {
            match map.get(&self.children_edges[i]) {
                Some(&new_index) => self.children_edges[i] = new_index,
                None => _ = self.children_edges.swap_remove(i),
            }
        }
    }
}
