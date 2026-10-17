//! One node of an MDD, with the edges that enter and leave it.
//!
//! # Meaning
//!
//! A node of layer `l` is a set of partial assignments of the first `l` variables of the branching
//! order that the compiler could not tell apart. It lists the edges that reach it (parents) and
//! the edges that leave it (children), both as `EdgeIndex`es into the diagram, and carries three
//! flags used by the compiler.
//!
//! # How it works
//!
//! - **Active.** A removed node is only deactivated; `Mdd::clean` drops it later.
//! - **Relaxed.** A node is relaxed once it is the result of merging nodes that the constraints did
//!   not consider equivalent, so it may accept paths that no constraint-consistent node would.
//!   A diagram without relaxed nodes is exact.
//! - **Property flag.** Set when the top-down or bottom-up properties of the node are out of date
//!   and must be recomputed by the next propagation.
//!
//! Edge lists are unordered: removing an edge swaps the last entry into its place, so positions
//! are only meaningful until the next removal.

use super::*;
use rustc_hash::FxHashMap;

/// A decision node of the MDD.
#[derive(Clone)]
pub struct Node {
    /// Layer containing the node.
    layer: usize,
    /// Position of the node in its layer.
    index_in_layer: usize,
    /// Edges entering the node.
    parents_edges: Vec<EdgeIndex>,
    /// Edges leaving the node.
    children_edges: Vec<EdgeIndex>,
    /// `false` once the node has been removed.
    active: bool,
    /// `true` if the node is the result of a lossy merge.
    relaxed: bool,
    /// `true` if the node's properties must be recomputed.
    property_flag: bool,
}

impl Node {
    /// Creates an active node without edges at position `index_in_layer` of `layer`.
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

    /// Layer containing the node.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Position of the node in its layer.
    pub fn index_in_layer(&self) -> usize {
        self.index_in_layer
    }

    /// Number of listed child edges, deactivated ones included.
    pub(crate) fn number_children(&self) -> usize {
        self.children_edges.len()
    }

    /// Lists `edge` as a child edge.
    pub(crate) fn add_child_edge(&mut self, edge: EdgeIndex) {
        self.children_edges.push(edge);
    }

    /// The child edge at position `index` of the list.
    pub(crate) fn child_edge_at(&self, index: usize) -> EdgeIndex {
        self.children_edges[index]
    }

    /// All listed child edges, deactivated ones included.
    pub fn child_edges(&self) -> &[EdgeIndex] {
        &self.children_edges
    }

    /// Removes the child edge at position `index`; the last entry takes its place.
    pub(crate) fn swap_remove_child_edge(&mut self, index: usize) {
        self.children_edges.swap_remove(index);
    }

    /// Removes one listing of `edge` from the child edges (the last one if it is listed twice).
    /// Does nothing if the edge is not listed.
    pub(crate) fn remove_child_edge(&mut self, edge: EdgeIndex) {
        if let Some(i) = self.children_edges.iter().rposition(|&e| e == edge) {
            self.children_edges.swap_remove(i);
        }
    }

    /// Number of listed parent edges.
    pub(crate) fn number_parents(&self) -> usize {
        self.parents_edges.len()
    }

    /// Lists `edge` as a parent edge.
    pub(crate) fn add_parent_edge(&mut self, edge: EdgeIndex) {
        self.parents_edges.push(edge);
    }

    /// The parent edge at position `index` of the list.
    pub(crate) fn parent_edge_at(&self, index: usize) -> EdgeIndex {
        self.parents_edges[index]
    }

    /// Removes the parent edge at position `index`; the last entry takes its place.
    pub(crate) fn swap_remove_parent_edge(&mut self, index: usize) {
        self.parents_edges.swap_remove(index);
    }

    /// Removes one listing of `edge` from the parent edges (the last one if it is listed twice).
    /// Does nothing if the edge is not listed.
    pub(crate) fn remove_parent_edge(&mut self, edge: EdgeIndex) {
        if let Some(i) = self.parents_edges.iter().rposition(|&e| e == edge) {
            self.parents_edges.swap_remove(i);
        }
    }

    /// Marks the node as removed. It stays in its layer until `Mdd::clean`.
    pub(crate) fn deactivate(&mut self) {
        self.active = false;
    }

    /// `false` once the node has been removed.
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Iterates over the listed parent edges.
    pub(crate) fn iter_parents(&self) -> impl Iterator<Item = EdgeIndex> {
        self.parents_edges.iter().copied()
    }

    /// Iterates over the listed child edges, deactivated ones included.
    pub(crate) fn iter_children(&self) -> impl Iterator<Item = EdgeIndex> {
        self.children_edges.iter().copied()
    }

    /// `true` if the node is the result of a lossy merge.
    pub(crate) fn is_relaxed(&self) -> bool {
        self.relaxed
    }

    /// Sets whether the node is the result of a lossy merge.
    pub(crate) fn set_relaxed(&mut self, relaxed: bool) {
        self.relaxed = relaxed
    }

    /// `true` if the node's properties must be recomputed.
    pub(crate) fn is_property_flag(&self) -> bool {
        self.property_flag
    }

    /// Flags the node's properties as out of date.
    pub(crate) fn set_property_flag(&mut self) {
        self.property_flag = true;
    }

    /// Clears the flag after the properties have been recomputed.
    pub(crate) fn unset_property_flag(&mut self) {
        self.property_flag = false;
    }

    /// Renames the listed edges through `map`, the old-to-new edge numbering built by
    /// `Mdd::clean`, and drops the edges that are not in `map` (the removed ones).
    pub(crate) fn update_edge_indices(&mut self, map: &FxHashMap<EdgeIndex, EdgeIndex>) {
        for edges in [&mut self.parents_edges, &mut self.children_edges] {
            for i in (0..edges.len()).rev() {
                match map.get(&edges[i]) {
                    Some(&new_index) => edges[i] = new_index,
                    None => _ = edges.swap_remove(i),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(layer: usize, i: usize) -> EdgeIndex {
        EdgeIndex(layer, i)
    }

    fn node_with(parents: &[EdgeIndex], children: &[EdgeIndex]) -> Node {
        let mut node = Node::new(1, 0, false);
        parents.iter().for_each(|&p| node.add_parent_edge(p));
        children.iter().for_each(|&c| node.add_child_edge(c));
        node
    }

    #[test]
    fn a_new_node_is_active_exact_unflagged_and_has_no_edges() {
        let node = Node::new(3, 7, false);
        assert_eq!((node.layer(), node.index_in_layer()), (3, 7));
        assert!(node.is_active() && !node.is_relaxed() && !node.is_property_flag());
        assert_eq!((node.number_parents(), node.number_children()), (0, 0));
        assert!(Node::new(0, 0, true).is_relaxed());
    }

    #[test]
    fn flags_toggle_independently() {
        let mut node = Node::new(1, 0, false);
        node.set_property_flag();
        assert!(node.is_property_flag() && node.is_active() && !node.is_relaxed());
        node.unset_property_flag();
        node.set_relaxed(true);
        node.deactivate();
        assert!(!node.is_property_flag() && !node.is_active() && node.is_relaxed());
    }

    #[test]
    fn edges_are_listed_in_insertion_order_and_readable_three_ways() {
        let node = node_with(&[e(0, 0), e(0, 1)], &[e(1, 4), e(1, 5), e(1, 6)]);
        assert_eq!(node.child_edges(), &[e(1, 4), e(1, 5), e(1, 6)]);
        assert_eq!(node.iter_children().collect::<Vec<_>>(), node.child_edges());
        assert_eq!(node.child_edge_at(1), e(1, 5));
        assert_eq!(
            node.iter_parents().collect::<Vec<_>>(),
            vec![e(0, 0), e(0, 1)]
        );
        assert_eq!(node.parent_edge_at(1), e(0, 1));
    }

    #[test]
    fn removing_by_position_moves_the_last_edge_into_the_gap() {
        let mut node = node_with(&[e(0, 0), e(0, 1), e(0, 2)], &[e(1, 0), e(1, 1), e(1, 2)]);
        node.swap_remove_child_edge(0);
        node.swap_remove_parent_edge(0);
        assert_eq!(node.child_edges(), &[e(1, 2), e(1, 1)]);
        assert_eq!(
            node.iter_parents().collect::<Vec<_>>(),
            vec![e(0, 2), e(0, 1)]
        );
    }

    #[test]
    fn removing_by_edge_removes_a_single_listing_and_ignores_unknown_edges() {
        let mut node = node_with(&[e(0, 1), e(0, 2), e(0, 1)], &[e(1, 1), e(1, 2), e(1, 1)]);
        node.remove_child_edge(e(1, 1));
        node.remove_parent_edge(e(0, 1));
        assert_eq!(node.number_children(), 2);
        assert_eq!(node.number_parents(), 2);
        assert_eq!(
            node.child_edges().iter().filter(|&&c| c == e(1, 1)).count(),
            1
        );
        assert_eq!(node.iter_parents().filter(|&p| p == e(0, 1)).count(), 1);
        node.remove_child_edge(e(1, 9));
        node.remove_parent_edge(e(0, 9));
        assert_eq!((node.number_children(), node.number_parents()), (2, 2));
    }

    #[test]
    fn renumbering_edges_renames_survivors_and_drops_the_others() {
        let mut node = node_with(&[e(0, 0), e(0, 3)], &[e(1, 2), e(1, 5), e(1, 7)]);
        let map: FxHashMap<_, _> = [(e(0, 3), e(0, 0)), (e(1, 5), e(1, 1)), (e(1, 7), e(1, 2))]
            .into_iter()
            .collect();
        node.update_edge_indices(&map);
        assert_eq!(node.iter_parents().collect::<Vec<_>>(), vec![e(0, 0)]);
        let mut children = node.child_edges().to_vec();
        children.sort();
        assert_eq!(children, vec![e(1, 1), e(1, 2)]);
    }
}
