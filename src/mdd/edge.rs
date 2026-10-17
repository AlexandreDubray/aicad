//! One edge of an MDD: the choice of a value for the variable of its layer.
//!
//! # Meaning
//!
//! An edge stored in layer `l` goes from a node of layer `l` to a node of layer `l + 1` and stands
//! for "the variable branched on at layer `l` takes the value `assignment`". A path from the root
//! to the sink is one (candidate) solution. Edges are addressed by `EdgeIndex(l, position)`.
//!
//! # How it works
//!
//! An edge is never removed from its layer while the diagram is being modified: it is only
//! deactivated, so that no `EdgeIndex` changes under the compiler's feet. `Mdd::clean` finally
//! drops the inactive edges and renumbers the survivors (see [`Edge::update_node_indices`]).
//!
//! # Design notes
//!
//! - **A dead edge can still be listed by its parent.** The compiler removes a deactivated edge from
//!   the child's parent list, but the parent's child list keeps it until `clean`. Code that walks a
//!   node's children must therefore test [`Edge::is_active`] (reading dead entries made merging
//!   lose solutions; see `merging_does_not_read_removed_edges` in `mdd.rs`).

use super::*;
use crate::modelling::*;
use rustc_hash::FxHashMap;

/// An edge of the MDD, labelled by the value it assigns to the variable of its layer.
#[derive(Clone)]
pub struct Edge {
    /// Node of layer `l` the edge leaves.
    from: NodeIndex,
    /// Node of layer `l + 1` the edge enters.
    to: NodeIndex,
    /// Value given to the variable of layer `l` when the edge is taken.
    assignment: ValueIndex,
    /// `false` once the edge has been removed; see the module documentation.
    active: bool,
}

impl Edge {
    /// Creates an active edge from `from` to `to` that assigns `assignment`.
    pub fn new(from: NodeIndex, to: NodeIndex, assignment: ValueIndex) -> Self {
        Self {
            from,
            to,
            assignment,
            active: true,
        }
    }

    /// The node the edge leaves.
    pub fn from(&self) -> NodeIndex {
        self.from
    }

    /// Redirects the edge to leave `from` (used when nodes are merged or split).
    pub(crate) fn set_from(&mut self, from: NodeIndex) {
        self.from = from;
    }

    /// The node the edge enters.
    pub fn to(&self) -> NodeIndex {
        self.to
    }

    /// Redirects the edge to enter `to` (used when nodes are merged or split).
    pub(crate) fn set_to(&mut self, to: NodeIndex) {
        self.to = to;
    }

    /// The value assigned to the variable of the edge's layer.
    pub fn assignment(&self) -> ValueIndex {
        self.assignment
    }

    /// Marks the edge as removed. It stays in its layer until `Mdd::clean`.
    pub(crate) fn deactivate(&mut self) {
        self.active = false;
    }

    /// `false` once the edge has been removed.
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Renames both endpoints through `map`, the old-to-new node numbering built by `Mdd::clean`.
    ///
    /// # Panics
    ///
    /// Panics if an endpoint is missing from `map`. `clean` only renumbers edges whose endpoints
    /// both survive, so a missing endpoint means an edge was left pointing at a removed node.
    pub(crate) fn update_node_indices(&mut self, map: &FxHashMap<NodeIndex, NodeIndex>) {
        let rename = |node: NodeIndex| {
            *map.get(&node).unwrap_or_else(|| {
                panic!("cannot renumber an edge towards {node:?}: that node was removed")
            })
        };
        self.from = rename(self.from);
        self.to = rename(self.to);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> Edge {
        Edge::new(NodeIndex(0, 0), NodeIndex(1, 2), ValueIndex(3))
    }

    #[test]
    fn a_new_edge_is_active_and_keeps_its_endpoints_and_label() {
        let e = edge();
        assert!(e.is_active());
        assert_eq!((e.from(), e.to()), (NodeIndex(0, 0), NodeIndex(1, 2)));
        assert_eq!(e.assignment(), ValueIndex(3));
    }

    #[test]
    fn deactivating_removes_the_edge_without_changing_what_it_says() {
        let mut e = edge();
        e.deactivate();
        assert!(!e.is_active());
        assert_eq!(
            (e.from(), e.to(), e.assignment()),
            (NodeIndex(0, 0), NodeIndex(1, 2), ValueIndex(3))
        );
    }

    #[test]
    fn redirecting_changes_only_the_endpoint_asked() {
        let mut e = edge();
        e.set_to(NodeIndex(1, 5));
        assert_eq!((e.from(), e.to()), (NodeIndex(0, 0), NodeIndex(1, 5)));
        e.set_from(NodeIndex(0, 4));
        assert_eq!((e.from(), e.to()), (NodeIndex(0, 4), NodeIndex(1, 5)));
    }

    #[test]
    fn renumbering_renames_both_endpoints_and_leaves_the_label() {
        let mut e = edge();
        let map: FxHashMap<_, _> = [
            (NodeIndex(0, 0), NodeIndex(0, 0)),
            (NodeIndex(1, 2), NodeIndex(1, 0)),
        ]
        .into_iter()
        .collect();
        e.update_node_indices(&map);
        assert_eq!((e.from(), e.to()), (NodeIndex(0, 0), NodeIndex(1, 0)));
        assert_eq!(e.assignment(), ValueIndex(3));
    }

    #[test]
    #[should_panic(expected = "that node was removed")]
    fn renumbering_towards_a_removed_node_panics_with_its_index() {
        let mut e = edge();
        let map: FxHashMap<_, _> = [(NodeIndex(0, 0), NodeIndex(0, 0))].into_iter().collect();
        e.update_node_indices(&map);
    }
}
