//! `MddView`/`MddViewWithOrder`: the structural read-only surface `crate::mdd::wmc` and
//! `crate::sampling` need, implemented by both `Mdd` (the legacy, self-contained compiled
//! diagram) and `MddStructure`/`CompiledConstraint` (the arena-backed split of structure from scope --
//! see `crate::mdd::arena`'s module doc). Generic algorithms written against these traits work
//! unchanged against either representation, so `wmc`/belief-propagation didn't need rewriting,
//! only retyping.

use super::{Edge, MddStructure, Node, NodeIndex};
use crate::modelling::VariableIndex;

/// The purely structural surface: everything `crate::mdd::wmc::forward`/`backward`/`wmc`/
/// `gradient` need, and nothing that depends on which real problem or `VariableIndex`s this
/// diagram was compiled for.
pub trait MddView: std::ops::Index<NodeIndex, Output = Node> + std::ops::Index<super::EdgeIndex, Output = Edge> {
    fn root(&self) -> NodeIndex;
    fn sink(&self) -> NodeIndex;
    fn number_layers(&self) -> usize;
    fn number_nodes_in_layer(&self, layer: usize) -> usize;
    fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_;
    fn is_unsat(&self) -> bool;
}

/// `MddView` plus the branching order -- needed by `crate::mdd::wmc::partial_forward`/
/// `partial_backward` and `crate::sampling::bp::belief_propagation`, which index `weights` by
/// global variable id rather than by layer.
pub trait MddViewWithOrder: MddView {
    fn decision_at_layer(&self, layer: usize) -> VariableIndex;
}

impl MddView for super::Mdd {
    fn root(&self) -> NodeIndex {
        super::Mdd::root(self)
    }

    fn sink(&self) -> NodeIndex {
        super::Mdd::sink(self)
    }

    fn number_layers(&self) -> usize {
        super::Mdd::number_layers(self)
    }

    fn number_nodes_in_layer(&self, layer: usize) -> usize {
        super::Mdd::number_nodes_in_layer(self, layer)
    }

    fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        super::Mdd::nodes_in_layer(self, layer)
    }

    fn is_unsat(&self) -> bool {
        super::Mdd::is_unsat(self)
    }
}

impl MddViewWithOrder for super::Mdd {
    fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        super::Mdd::decision_at_layer(self, layer)
    }
}

impl MddView for MddStructure {
    fn root(&self) -> NodeIndex {
        MddStructure::root(self)
    }

    fn sink(&self) -> NodeIndex {
        MddStructure::sink(self)
    }

    fn number_layers(&self) -> usize {
        MddStructure::number_layers(self)
    }

    fn number_nodes_in_layer(&self, layer: usize) -> usize {
        MddStructure::number_nodes_in_layer(self, layer)
    }

    fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        MddStructure::nodes_in_layer(self, layer)
    }

    fn is_unsat(&self) -> bool {
        MddStructure::is_unsat(self)
    }
}
