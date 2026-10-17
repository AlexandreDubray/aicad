//! The read-only surface that the MDD algorithms need, shared by `Mdd` and `MddStructure`.
//!
//! # Purpose
//!
//! Weighted model counting (`wmc`) and belief propagation only read the shape of a diagram: its
//! layers, nodes, edges, root and sink. They are written against [`MddView`] (and
//! [`MddViewWithOrder`] when they also need to know which variable each layer branches on), so the
//! same code runs on a self-contained `Mdd` and on the shared `MddStructure` of the arena.
//!
//! # Contract
//!
//! An implementation indexes nodes and edges by `NodeIndex` and `EdgeIndex`; the root is the only
//! node of layer 0, the sink the only node of the last layer, and `number_layers` counts the sink's
//! layer (one more than the number of variables). Both implementations forward to the methods of
//! the same name, so the two views of one compiled diagram answer identically.

use super::{Edge, MddStructure, Node, NodeIndex};
use crate::modelling::VariableIndex;

/// The structure of a diagram: layers, nodes and edges, without the variables they stand for.
pub trait MddView:
    std::ops::Index<NodeIndex, Output = Node> + std::ops::Index<super::EdgeIndex, Output = Edge>
{
    /// The only node of the first layer.
    fn root(&self) -> NodeIndex;
    /// The only node of the last layer.
    fn sink(&self) -> NodeIndex;
    /// Number of layers, the sink's included.
    fn number_layers(&self) -> usize;
    /// Number of nodes in `layer`.
    fn number_nodes_in_layer(&self, layer: usize) -> usize;
    /// Iterates over the nodes of `layer`.
    fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_;
    /// `true` if the diagram accepts no assignment.
    fn is_unsat(&self) -> bool;
}

/// An [`MddView`] that also knows which variable each layer branches on.
///
/// Needed by the algorithms that index their weights by variable rather than by layer.
pub trait MddViewWithOrder: MddView {
    /// The variable branched on at `layer` (every layer but the sink's).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use crate::mdd::Mdd;
    use crate::mdd::heuristics::*;
    use crate::modelling::Problem;
    use std::sync::Arc;

    fn compiled(unsat: bool) -> Mdd {
        let mut problem = Problem::default();
        let domain = if unsat { vec![0] } else { vec![0, 1, 2] };
        let vars = problem.add_variables(3, domain, None);
        let c = all_different(&mut problem, vars);
        let mut mdd = Mdd::new(
            Arc::new(problem),
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &[c],
        );
        mdd.refine(usize::MAX);
        mdd
    }

    /// Everything a view says about the diagram, as text so that two views can be compared.
    fn describe<V: MddView>(view: &V) -> String {
        let layers = view.number_layers();
        let nodes: Vec<Vec<NodeIndex>> = (0..layers)
            .map(|l| view.nodes_in_layer(l).collect())
            .collect();
        let widths: Vec<usize> = (0..layers).map(|l| view.number_nodes_in_layer(l)).collect();
        format!(
            "{:?} {:?} {} {} {:?} {:?}",
            view.root(),
            view.sink(),
            layers,
            view.is_unsat(),
            widths,
            nodes
        )
    }

    #[test]
    fn a_diagram_and_its_extracted_structure_answer_identically() {
        for unsat in [false, true] {
            let mdd = compiled(unsat);
            assert_eq!(mdd.is_unsat(), unsat);
            let before = describe(&mdd);
            let structure = mdd.into_structure();
            assert_eq!(before, describe(&structure), "unsat = {unsat}");
        }
    }

    #[test]
    fn the_first_layer_holds_the_root_and_the_last_the_sink() {
        let mdd = compiled(false);
        assert_eq!(MddView::root(&mdd), NodeIndex(0, 0));
        assert_eq!(MddView::number_layers(&mdd), 4);
        assert_eq!(MddView::sink(&mdd).0, MddView::number_layers(&mdd) - 1);
        assert_eq!(MddView::number_nodes_in_layer(&mdd, 0), 1);
        assert_eq!(MddView::number_nodes_in_layer(&mdd, 3), 1);
        assert!(!MddView::is_unsat(&mdd));
    }

    #[test]
    fn the_order_view_names_each_variable_once() {
        let mdd = compiled(false);
        let mut seen: Vec<_> = (0..MddView::number_layers(&mdd) - 1)
            .map(|l| *MddViewWithOrder::decision_at_layer(&mdd, l))
            .collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
    }
}
