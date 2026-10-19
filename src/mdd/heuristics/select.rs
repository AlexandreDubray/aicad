//! How to choose the node that refinement splits next.
//!
//! # Purpose
//!
//! A relaxed node stands for several paths that the constraints would tell apart. Refining the
//! diagram means splitting such a node into its more precise parts, within the width allowed.
//! The heuristic only names the node; `Mdd::refine` does the splitting. Only relaxed nodes can be
//! selected: when a layer has none, it is exact and there is nothing to refine.

use crate::mdd::*;

/// The strategy used to select the node to split in a layer.
#[derive(Clone, Copy, Debug)]
pub enum SelectHeuristic {
    /// Takes the first relaxed node of the layer, in the order the layer stores its nodes.
    Greedy,
}

impl SelectHeuristic {
    /// The node of `layer` to split next, or `None` if the layer has no relaxed node.
    pub(crate) fn select_node(&self, mdd: &Mdd, layer: usize) -> Option<NodeIndex> {
        match self {
            Self::Greedy => {
                for index in 0..mdd.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, index);
                    if mdd[node].is_relaxed() {
                        return Some(node);
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic};
    use crate::modelling::Problem;
    use std::sync::Arc;

    fn all_different_mdd() -> Mdd {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        let c = all_different(&mut problem, vars);
        Mdd::new(
            Arc::new(problem),
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &[c],
        )
    }

    #[test]
    fn it_names_a_relaxed_node_of_the_layer_while_the_diagram_is_coarse() {
        let mdd = all_different_mdd();
        let selected: Vec<_> = (1..mdd.number_layers() - 1)
            .filter_map(|layer| SelectHeuristic::Greedy.select_node(&mdd, layer))
            .collect();
        assert!(!selected.is_empty());
        for node in selected {
            assert!(mdd[node].is_relaxed() && mdd[node].is_active());
        }
    }

    #[test]
    fn it_finds_nothing_to_split_once_the_diagram_is_exact() {
        let mut mdd = all_different_mdd();
        mdd.refine(usize::MAX);
        // Refinement never looks at the sink's layer, which keeps its initial relaxed flag.
        for layer in 0..mdd.number_layers() - 1 {
            assert_eq!(SelectHeuristic::Greedy.select_node(&mdd, layer), None);
        }
    }

    #[test]
    fn it_takes_the_first_relaxed_node_by_position() {
        let mdd = all_different_mdd();
        for layer in 1..mdd.number_layers() - 1 {
            if let Some(NodeIndex(_, position)) = SelectHeuristic::Greedy.select_node(&mdd, layer) {
                assert!((0..position).all(|p| !mdd[NodeIndex(layer, p)].is_relaxed()));
            }
        }
    }
}
