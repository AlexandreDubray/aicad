//! How to choose the node that refinement splits next.
//!
//! # Purpose
//!
//! A node can stand for several paths that the constraints would tell apart. Refining the diagram
//! means splitting such a node into its more precise parts, within the width allowed. The
//! heuristic only names the node; `Compiler::refine` does the splitting. Only splittable nodes can
//! be selected (`Compiler::is_splittable`): those with several parent edges that do not all bring
//! the same properties. When a layer has none there is nothing to gain from refining it.

use crate::mdd::*;

/// The strategy used to select the node to split in a layer.
#[derive(Clone, Copy, Debug)]
pub enum SelectHeuristic {
    /// Takes the first splittable node of the layer, in the order the layer stores its nodes.
    Greedy,
}

impl SelectHeuristic {
    /// The node of `layer` to split next, or `None` if the layer has no splittable node.
    pub(crate) fn select_node(&self, compiler: &Compiler, layer: usize) -> Option<NodeIndex> {
        let mdd = compiler.mdd();
        match self {
            Self::Greedy => {
                for index in 0..mdd.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, index);
                    if compiler.is_splittable(node) {
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
    use crate::mdd::Compiler;
    use crate::mdd::heuristics::{CompilationHeuristic, MergeHeuristic, OrderingHeuristic};
    use crate::modelling::Problem;
    use std::sync::Arc;

    fn all_different_mdd() -> Compiler {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        let c = all_different(&mut problem, vars);
        Compiler::new(
            Arc::new(problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &[c],
        )
    }

    #[test]
    fn it_names_a_splittable_node_of_the_layer_while_the_diagram_is_coarse() {
        let compiler = all_different_mdd();
        let mdd = compiler.mdd();
        let selected: Vec<_> = (1..mdd.number_layers() - 1)
            .filter_map(|layer| SelectHeuristic::Greedy.select_node(&compiler, layer))
            .collect();
        assert!(!selected.is_empty());
        for node in selected {
            assert!(compiler.is_splittable(node) && mdd[node].is_active());
            // A node that mixes states is relaxed.
            assert!(mdd[node].is_relaxed());
        }
    }

    #[test]
    fn it_finds_nothing_to_split_once_the_diagram_is_exact() {
        let mut compiler = all_different_mdd();
        compiler.refine(usize::MAX);
        let mdd = compiler.mdd();
        for layer in 0..mdd.number_layers() {
            assert_eq!(SelectHeuristic::Greedy.select_node(&compiler, layer), None);
        }
    }

    #[test]
    fn it_takes_the_first_splittable_node_by_position() {
        let compiler = all_different_mdd();
        let mdd = compiler.mdd();
        for layer in 1..mdd.number_layers() - 1 {
            if let Some(NodeIndex(_, position)) =
                SelectHeuristic::Greedy.select_node(&compiler, layer)
            {
                assert!((0..position).all(|p| !compiler.is_splittable(NodeIndex(layer, p))));
            }
        }
    }
}
