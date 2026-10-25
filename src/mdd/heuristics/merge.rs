//! How to choose the nodes that are merged when a layer is wider than allowed.
//!
//! # Purpose
//!
//! Refinement splits nodes until the layer holds the width it is allowed; whenever it goes over,
//! the nodes of the layer are ranked and the surplus is merged away. Merging two nodes keeps
//! every path of both, so the diagram only ever gains paths: it stays a relaxation. The heuristic
//! decides which nodes go together; `Mdd` performs the merges.
//!
//! # How it works
//!
//! `MergeHeuristic::group_nodes` returns the groups of nodes to merge, as many as the layer may
//! have nodes. Each group becomes one node, so any grouping is allowed: all the surplus nodes into
//! one, a few big groups next to singletons, or nodes clustered by similarity. The heuristics of
//! this module rank the nodes of the layer, most mergeable first, and cut the ranking into
//! `max_width` runs of consecutive nodes whose sizes differ by at most one: nodes that the
//! ranking finds alike end up in the same node, and the relaxation is spread over the layer
//! instead of piled into a single node.

use crate::mdd::{Compiler, EdgeIndex, NodeIndex};

/// The probability, in `compiler`, of the value that `edge` assigns to the variable of its layer.
fn edge_probability(compiler: &Compiler, edge: EdgeIndex) -> f64 {
    let mdd = compiler.mdd();
    let variable = mdd.decision_at_layer(edge.0);
    compiler.probability(variable, mdd[edge].assignment())
}

/// The strategy used to rank and merge the nodes of a layer that is too wide.
#[derive(Clone, Copy, Debug)]
pub enum MergeHeuristic {
    /// Merges nodes that are alike in how relaxed they are. Nodes are ranked by the share of their
    /// parents that are not relaxed, lowest first.
    LessRelaxed,
    /// Merges nodes that are alike in how likely they are. Nodes are ranked by the mean
    /// probability of their incoming edges, taken from the probabilities of the diagram (those
    /// of the variables unless `Compiler::set_probabilities` replaced them), lowest first.
    MostLikely,
    /// Merges nodes whose constraint properties are close. Nodes are sorted by the keys of their
    /// properties, constraint by constraint, so that similar states are neighbours.
    StateSimilarity,
}

impl MergeHeuristic {
    /// The active nodes of `layer`, ordered as described on each variant.
    fn rank_nodes(&self, compiler: &Compiler, layer: usize) -> Vec<NodeIndex> {
        let mdd = compiler.mdd();
        let active_nodes = || {
            (0..mdd.number_nodes_in_layer(layer))
                .map(move |position| NodeIndex(layer, position))
                .filter(|&node| mdd[node].is_active())
        };
        match self {
            Self::LessRelaxed => rank_by_score(active_nodes().map(|node| {
                let parents = mdd[node].number_parents() as f64;
                let exact_parents = mdd[node]
                    .iter_parents()
                    .map(|edge| mdd[edge].from())
                    .filter(|&parent| !mdd[parent].is_relaxed())
                    .count() as f64;
                (exact_parents / parents, node)
            })),
            Self::MostLikely => rank_by_score(active_nodes().map(|node| {
                let parents = mdd[node].number_parents() as f64;
                let probability_mass: f64 = mdd[node]
                    .iter_parents()
                    .map(|edge| edge_probability(compiler, edge))
                    .sum();
                (probability_mass / parents, node)
            })),
            Self::StateSimilarity => {
                // One composite key per node: every constraint's top-down `order_key()` then its
                // bottom-up `order_key()`, in the MDD's own constraint order. Sorting
                // lexicographically, rather than collapsing the keys to one weighted scalar,
                // keeps one constraint's scale (a Gcc key in the hundreds) from drowning
                // another's (a Regular popcount in the single digits): nodes that agree on more
                // constraints end up adjacent.
                let mut keyed: Vec<(Vec<f64>, NodeIndex)> = active_nodes()
                    .map(|node| {
                        let key = compiler
                            .top_down_properties(node)
                            .iter()
                            .zip(compiler.bottom_up_properties(node))
                            .flat_map(|(top_down, bottom_up)| {
                                top_down
                                    .order_key()
                                    .into_iter()
                                    .chain(bottom_up.order_key())
                            })
                            .collect();
                        (key, node)
                    })
                    .collect();
                keyed.sort_unstable_by(|a, b| {
                    for (x, y) in a.0.iter().zip(b.0.iter()) {
                        match x.total_cmp(y) {
                            std::cmp::Ordering::Equal => continue,
                            other => return other,
                        }
                    }
                    std::cmp::Ordering::Equal
                });
                keyed.into_iter().map(|(_, node)| node).collect()
            }
        }
    }

    /// The groups of nodes of `layer` to merge so that `max_width` nodes remain: exactly
    /// `max_width` groups that partition the active nodes of the layer. Each group is merged into
    /// its first node. Only to be called on a layer that has more active nodes than `max_width`;
    /// the nodes are not even ranked otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `max_width` is 0, since a layer cannot be merged into no node, or if the layer
    /// has no more than `max_width` active nodes.
    pub(crate) fn group_nodes(
        &self,
        compiler: &Compiler,
        layer: usize,
        max_width: usize,
    ) -> Vec<Vec<NodeIndex>> {
        let mdd = compiler.mdd();
        assert!(
            max_width >= 1,
            "a layer cannot be merged down to a width of 0 nodes"
        );
        let active_nodes = mdd
            .nodes_in_layer(layer)
            .filter(|&node| mdd[node].is_active())
            .count();
        assert!(
            active_nodes > max_width,
            "layer {layer} has {active_nodes} active nodes, which fits the width of {max_width}: \
             there is nothing to merge"
        );
        consecutive_runs(&self.rank_nodes(compiler, layer), max_width)
    }
}

/// Cuts `ranking`, which has more than `max_width` nodes, into `max_width` runs of consecutive
/// nodes, the first ones holding one node less than the last ones when the sizes cannot be equal.
fn consecutive_runs(ranking: &[NodeIndex], max_width: usize) -> Vec<Vec<NodeIndex>> {
    let nodes = ranking.len();
    debug_assert!(nodes > max_width && max_width >= 1);
    let (size, larger) = (nodes / max_width, nodes % max_width);
    let mut groups = Vec::with_capacity(max_width);
    let mut start = 0;
    for run in 0..max_width {
        // `max_width - larger` runs of `size` nodes, then `larger` runs of `size + 1`.
        let end = start + size + usize::from(run >= max_width - larger);
        groups.push(ranking[start..end].to_vec());
        start = end;
    }
    groups
}

/// Orders nodes by increasing score.
fn rank_by_score(scored: impl Iterator<Item = (f64, NodeIndex)>) -> Vec<NodeIndex> {
    let mut scored: Vec<(f64, NodeIndex)> = scored.collect();
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().map(|(_, node)| node).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, gcc};
    use crate::mdd::Compiler;
    use crate::mdd::heuristics::{CompilationHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::modelling::Problem;
    use std::sync::Arc;

    const ALL: [MergeHeuristic; 3] = [
        MergeHeuristic::LessRelaxed,
        MergeHeuristic::MostLikely,
        MergeHeuristic::StateSimilarity,
    ];

    fn nodes(count: usize) -> Vec<NodeIndex> {
        (0..count).map(|i| NodeIndex(1, i)).collect()
    }

    #[test]
    #[should_panic(expected = "there is nothing to merge")]
    fn a_layer_that_fits_is_not_ranked() {
        let compiler = exact_mdd(MergeHeuristic::LessRelaxed);
        let mdd = compiler.mdd();
        let active = mdd
            .nodes_in_layer(2)
            .filter(|&n| mdd[n].is_active())
            .count();
        MergeHeuristic::LessRelaxed.group_nodes(&compiler, 2, active);
    }

    #[test]
    fn runs_are_consecutive_and_the_larger_ones_come_last() {
        let ranking = nodes(7);
        assert_eq!(
            consecutive_runs(&ranking, 3),
            vec![
                vec![ranking[0], ranking[1]],
                vec![ranking[2], ranking[3]],
                vec![ranking[4], ranking[5], ranking[6]],
            ]
        );
    }

    #[test]
    fn a_width_of_one_merges_everything_into_one_node() {
        let ranking = nodes(5);
        assert_eq!(consecutive_runs(&ranking, 1), vec![ranking]);
    }

    #[test]
    fn the_runs_partition_the_ranking_in_order_and_differ_by_at_most_one() {
        for count in 2..=14 {
            for width in 1..count {
                let ranking = nodes(count);
                let runs = consecutive_runs(&ranking, width);
                assert_eq!(runs.len(), width, "{count}/{width}");
                assert_eq!(runs.concat(), ranking, "{count}/{width}");
                let (smallest, largest) = runs
                    .iter()
                    .map(Vec::len)
                    .fold((usize::MAX, 0), |(lo, hi), len| (lo.min(len), hi.max(len)));
                assert!(largest - smallest <= 1, "{count}/{width}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "width of 0 nodes")]
    fn merging_down_to_no_node_is_refused() {
        let compiler = exact_mdd(MergeHeuristic::LessRelaxed);
        MergeHeuristic::LessRelaxed.group_nodes(&compiler, 1, 0);
    }

    /// A diagram that has not been refined yet.
    fn exact_mdd_unrefined(heuristic: MergeHeuristic) -> Compiler {
        let mut problem = Problem::default();
        let vars = problem.add_variables(5, vec![0, 1, 2, 3, 4], None);
        let all_different = all_different(&mut problem, vars.clone());
        let counted = gcc(&mut problem, vars, vec![(0, 0, 2), (1, 0, 2)]);
        Compiler::new(
            Arc::new(problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: heuristic,
                selection: SelectHeuristic::Greedy,
            },
            &[all_different, counted],
        )
    }

    /// An exact diagram: layers with several nodes to rank.
    fn exact_mdd(heuristic: MergeHeuristic) -> Compiler {
        let mut compiler = exact_mdd_unrefined(heuristic);
        compiler.refine(usize::MAX);
        let mdd = compiler.mdd();
        assert!(!mdd.is_unsat() && mdd.count_solutions() > 0u32.into());
        compiler
    }

    #[test]
    fn the_groups_partition_the_active_nodes_of_the_layer() {
        for heuristic in ALL {
            let compiler = exact_mdd(heuristic);
            let mdd = compiler.mdd();
            for layer in 1..mdd.number_layers() - 1 {
                let active: Vec<_> = mdd
                    .nodes_in_layer(layer)
                    .filter(|&n| mdd[n].is_active())
                    .collect();
                for width in [1, 2, active.len().saturating_sub(1).max(1)] {
                    if active.len() <= width {
                        continue;
                    }
                    let groups = heuristic.group_nodes(&compiler, layer, width);
                    assert_eq!(groups.len(), width, "{heuristic:?} layer {layer}");
                    let mut merged = groups.concat();
                    merged.sort();
                    assert_eq!(merged, active, "{heuristic:?} layer {layer} width {width}");
                }
            }
        }
    }

    #[test]
    fn a_relaxation_keeps_the_width_and_every_solution_whatever_the_heuristic() {
        let exact: u64 = exact_mdd(MergeHeuristic::LessRelaxed)
            .mdd()
            .count_solutions()
            .try_into()
            .unwrap();
        for heuristic in ALL {
            for width in [1, 2, 3, 6] {
                let mut compiler = exact_mdd_unrefined(heuristic);
                compiler.refine(width);
                let mdd = compiler.mdd();
                assert!(
                    (0..mdd.number_layers()).all(|l| mdd.number_nodes_in_layer(l) <= width),
                    "{heuristic:?} width {width}"
                );
                let relaxed: u64 = mdd.count_solutions().try_into().unwrap();
                assert!(relaxed >= exact, "{heuristic:?} width {width}");
            }
        }
    }

    #[test]
    fn a_ranking_lists_each_active_node_of_the_layer_once() {
        for heuristic in ALL {
            let compiler = exact_mdd(heuristic);
            let mdd = compiler.mdd();
            for layer in 0..mdd.number_layers() {
                let mut ranking = heuristic.rank_nodes(&compiler, layer);
                ranking.sort();
                let active: Vec<_> = mdd
                    .nodes_in_layer(layer)
                    .filter(|&n| mdd[n].is_active())
                    .collect();
                assert_eq!(ranking, active, "{heuristic:?} layer {layer}");
            }
        }
    }
}
