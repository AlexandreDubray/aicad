use std::collections::BTreeSet;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::modelling::{ConstraintIndex, Problem, VariableIndex};

/// How to choose the variable elimination order that drives mini-bucket construction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EliminationOrdering {
    /// At each step, eliminate whichever remaining variable would add the fewest new edges among
    /// its still-remaining neighbors in the (evolving) primal constraint graph.
    GreedyMinFill,
}

impl EliminationOrdering {
    pub fn rolling_window_groups(
        &self,
        problem: &Problem,
        window_size: usize,
    ) -> Vec<Vec<ConstraintIndex>> {
        if window_size <= 1 {
            return problem.iter_constraints().map(|c| vec![c]).collect();
        }

        let order = match self {
            Self::GreedyMinFill => min_fill_order(problem),
        };
        let position: FxHashMap<VariableIndex, usize> = order
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();

        let mut sorted: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        sorted.sort_by_key(|&c| {
            let earliest = problem[c]
                .iter_scope()
                .map(|v| position[&v])
                .min()
                .expect("a compiled constraint must have a non-empty scope");
            (earliest, c.0)
        });

        let n = sorted.len();
        if n == 0 {
            return Vec::new();
        }
        let window_size = window_size.min(n);
        (0..=(n - window_size))
            .map(|start| sorted[start..start + window_size].to_vec())
            .collect()
    }
}

fn min_fill_order(problem: &Problem) -> Vec<VariableIndex> {
    let n = problem.number_variables();
    let mut neighbors: Vec<FxHashSet<VariableIndex>> = vec![FxHashSet::default(); n];
    for constraint in problem.iter_constraints() {
        let scope: Vec<VariableIndex> = problem[constraint].iter_scope().collect();
        for &u in &scope {
            for &w in &scope {
                if u != w {
                    neighbors[u.0].insert(w);
                }
            }
        }
    }

    let mut remaining: BTreeSet<usize> = (0..n).collect();
    let mut order = Vec::with_capacity(n);

    let remaining_neighbors = |neighbors: &[FxHashSet<VariableIndex>],
                               remaining: &BTreeSet<usize>,
                               v: usize|
     -> Vec<usize> {
        neighbors[v]
            .iter()
            .map(|w| w.0)
            .filter(|w| remaining.contains(w))
            .collect()
    };

    while !remaining.is_empty() {
        let mut best: Option<(usize, usize)> = None; // (fill_count, variable_id)
        for &v in &remaining {
            let ns = remaining_neighbors(&neighbors, &remaining, v);
            let mut fill = 0usize;
            for i in 0..ns.len() {
                for j in (i + 1)..ns.len() {
                    if !neighbors[ns[i]].contains(&VariableIndex(ns[j])) {
                        fill += 1;
                    }
                }
            }
            let better = match best {
                None => true,
                Some((best_fill, best_v)) => fill < best_fill || (fill == best_fill && v < best_v),
            };
            if better {
                best = Some((fill, v));
            }
        }
        let (_, chosen) = best.expect("remaining is non-empty");

        // Add fill edges among `chosen`'s still-remaining neighbors, then remove `chosen`.
        let ns = remaining_neighbors(&neighbors, &remaining, chosen);
        for i in 0..ns.len() {
            for j in (i + 1)..ns.len() {
                let (a, b) = (ns[i], ns[j]);
                neighbors[a].insert(VariableIndex(b));
                neighbors[b].insert(VariableIndex(a));
            }
        }
        remaining.remove(&chosen);
        order.push(VariableIndex(chosen));
    }

    order
}
