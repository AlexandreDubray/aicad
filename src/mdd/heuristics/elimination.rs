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
    /// Computes an elimination order over `problem`'s variables using this heuristic, assigns
    /// every constraint to the bucket of its earliest-eliminated scope variable, then greedily
    /// packs each bucket into mini-buckets whose union scope stays within `size_bound` variables
    /// (mini-bucket elimination; Dechter & Rish, "Mini-buckets: A general scheme for bounded
    /// inference", JACM 2003).
    ///
    /// `size_bound == 0` is a special case: every constraint is placed in its own singleton
    /// bucket and no elimination-order-driven merging is attempted at all. This is an exact,
    /// structural guarantee of one-MDD-per-constraint compilation.
    ///
    /// Buckets with no constraints are omitted from the result.
    pub fn buckets(&self, problem: &Problem, size_bound: usize) -> Vec<Vec<ConstraintIndex>> {
        if size_bound == 0 {
            return problem.iter_constraints().map(|c| vec![c]).collect();
        }
        match self {
            Self::GreedyMinFill => greedy_min_fill_buckets(problem, size_bound),
        }
    }
}

fn greedy_min_fill_buckets(problem: &Problem, size_bound: usize) -> Vec<Vec<ConstraintIndex>> {
    let order = min_fill_order(problem);
    let position: FxHashMap<VariableIndex, usize> = order
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    // Assign each constraint to the bucket of its earliest-eliminated scope variable.
    let mut raw_buckets: Vec<Vec<ConstraintIndex>> = vec![Vec::new(); order.len()];
    for constraint in problem.iter_constraints() {
        let earliest = problem[constraint]
            .iter_scope()
            .min_by_key(|v| position[v])
            .expect("a compiled constraint must have a non-empty scope");
        raw_buckets[position[&earliest]].push(constraint);
    }

    raw_buckets
        .into_iter()
        .filter(|bucket| !bucket.is_empty())
        .flat_map(|bucket| split_into_mini_buckets(problem, bucket, size_bound))
        .collect()
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

fn split_into_mini_buckets(
    problem: &Problem,
    mut constraints: Vec<ConstraintIndex>,
    size_bound: usize,
) -> Vec<Vec<ConstraintIndex>> {
    constraints.sort_by_key(|&c| std::cmp::Reverse(problem[c].iter_scope().count()));

    let mut mini_buckets: Vec<(FxHashSet<VariableIndex>, Vec<ConstraintIndex>)> = Vec::new();
    for constraint in constraints {
        let scope: FxHashSet<VariableIndex> = problem[constraint].iter_scope().collect();

        let mut best: Option<(usize, usize)> = None; // (overlap, mini_bucket_index)
        for (i, (existing_scope, _)) in mini_buckets.iter().enumerate() {
            let union_size = existing_scope.union(&scope).count();
            if union_size <= size_bound {
                let overlap = existing_scope.intersection(&scope).count();
                let better = match best {
                    None => true,
                    Some((best_overlap, _)) => overlap > best_overlap,
                };
                if better {
                    best = Some((overlap, i));
                }
            }
        }

        match best {
            Some((_, i)) => {
                mini_buckets[i].0.extend(scope);
                mini_buckets[i].1.push(constraint);
            }
            None => mini_buckets.push((scope, vec![constraint])),
        }
    }

    mini_buckets.into_iter().map(|(_, group)| group).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::{all_different, not_equals};

    #[test]
    fn size_bound_zero_gives_one_singleton_bucket_per_constraint() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);

        let buckets = EliminationOrdering::GreedyMinFill.buckets(&problem, 0);
        assert_eq!(buckets.len(), 2);
        for bucket in &buckets {
            assert_eq!(bucket.len(), 1);
        }
    }

    #[test]
    fn triangle_gets_merged_into_one_bucket_when_the_bound_allows_it() {
        // x-y, y-z, x-z: a triangle. min-fill will eliminate one variable first, pulling both of
        // its incident constraints into the same bucket; with a bound of 3 that bucket's union
        // scope ({x,y,z}) fits, so the merge should actually happen instead of being capped.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let buckets = EliminationOrdering::GreedyMinFill.buckets(&problem, 3);
        // All 3 variables are mutually connected, so eliminating any one of them first pulls its
        // two incident constraints into a single bucket of size 2; the third constraint (between
        // the two remaining variables) lands in its own bucket next. Total constraints preserved.
        let total: usize = buckets.iter().map(|b| b.len()).sum();
        assert_eq!(total, 3);
        assert!(buckets.iter().any(|b| b.len() == 2));
    }

    #[test]
    fn a_wide_constraint_gets_its_own_bucket_even_under_a_tight_bound() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(5, vec![0, 1, 2, 3, 4], None);
        all_different(&mut problem, vars.clone());

        // The bound is smaller than the constraint's own scope (5); it must still be compiled
        // whole, just not merged with anything else.
        let buckets = EliminationOrdering::GreedyMinFill.buckets(&problem, 2);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].len(), 1);
    }

    #[test]
    fn every_constraint_is_covered_exactly_once() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        let w = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, z, w);
        not_equals(&mut problem, w, x);

        for bound in [0, 1, 2, 3, 4, 8] {
            let buckets = EliminationOrdering::GreedyMinFill.buckets(&problem, bound);
            let mut seen: Vec<ConstraintIndex> = buckets.into_iter().flatten().collect();
            seen.sort_by_key(|c| c.0);
            let expected: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            assert_eq!(seen, expected, "bound={bound}");
        }
    }
}
