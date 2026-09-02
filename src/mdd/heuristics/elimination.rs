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

    /// Sorts every constraint by the elimination position (under this heuristic's order) of the
    /// earliest-eliminated variable in its scope, then slides a stride-1 window of `window_size`
    /// constraints across that sorted list, producing one (generally overlapping) group per window
    /// position -- `[0, window_size)`, `[1, window_size + 1)`, and so on.
    ///
    /// Unlike `buckets`, groups here are **not** disjoint: a constraint can appear in up to
    /// `window_size` different groups (once per window it falls in), so compiling one MDD per group
    /// and reusing all of them for e.g. belief propagation double-counts that constraint's evidence
    /// once per extra membership. This is a deliberate accuracy/completeness tradeoff: because a
    /// good elimination order tends to place a tightly-connected clique's variables close together,
    /// a clique's constraints usually end up sharing at least one window even when no single
    /// variable is the earliest-eliminated endpoint of all of them (the structural gap `buckets`
    /// has for e.g. odd cycles), without requiring true bucket-elimination message passing.
    ///
    /// `window_size <= 1` degenerates to one singleton group per constraint (no overlap, same as
    /// `buckets` with `size_bound == 0`). A `window_size` at or above the number of constraints
    /// yields a single group containing everything.
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
    fn window_size_at_most_one_gives_one_singleton_group_per_constraint() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);

        for window_size in [0, 1] {
            let groups = EliminationOrdering::GreedyMinFill.rolling_window_groups(&problem, window_size);
            assert_eq!(groups.len(), 2);
            for group in &groups {
                assert_eq!(group.len(), 1);
            }
        }
    }

    #[test]
    fn window_size_covering_everything_yields_a_single_group() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        for window_size in [3, 4, 100] {
            let groups =
                EliminationOrdering::GreedyMinFill.rolling_window_groups(&problem, window_size);
            assert_eq!(groups.len(), 1);
            assert_eq!(groups[0].len(), 3);
        }
    }

    #[test]
    fn windows_overlap_by_window_size_minus_one() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2, 3], None);
        let y = problem.add_variable(vec![0, 1, 2, 3], None);
        let z = problem.add_variable(vec![0, 1, 2, 3], None);
        let w = problem.add_variable(vec![0, 1, 2, 3], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, z, w);
        not_equals(&mut problem, w, x);

        let groups = EliminationOrdering::GreedyMinFill.rolling_window_groups(&problem, 2);
        // 4 constraints, window_size 2 -> 3 sliding windows, each of size 2.
        assert_eq!(groups.len(), 3);
        for group in &groups {
            assert_eq!(group.len(), 2);
        }
        // Consecutive windows share exactly one constraint (the overlap), a hallmark of a
        // stride-1 sliding window rather than disjoint chunking.
        for pair in groups.windows(2) {
            let shared = pair[0].iter().filter(|c| pair[1].contains(c)).count();
            assert_eq!(shared, 1);
        }
    }

    #[test]
    fn a_triangle_shares_a_window_even_though_no_single_bucket_can_hold_it() {
        // A 3-cycle of pairwise not_equals constraints can never land in the same *bucket* (see
        // `triangle_gets_merged_into_one_bucket_when_the_bound_allows_it`'s sibling limitation:
        // that only holds because min-fill happens to eliminate a triangle vertex first; for an
        // odd cycle in general, no single earliest-eliminated variable covers every edge). A
        // rolling window with the same width as the triangle's constraint count doesn't have that
        // restriction: sorted by elimination position, the 3 edges are all close together, so a
        // window of size 3 over 3 total constraints simply contains all of them.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let groups = EliminationOrdering::GreedyMinFill.rolling_window_groups(&problem, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
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

    #[test]
    fn rolling_window_groups_cover_every_constraint_at_least_once() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        let w = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, z, w);
        not_equals(&mut problem, w, x);

        for window_size in [0, 1, 2, 3, 4, 8] {
            let groups =
                EliminationOrdering::GreedyMinFill.rolling_window_groups(&problem, window_size);
            let mut seen: Vec<ConstraintIndex> = groups.into_iter().flatten().collect();
            seen.sort_by_key(|c| c.0);
            seen.dedup();
            let expected: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            assert_eq!(seen, expected, "window_size={window_size}");
        }
    }
}
