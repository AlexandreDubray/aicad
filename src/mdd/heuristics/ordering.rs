//! How to choose the variable branched on at each layer of an MDD.
//!
//! # Purpose
//!
//! The order of the layers decides how many nodes an exact diagram needs and how well a
//! relaxation of a given width approximates it. The heuristic receives the variables in the scope
//! of the MDD (the union of the scopes of the constraints being compiled) and returns them in
//! branching order, first layer first.
//!
//! # How it works
//!
//! [`OrderingHeuristic::MinDomMaxLinked`] builds the order greedily:
//!
//! 1. Variables with a single value are placed first, since they cannot widen the diagram. (Their
//!    relative order is the reverse of the one given.)
//! 2. Each placed variable raises the score of every variable that shares a constraint with it,
//!    by one per shared constraint scope it appears in (itself included).
//! 3. The next variable is the one with the highest score; ties go to the variable in the most
//!    constraints, then to the first one in the remaining list.
//!
//! The first variable chosen after step 1 is therefore the one in the most constraints, and the
//! order then stays inside the part of the constraint graph already visited.
//!
//! # Design notes
//!
//! - **The result is a starting point.** It knows nothing about constraints that need a specific
//!   sequence (such as `Regular`); `Mdd::new` repairs the order for them afterwards.

use crate::modelling::{Problem, VariableIndex};
use rustc_hash::FxHashSet;

/// The strategy used to order the layers of an MDD.
#[derive(Clone, Debug)]
pub enum OrderingHeuristic {
    /// Greedy order that favours variables connected to those already placed; see the module
    /// documentation.
    MinDomMaxLinked,
    /// The caller's own order, as variable indices from the first layer to the last. It must list
    /// every variable of the MDD's scope exactly once and no other: anything else makes
    /// compilation panic with the offending variable.
    Custom(Vec<usize>),
}

impl OrderingHeuristic {
    /// Orders `variables`, the scope of the MDD, from the first layer to the last.
    ///
    /// # Panics
    ///
    /// Panics for [`OrderingHeuristic::Custom`] if the order is not a permutation of `variables`.
    pub(crate) fn get_order(
        &self,
        problem: &Problem,
        variables: &[VariableIndex],
    ) -> Vec<VariableIndex> {
        match self {
            Self::Custom(order) => {
                check_permutation(order, variables);
                order.iter().copied().map(VariableIndex).collect()
            }
            Self::MinDomMaxLinked => {
                let mut candidates = variables.to_vec();
                let mut scores = vec![0; problem.number_variables()];
                let mut order: Vec<VariableIndex> = vec![];
                for i in (0..candidates.len()).rev() {
                    let candidate = candidates[i];
                    if problem[candidate].domain_size() == 1 {
                        order.push(candidate);
                        for constraint in problem[candidate].iter_constraints() {
                            for linked_variable in problem[constraint].iter_scope() {
                                scores[linked_variable.0] += 1;
                            }
                        }
                        candidates.swap_remove(i);
                    }
                }
                for _ in 0..candidates.len() {
                    let mut best_index = 0;
                    let mut best_score = 0;
                    let mut best_tie = 0;
                    for (index, candidate) in candidates.iter().copied().enumerate() {
                        let score = scores[candidate.0];
                        let tie = problem[candidate].number_constraints();
                        if score > best_score || (score == best_score && tie > best_tie) {
                            best_index = index;
                            best_score = score;
                            best_tie = tie;
                        }
                    }
                    let selected = candidates[best_index];
                    order.push(selected);
                    for constraint in problem[selected].iter_constraints() {
                        for linked_variable in problem[constraint].iter_scope() {
                            scores[linked_variable.0] += 1;
                        }
                    }
                    candidates.swap_remove(best_index);
                }
                order
            }
        }
    }
}

/// Checks that `order` lists every variable of `scope` exactly once and no other.
///
/// # Panics
///
/// Panics naming the first variable that is repeated, outside the scope, or missing.
fn check_permutation(order: &[usize], scope: &[VariableIndex]) {
    let mut missing: FxHashSet<usize> = scope.iter().map(|v| v.0).collect();
    for &variable in order {
        assert!(
            missing.remove(&variable),
            "custom variable order: variable {variable} is listed twice or is not in the scope \
             of the MDD being compiled (the order must list each variable of the scope once)"
        );
    }
    if let Some(variable) = missing.into_iter().min() {
        panic!(
            "custom variable order: variable {variable} is in the scope of the MDD being compiled \
             but missing from the order (the order must list each variable of the scope once)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, not_equals};

    fn scope(problem: &Problem) -> Vec<VariableIndex> {
        (0..problem.number_variables()).map(VariableIndex).collect()
    }

    /// Variables 0 and 1 are in two constraints, 2 and 3 in one, 4 in none that is shared.
    fn problem_with_a_hub() -> Problem {
        let mut problem = Problem::default();
        let v = problem.add_variables(5, vec![0, 1, 2], None);
        all_different(&mut problem, vec![v[0], v[1], v[2]]);
        all_different(&mut problem, vec![v[0], v[1], v[3]]);
        not_equals(&mut problem, v[3], v[4]);
        problem
    }

    #[test]
    fn the_greedy_order_is_a_permutation_of_the_scope() {
        let problem = problem_with_a_hub();
        let scope = scope(&problem);
        let mut order = OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &scope);
        order.sort();
        assert_eq!(order, scope);
    }

    #[test]
    fn the_greedy_order_only_covers_the_variables_it_is_given() {
        let problem = problem_with_a_hub();
        let given = vec![VariableIndex(2), VariableIndex(3), VariableIndex(4)];
        let mut order = OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &given);
        order.sort();
        assert_eq!(order, given);
    }

    #[test]
    fn the_greedy_order_starts_with_the_variable_in_most_constraints() {
        let problem = problem_with_a_hub();
        let order = OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &scope(&problem));
        assert_eq!(order[0], VariableIndex(0));
    }

    #[test]
    fn the_greedy_order_places_single_valued_variables_first() {
        let mut problem = Problem::default();
        let wide = problem.add_variables(3, vec![0, 1, 2], None);
        let fixed = problem.add_variable(vec![7], None);
        all_different(&mut problem, vec![wide[0], wide[1], wide[2], fixed]);
        let order = OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &scope(&problem));
        assert_eq!(order[0], fixed);
    }

    #[test]
    fn the_greedy_order_is_deterministic() {
        let problem = problem_with_a_hub();
        let scope = scope(&problem);
        let first = OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &scope);
        assert_eq!(
            first,
            OrderingHeuristic::MinDomMaxLinked.get_order(&problem, &scope)
        );
    }

    #[test]
    fn a_custom_order_is_returned_unchanged() {
        let problem = problem_with_a_hub();
        let order =
            OrderingHeuristic::Custom(vec![4, 2, 0, 3, 1]).get_order(&problem, &scope(&problem));
        assert_eq!(order, [4, 2, 0, 3, 1].map(VariableIndex));
    }

    #[test]
    #[should_panic(expected = "variable 9 is listed twice or is not in the scope")]
    fn a_custom_order_with_a_variable_outside_the_scope_is_refused() {
        let problem = problem_with_a_hub();
        OrderingHeuristic::Custom(vec![0, 1, 2, 3, 9]).get_order(&problem, &scope(&problem));
    }

    #[test]
    #[should_panic(expected = "variable 1 is listed twice or is not in the scope")]
    fn a_custom_order_that_repeats_a_variable_is_refused() {
        let problem = problem_with_a_hub();
        OrderingHeuristic::Custom(vec![0, 1, 1, 2, 3]).get_order(&problem, &scope(&problem));
    }

    #[test]
    #[should_panic(expected = "variable 4 is in the scope of the MDD being compiled but missing")]
    fn a_custom_order_that_leaves_a_variable_out_is_refused() {
        let problem = problem_with_a_hub();
        OrderingHeuristic::Custom(vec![0, 1, 2, 3]).get_order(&problem, &scope(&problem));
    }
}
