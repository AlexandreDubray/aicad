//! Functions that build a constraint and add it to a [`Problem`], one per kind of constraint.

use super::{AllDifferent, Among, AtLeast, Gcc, NotEquals, Regular, Sum};
use crate::modelling::{ConstraintIndex, Problem, VariableIndex};
use rustc_hash::FxHashSet;

pub fn all_different(problem: &mut Problem, variables: Vec<VariableIndex>) -> ConstraintIndex {
    let constraint = AllDifferent::new(variables, problem);
    problem.add_constraint(constraint)
}

pub fn not_equals(problem: &mut Problem, x: VariableIndex, y: VariableIndex) -> ConstraintIndex {
    let constraint = NotEquals::new(x, y, problem);
    problem.add_constraint(constraint)
}

pub fn among(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    values: Vec<isize>,
    lb: usize,
    ub: usize,
) -> ConstraintIndex {
    problem.add_constraint(Among::new(
        variables,
        FxHashSet::from_iter(values.iter().cloned()),
        lb,
        ub,
    ))
}

pub fn sum(problem: &mut Problem, variables: Vec<VariableIndex>, target: isize) -> ConstraintIndex {
    problem.add_constraint(Sum::new(variables, target, problem))
}

/// At least `lb` of `variables` must take a value in `values`
pub fn at_least(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    values: Vec<isize>,
    lb: usize,
) -> ConstraintIndex {
    problem.add_constraint(AtLeast::new(
        variables,
        FxHashSet::from_iter(values.iter().cloned()),
        lb,
    ))
}

pub fn gcc(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    bounds: Vec<(isize, usize, usize)>,
) -> ConstraintIndex {
    problem.add_constraint(Gcc::new(variables, bounds))
}

/// Adds a regular constraint to the problem over the sequence of variables in `variables`.
/// The alphabet is the union of the variables domains, computed in the constraint constructor.
/// `transitions` is the allowed transition, `initial_state` and `accepting_states` index
/// the first dimension of `transitions`.
///
/// Note that when using a regular constraint, the compilation order *MUST* follow the order stated
/// in `variables`.
pub fn regular(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    transitions: Vec<Vec<Option<usize>>>,
    initial_state: usize,
    accepting_states: Vec<usize>,
) -> ConstraintIndex {
    problem.add_constraint(Regular::new(
        variables,
        transitions,
        initial_state,
        FxHashSet::from_iter(accepting_states.iter().copied()),
        problem,
    ))
}
