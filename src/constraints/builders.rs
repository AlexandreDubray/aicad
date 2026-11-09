//! Functions that build a constraint and add it to a [`Problem`], one per kind of constraint.

use super::{
    AllDifferent, Among, AtLeast, Disjunctive, Gcc, NotEquals, Regular, SetupTimes, Sum, Table,
};
use crate::modelling::{ConstraintIndex, Problem, VariableIndex};
use rustc_hash::FxHashSet;

/// Adds [`AllDifferent`] over `variables` to `problem` and returns its index.
///
/// # Panics
///
/// As [`AllDifferent::new`]: a repeated variable is accepted with a warning and makes the
/// problem unsatisfiable.
pub fn all_different(problem: &mut Problem, variables: Vec<VariableIndex>) -> ConstraintIndex {
    let constraint = AllDifferent::new(variables, problem);
    problem.add_constraint(constraint)
}

/// Adds `x != y` ([`NotEquals`]) to `problem` and returns its index.
///
/// # Panics
///
/// As [`NotEquals::new`].
pub fn not_equals(problem: &mut Problem, x: VariableIndex, y: VariableIndex) -> ConstraintIndex {
    let constraint = NotEquals::new(x, y, problem);
    problem.add_constraint(constraint)
}

/// Adds [`Among`] to `problem` and returns its index: between `lb` and `ub` of `variables` take a
/// value in `values`.
///
/// # Panics
///
/// As [`Among::new`]: if a variable appears more than once in `variables`.
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

/// Adds [`Sum`] to `problem` and returns its index: the `variables` add up to `target`.
///
/// # Panics
///
/// As [`Sum::new`]: if a variable appears more than once in `variables`.
pub fn sum(problem: &mut Problem, variables: Vec<VariableIndex>, target: isize) -> ConstraintIndex {
    problem.add_constraint(Sum::new(variables, target, problem))
}

/// Adds [`AtLeast`] to `problem` and returns its index: at least `lb` of `variables` take a value
/// in `values`.
///
/// # Panics
///
/// As [`AtLeast::new`]: if a variable appears more than once in `variables`.
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

/// Adds [`Gcc`] to `problem` and returns its index. `bounds` lists `(value, lo, hi)`: the value
/// occurs between `lo` and `hi` times among `variables`; a value not listed is unconstrained.
///
/// # Panics
///
/// As [`Gcc::new`]: if a variable appears more than once in `variables`, or a value in `bounds`.
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
///
/// # Panics
///
/// As [`Regular::new`].
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

/// Adds [`Table`] to `problem` and returns its index: `variables` take the values of one of the
/// `tuples`, column `j` of a tuple being the value of `variables[j]`.
///
/// # Panics
///
/// As [`Table::new`]: if a variable appears more than once in `variables`, or a tuple does not
/// have one value per variable.
pub fn table(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    tuples: Vec<Vec<isize>>,
) -> ConstraintIndex {
    problem.add_constraint(Table::new(variables, tuples))
}

/// Adds [`Disjunctive`] to `problem` and returns its index: `variables` is a sequence of all the
/// jobs, processed one at a time, where job `j` has release date `release[j]`, deadline
/// `deadline[j]` and duration `duration[j]`. `precedences` lists pairs `(i, j)` of jobs, `i`
/// having to come before `j`. `setup_times` holds the matrix of the setup times (`matrix[i][j]` is
/// the time that has to pass between the end of job `i` and the start of job `j` when `j` follows
/// `i`) and the [`SetupRelaxation`](crate::modelling::SetupRelaxation) of the states. With
/// `padding`, the value `n` is a dummy job that fills the end of the sequence, and `variables`
/// may be shorter or longer than the lists (see [`Disjunctive::new`]).
///
/// # Panics
///
/// As [`Disjunctive::new`]: if a variable appears more than once in `variables`, if the lists do
/// not have one entry per variable, if a duration is negative, or if the precedences are not
/// valid (a job that does not exist, a job before itself, a cycle), or if the setup times are not
/// an `n` by `n` matrix without negative entry.
#[allow(clippy::too_many_arguments)]
pub fn disjunctive(
    problem: &mut Problem,
    variables: Vec<VariableIndex>,
    release: Vec<isize>,
    deadline: Vec<isize>,
    duration: Vec<isize>,
    precedences: Option<Vec<(usize, usize)>>,
    setup_times: Option<SetupTimes>,
    padding: bool,
) -> ConstraintIndex {
    let constraint = Disjunctive::new(
        variables,
        release,
        deadline,
        duration,
        precedences,
        setup_times,
        padding,
        problem,
    );
    problem.add_constraint(constraint)
}
