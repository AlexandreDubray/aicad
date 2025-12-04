pub mod problem;
pub mod variable;

pub use problem::Problem;
use crate::constraints::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct VariableIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct ValueIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct ConstraintIndex(pub usize);

pub fn all_different(problem: &mut Problem, variables: Vec<VariableIndex>) {
    let constraint_index = ConstraintIndex(problem.number_constraints());
    for variable in variables.iter().copied() {
        problem[variable].add_constraint(constraint_index);
    }
    problem.add_constraint(AllDifferent::new(problem, variables));
}

pub fn equal(problem: &mut Problem, variable: VariableIndex, value: isize) {
    problem[variable].set_domain(vec![value]);
}
