pub mod problem;
pub mod variable;

pub use problem::Problem;
use crate::constraints::*;
use rustc_hash::FxHashSet;

pub fn all_different(problem: &mut Problem, variables: Vec<VariableIndex>) {
    problem.add_constraint(AllDifferent::new(variables));
}

pub fn not_equals(problem: &mut Problem, x: VariableIndex, y: VariableIndex) {
    problem.add_constraint(NotEquals::new(x, y));
}

pub fn equal(problem: &mut Problem, variable: VariableIndex, value: isize) {
    problem[variable].set_domain(vec![value]);
}

pub fn among(problem: &mut Problem, variables: Vec<VariableIndex>, values: Vec<isize>, lb: usize, ub: usize) {
    problem.add_constraint(Among::new(variables, FxHashSet::from_iter(values.iter().cloned()), lb, ub));
}

pub fn sum(problem: &mut Problem, variables: Vec<VariableIndex>, target: isize) {
    problem.add_constraint(Sum::new(variables, target));
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct VariableIndex(pub usize);

impl std::ops::Deref for VariableIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct ValueIndex(pub usize);

impl std::ops::Deref for ValueIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct ConstraintIndex(pub usize);

impl std::ops::Deref for ConstraintIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

