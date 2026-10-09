pub mod constraint;
pub mod problem;
pub mod variable;

#[doc(no_inline)]
pub use constraint::{Constraint, ConstraintProperty, ConstraintShapeKey};
pub use problem::Problem;

pub fn equal(problem: &mut Problem, variable: VariableIndex, value: isize) {
    problem[variable].set_domain(vec![value]);
}

#[derive(
    Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, deepsize::DeepSizeOf,
)]
pub struct VariableIndex(pub usize);

impl std::ops::Deref for VariableIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(
    Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, deepsize::DeepSizeOf,
)]
pub struct ValueIndex(pub usize);

impl std::ops::Deref for ValueIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(
    Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default, deepsize::DeepSizeOf,
)]
pub struct ConstraintIndex(pub usize);

impl std::ops::Deref for ConstraintIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
