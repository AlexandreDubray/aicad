//! The description of a problem: variables with finite integer domains, and constraints over them.
//!
//! # Meaning
//!
//! A [`Problem`] owns its variables and its constraints. Variables and constraints are addressed
//! by [`VariableIndex`] and [`ConstraintIndex`], which count from 0 in the order they were added;
//! nothing is ever removed or renumbered. A [`ValueIndex`] is different: it is the position of a
//! value in the domain of one variable, not the value itself.
//!
//! An assignment is a slice with one value per variable, in variable order, holding the values
//! themselves and not their positions in the domain. A slice is a solution
//! ([`Problem::is_solution`]) when it has the right length, every value is in the domain of its
//! variable, and every constraint is satisfied: that is the specification the rest of the crate
//! is tested against.
//!
//! Each [`Variable`] also carries a probability for each value of its domain, uniform unless
//! given. The compiler and the samplers read them as the prior of the variable.
//!
//! The traits that a constraint implements, [`Constraint`] and [`ConstraintProperty`], are
//! defined here because a problem stores its constraints as trait objects. The constraints
//! themselves are in [`constraints`](crate::constraints).
//!
//! # Design notes
//!
//! - **Assignments hold values, not domain positions.** A constraint can then be checked against
//!   a solution as it is, whatever the domains; the positions only matter where probabilities
//!   are indexed.
//! - **Changing a domain resets its probabilities to uniform**, unless they are given with
//!   [`Variable::set_weighted_domain`]. The probabilities are indexed by position in the domain,
//!   so they cannot survive a different one.
//! - **Misuse panics at once, with a message.** An empty domain, a repeated value in a domain,
//!   probabilities that do not match the domain, a constraint over a variable that is not in the
//!   problem, or fixing a variable to a value outside its domain stop the program where the
//!   mistake is made, instead of giving wrong results later.
//! - **Variables carry an optional position.** It places a variable in a structure of the problem
//!   (for example the row and column of a Sudoku cell) and is read only by the positional
//!   encoding of the learned models; it plays no part in the compilation.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::all_different;
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let x = problem.add_variables(3, vec![1, 2, 3], None);
//! all_different(&mut problem, x.clone());
//!
//! assert!(problem.is_solution(&[1, 2, 3]));
//! assert!(!problem.is_solution(&[1, 1, 3]));
//!
//! equal(&mut problem, x[0], 2);
//! assert!(!problem.is_solution(&[1, 2, 3]));
//! assert!(problem.is_solution(&[2, 1, 3]));
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub(crate) mod constraint;
pub(crate) mod problem;
pub(crate) mod solution;
pub(crate) mod variable;

#[doc(no_inline)]
pub use constraint::{Constraint, ConstraintProperty, ConstraintShapeKey};
pub use problem::Problem;
pub use solution::{Solution, Status};
pub use variable::Variable;

/// Fixes `variable` to `value` by shrinking its domain to that value alone, with probability 1.
/// Panics if `value` is not in the domain of `variable`.
pub fn equal(problem: &mut Problem, variable: VariableIndex, value: isize) {
    let domain_size = problem[variable].domain_size();
    if !problem[variable].in_domain(value) {
        panic!(
            "cannot fix variable {} to {value}: it is not in its domain of {domain_size} values",
            variable.0
        );
    }
    problem[variable].set_domain(vec![value]);
}

/// Position of a variable in its problem, in the order the variables were added.
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

/// Position of a value in the domain of one variable. It is not the value itself.
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

/// Position of a constraint in its problem, in the order the constraints were added.
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
