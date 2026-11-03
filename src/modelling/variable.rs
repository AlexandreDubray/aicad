//! A variable of a problem: a finite domain of integers.
//!
//! # Meaning
//!
//! A [`Variable`] takes one of the values of its domain. A domain has at least one value and no
//! value twice. The variable also remembers which constraints have it in their scope, in the
//! order they were added to the problem.
//!
//! The variables of a problem are created and reached through the [`Problem`].
//!
//! # Design notes
//!
//! - **A variable has no probabilities.** Whatever needs a distribution over the values (the
//!   sampling of a diagram, the merge heuristic of the compiler, belief propagation) is given
//!   one, so a distribution is never read from the problem by accident.

use super::*;

/// Panics unless `domain` is non-empty and has no repeated value.
fn check_domain(domain: &[isize]) {
    if domain.is_empty() {
        panic!("invalid domain: a variable needs at least one value in its domain");
    }
    let mut seen = rustc_hash::FxHashSet::default();
    for &value in domain {
        if !seen.insert(value) {
            panic!(
                "invalid domain: the value {value} appears more than once in a domain of {} values",
                domain.len()
            );
        }
    }
}

/// A variable with a finite domain of integers. See the [module documentation](crate::modelling).
pub struct Variable {
    /// The values the variable can take.
    domain: Vec<isize>,
    /// The constraints that have this variable in their scope.
    constraints: Vec<ConstraintIndex>,
    /// Where the variable sits in the structure of the problem, if it was given one.
    position: Option<Vec<usize>>,
}

impl Variable {
    /// Creates a variable with the given domain.
    ///
    /// Panics if the domain is empty or repeats a value.
    pub fn new(domain: Vec<isize>) -> Self {
        check_domain(&domain);
        Self {
            domain,
            constraints: vec![],
            position: None,
        }
    }

    /// Returns the value of the domain at the given index. Panics if the index is not in the
    /// domain.
    pub fn value(&self, index: ValueIndex) -> isize {
        match self.domain.get(index.0) {
            Some(&value) => value,
            None => panic!(
                "value index {} is out of range for a domain of {} values",
                index.0,
                self.domain.len()
            ),
        }
    }

    /// Returns the number of elements in the domain
    pub fn domain_size(&self) -> usize {
        self.domain.len()
    }

    /// Iterates over the domain of the variable
    pub fn iter_domain(&self) -> impl Iterator<Item = isize> {
        self.domain.iter().copied()
    }

    /// True if `value` is in the domain.
    pub fn in_domain(&self, value: isize) -> bool {
        self.domain.iter().copied().any(|v| v == value)
    }

    /// Sets the domain of the variable to the given values. Panics if the domain is empty or
    /// repeats a value.
    pub fn set_domain(&mut self, domain: Vec<isize>) {
        check_domain(&domain);
        self.domain = domain;
    }

    /// Records that `constraint` has this variable in its scope. Called by the problem when a
    /// constraint is added.
    pub fn add_constraint(&mut self, constraint: ConstraintIndex) {
        self.constraints.push(constraint);
    }

    /// Iterates over the constraints that have this variable in their scope.
    pub fn iter_constraints(&self) -> impl Iterator<Item = ConstraintIndex> {
        self.constraints.iter().copied()
    }

    /// Returns the number of constraints that have this variable in their scope.
    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Places the variable in the structure of the problem, for the positional encoding of the
    /// learned models.
    pub fn set_position(&mut self, position: Vec<usize>) {
        self.position = Some(position);
    }

    /// The position given with `set_position`, if any.
    pub fn position(&self) -> &Option<Vec<usize>> {
        &self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_domain_replaces_the_old_one() {
        let mut v = Variable::new(vec![4, 5]);
        v.set_domain(vec![1, 2, 3]);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn value_is_read_by_position_in_the_domain() {
        let v = Variable::new(vec![10, -3, 7]);
        assert_eq!(v.value(ValueIndex(1)), -3);
        assert_eq!(v.domain_size(), 3);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![10, -3, 7]);
    }

    #[test]
    fn in_domain_tells_values_apart_from_positions() {
        let v = Variable::new(vec![10, -3, 7]);
        assert!(v.in_domain(10) && v.in_domain(-3) && v.in_domain(7));
        assert!(!v.in_domain(0) && !v.in_domain(1) && !v.in_domain(8));
    }

    #[test]
    #[should_panic(expected = "value index 3 is out of range for a domain of 2 values")]
    fn reading_a_value_out_of_range_panics() {
        Variable::new(vec![4, 5]).value(ValueIndex(3));
    }

    #[test]
    #[should_panic(expected = "the value 4 appears more than once")]
    fn a_domain_with_a_repeated_value_is_refused() {
        Variable::new(vec![4, 5, 4]);
    }

    #[test]
    #[should_panic(expected = "the value 1 appears more than once")]
    fn a_new_domain_with_a_repeated_value_is_refused() {
        Variable::new(vec![0]).set_domain(vec![1, 1]);
    }

    #[test]
    #[should_panic(expected = "a variable needs at least one value in its domain")]
    fn a_variable_cannot_be_created_without_a_domain() {
        Variable::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "a variable needs at least one value in its domain")]
    fn a_domain_cannot_be_emptied() {
        Variable::new(vec![0]).set_domain(vec![]);
    }

    #[test]
    fn constraints_are_listed_in_the_order_they_were_added() {
        let mut v = Variable::new(vec![0, 1]);
        assert_eq!(v.number_constraints(), 0);
        v.add_constraint(ConstraintIndex(3));
        v.add_constraint(ConstraintIndex(1));
        assert_eq!(v.number_constraints(), 2);
        assert_eq!(
            v.iter_constraints().collect::<Vec<_>>(),
            vec![ConstraintIndex(3), ConstraintIndex(1)]
        );
    }

    #[test]
    fn the_position_is_unset_until_given() {
        let mut v = Variable::new(vec![0, 1]);
        assert!(v.position().is_none());
        v.set_position(vec![2, 5]);
        assert_eq!(v.position().as_deref(), Some(&[2, 5][..]));
    }
}
