//! A variable of a problem: a finite domain of integers and a probability for each value.
//!
//! # Meaning
//!
//! A [`Variable`] takes one of the values of its domain. Its probabilities, indexed by position in
//! the domain, are the prior that [`sample`](Variable::sample) draws from and that the compiler
//! and the samplers start from; they are uniform when not given. Probabilities are checked when
//! they are set: one per value, each finite and non-negative, summing to 1. A domain has at least
//! one value and no value twice. The variable also remembers which constraints have it in their
//! scope, in the order they were added to the problem.
//!
//! The variables of a problem are created and reached through the [`Problem`].

use super::*;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

/// How far from 1 the sum of the probabilities of a variable may be. It absorbs the rounding of
/// distributions computed in single precision, and nothing more.
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;

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

/// Panics unless `probabilities` has one entry per value of a domain of `domain_size` values,
/// each finite and non-negative, and they sum to 1 within `PROBABILITY_SUM_TOLERANCE`.
fn check_probabilities(domain_size: usize, probabilities: &[f64]) {
    if probabilities.len() != domain_size {
        panic!(
            "invalid probabilities: {} probabilities were given for a domain of {domain_size} values",
            probabilities.len()
        );
    }
    for (index, &p) in probabilities.iter().enumerate() {
        if !p.is_finite() || p < 0.0 {
            panic!(
                "invalid probabilities: the probability of the value at index {index} is {p}, but probabilities must be finite and non-negative"
            );
        }
    }
    let sum: f64 = probabilities.iter().sum();
    if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        panic!("invalid probabilities: they sum to {sum}, but they must sum to 1");
    }
}

/// The uniform distribution over a domain of `domain_size` values.
fn uniform(domain_size: usize) -> Vec<f64> {
    vec![1.0 / domain_size as f64; domain_size]
}

/// The sampler for `probabilities`, which must already have passed `check_probabilities`.
fn distribution(probabilities: &[f64]) -> WeightedIndex<f64> {
    WeightedIndex::new(probabilities.iter().copied())
        .expect("probabilities were checked to be non-negative and to sum to 1")
}

/// A variable with a finite domain of integers. See the [module documentation](crate::modelling).
pub struct Variable {
    /// The values the variable can take.
    domain: Vec<isize>,
    /// The probability of each value, by position in `domain`.
    probabilities: Vec<f64>,
    /// `probabilities` prepared for drawing a position.
    distribution: WeightedIndex<f64>,
    /// The constraints that have this variable in their scope.
    constraints: Vec<ConstraintIndex>,
    /// Where the variable sits in the structure of the problem, if it was given one.
    position: Option<Vec<usize>>,
}

impl Variable {
    /// Creates a variable with the given domain. `probs` gives the probability of each value, in
    /// the order of the domain, and is uniform when `None`.
    ///
    /// Panics if the domain is empty or repeats a value, or if `probs` does not have one
    /// probability per value, each finite and non-negative, summing to 1.
    pub fn new(domain: Vec<isize>, probs: Option<Vec<f64>>) -> Self {
        check_domain(&domain);
        let probabilities = match probs {
            Some(probabilities) => {
                check_probabilities(domain.len(), &probabilities);
                probabilities
            }
            None => uniform(domain.len()),
        };
        Self {
            domain,
            distribution: distribution(&probabilities),
            probabilities,
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

    /// Returns the probability that the variable takes the value from its domain at the given
    /// index. Panics if the index is not in the domain.
    pub fn probability(&self, index: ValueIndex) -> f64 {
        match self.probabilities.get(index.0) {
            Some(&p) => p,
            None => panic!(
                "value index {} is out of range for a domain of {} values",
                index.0,
                self.domain.len()
            ),
        }
    }

    /// Replaces the probabilities, given in the order of the domain. Panics unless there is one
    /// per value, each finite and non-negative, summing to 1.
    pub fn set_probabilities(&mut self, probabilities: &[f64]) {
        check_probabilities(self.domain.len(), probabilities);
        self.distribution = distribution(probabilities);
        self.probabilities = probabilities.to_owned();
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

    /// Sets the domain of the variable to the given values. The probabilities become uniform over
    /// the new domain, whatever they were; use [`set_weighted_domain`](Variable::set_weighted_domain)
    /// to give them. Panics if the domain is empty or repeats a value.
    pub fn set_domain(&mut self, domain: Vec<isize>) {
        check_domain(&domain);
        self.probabilities = uniform(domain.len());
        self.distribution = distribution(&self.probabilities);
        self.domain = domain;
    }

    /// Sets the domain of the variable and the probability of each of its values, in the same
    /// order. Panics if the domain is empty or repeats a value, or unless there is one probability per
    /// value, each finite and non-negative, summing to 1.
    pub fn set_weighted_domain(&mut self, domain: Vec<isize>, probabilities: Vec<f64>) {
        check_domain(&domain);
        check_probabilities(domain.len(), &probabilities);
        self.distribution = distribution(&probabilities);
        self.domain = domain;
        self.probabilities = probabilities;
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

    /// Draws a value of the domain with the probability of each one. Uses the process-wide
    /// generator, so the draw follows the seed given with `set_seed`. A value with probability 0
    /// is never drawn.
    pub fn sample(&self) -> isize {
        let index = crate::utils::with_rng(|rng| self.distribution.sample(rng));
        self.domain[index]
    }

    /// Places the variable in the structure of the problem, for the positional encoding of the
    /// learned models.
    pub(crate) fn set_position(&mut self, position: Vec<usize>) {
        self.position = Some(position);
    }

    /// The position given with `set_position`, if any.
    pub(crate) fn position(&self) -> &Option<Vec<usize>> {
        &self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_are_uniform_when_not_given() {
        let v = Variable::new(vec![4, 5, 6, 7], None);
        for i in 0..4 {
            assert_eq!(v.probability(ValueIndex(i)), 0.25);
        }
    }

    #[test]
    fn given_probabilities_are_kept() {
        let v = Variable::new(vec![4, 5], Some(vec![0.8, 0.2]));
        assert_eq!(v.probability(ValueIndex(0)), 0.8);
        assert_eq!(v.probability(ValueIndex(1)), 0.2);
    }

    #[test]
    fn value_is_read_by_position_in_the_domain() {
        let v = Variable::new(vec![10, -3, 7], None);
        assert_eq!(v.value(ValueIndex(1)), -3);
        assert_eq!(v.domain_size(), 3);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![10, -3, 7]);
    }

    #[test]
    fn in_domain_tells_values_apart_from_positions() {
        let v = Variable::new(vec![10, -3, 7], None);
        assert!(v.in_domain(10) && v.in_domain(-3) && v.in_domain(7));
        assert!(!v.in_domain(0) && !v.in_domain(1) && !v.in_domain(8));
    }

    #[test]
    #[should_panic(expected = "value index 3 is out of range for a domain of 2 values")]
    fn reading_a_value_out_of_range_panics() {
        Variable::new(vec![4, 5], None).value(ValueIndex(3));
    }

    #[test]
    #[should_panic(expected = "value index 2 is out of range for a domain of 2 values")]
    fn reading_a_probability_out_of_range_panics() {
        Variable::new(vec![4, 5], None).probability(ValueIndex(2));
    }

    #[test]
    #[should_panic(expected = "the value 4 appears more than once")]
    fn a_domain_with_a_repeated_value_is_refused() {
        Variable::new(vec![4, 5, 4], None);
    }

    #[test]
    #[should_panic(expected = "the value 1 appears more than once")]
    fn a_new_domain_with_a_repeated_value_is_refused() {
        Variable::new(vec![0], None).set_domain(vec![1, 1]);
    }

    #[test]
    #[should_panic(expected = "3 probabilities were given for a domain of 2 values")]
    fn probabilities_must_be_as_many_as_the_values() {
        Variable::new(vec![4, 5], Some(vec![0.5, 0.25, 0.25]));
    }

    #[test]
    #[should_panic(expected = "the probability of the value at index 1 is -0.5")]
    fn probabilities_must_not_be_negative() {
        Variable::new(vec![4, 5], Some(vec![1.5, -0.5]));
    }

    #[test]
    #[should_panic(expected = "the probability of the value at index 0 is NaN")]
    fn probabilities_must_be_finite() {
        Variable::new(vec![4, 5], Some(vec![f64::NAN, 1.0]));
    }

    #[test]
    #[should_panic(expected = "they sum to 0.9, but they must sum to 1")]
    fn probabilities_must_sum_to_one() {
        Variable::new(vec![4, 5], Some(vec![0.5, 0.4]));
    }

    #[test]
    fn probabilities_that_sum_to_one_up_to_rounding_are_accepted() {
        let v = Variable::new(vec![4, 5, 6], Some(vec![0.1, 0.2, 0.7 + 4e-7]));
        assert_eq!(v.domain_size(), 3);
    }

    #[test]
    #[should_panic(expected = "1 probabilities were given for a domain of 2 values")]
    fn replacing_the_probabilities_checks_them() {
        Variable::new(vec![4, 5], None).set_probabilities(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "a variable needs at least one value in its domain")]
    fn a_variable_cannot_be_created_without_a_domain() {
        Variable::new(vec![], None);
    }

    #[test]
    #[should_panic(expected = "a variable needs at least one value in its domain")]
    fn a_domain_cannot_be_emptied() {
        Variable::new(vec![0], None).set_domain(vec![]);
    }

    #[test]
    #[should_panic(expected = "a variable needs at least one value in its domain")]
    fn a_weighted_domain_cannot_be_empty() {
        Variable::new(vec![0], None).set_weighted_domain(vec![], vec![]);
    }

    #[test]
    fn a_value_without_probability_is_never_sampled() {
        let v = Variable::new(vec![4, 5, 6], Some(vec![0.5, 0.0, 0.5]));
        for _ in 0..500 {
            assert_ne!(v.sample(), 5);
        }
    }

    #[test]
    fn sampling_follows_new_probabilities_after_they_change() {
        let mut v = Variable::new(vec![4, 5], None);
        v.set_probabilities(&[0.0, 1.0]);
        assert!((0..100).all(|_| v.sample() == 5));
        v.set_weighted_domain(vec![7, 8], vec![1.0, 0.0]);
        assert!((0..100).all(|_| v.sample() == 7));
        v.set_domain(vec![3]);
        assert_eq!(v.sample(), 3);
    }

    #[test]
    fn a_weighted_domain_sets_the_values_and_their_probabilities() {
        let mut v = Variable::new(vec![0], None);
        v.set_weighted_domain(vec![7, 8, 9], vec![0.2, 0.3, 0.5]);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![7, 8, 9]);
        assert_eq!(v.probability(ValueIndex(2)), 0.5);
    }

    #[test]
    #[should_panic(expected = "they sum to 1.5, but they must sum to 1")]
    fn a_weighted_domain_checks_its_probabilities() {
        Variable::new(vec![0], None).set_weighted_domain(vec![7, 8], vec![0.5, 1.0]);
    }

    #[test]
    fn a_new_domain_resets_the_probabilities_to_uniform() {
        let mut v = Variable::new(vec![4, 5], Some(vec![0.9, 0.1]));
        v.set_domain(vec![1, 2, 3]);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![1, 2, 3]);
        for i in 0..3 {
            assert!((v.probability(ValueIndex(i)) - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn probabilities_can_be_replaced_without_touching_the_domain() {
        let mut v = Variable::new(vec![4, 5], None);
        v.set_probabilities(&[0.3, 0.7]);
        assert_eq!(v.probability(ValueIndex(1)), 0.7);
        assert_eq!(v.iter_domain().collect::<Vec<_>>(), vec![4, 5]);
    }

    #[test]
    fn constraints_are_listed_in_the_order_they_were_added() {
        let mut v = Variable::new(vec![0, 1], None);
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
    fn a_value_with_all_the_probability_is_always_sampled() {
        let v = Variable::new(vec![4, 5, 6], Some(vec![0.0, 1.0, 0.0]));
        for _ in 0..200 {
            assert_eq!(v.sample(), 5);
        }
    }

    #[test]
    fn sampling_follows_the_probabilities() {
        let v = Variable::new(vec![4, 5], Some(vec![0.9, 0.1]));
        let draws = 4000;
        let fours = (0..draws).filter(|_| v.sample() == 4).count();
        let share = fours as f64 / draws as f64;
        assert!((share - 0.9).abs() < 0.05, "share of 4 was {share}");
    }

    #[test]
    fn the_position_is_unset_until_given() {
        let mut v = Variable::new(vec![0, 1], None);
        assert!(v.position().is_none());
        v.set_position(vec![2, 5]);
        assert_eq!(v.position().as_deref(), Some(&[2, 5][..]));
    }
}
