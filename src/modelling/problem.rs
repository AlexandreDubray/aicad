//! The container of a problem: its variables and its constraints.
//!
//! # Meaning
//!
//! [`Problem`] is built by adding variables and then constraints over them, and is read through
//! the indices that the additions return (`problem[variable]`, `problem[constraint]`). A
//! constraint must be added after the variables of its scope; adding it records it on each of
//! them. See the [module documentation](crate::modelling) for what an assignment and a solution are.
//!
//! A problem is plain data: it is `Send` and can be shared behind an `Arc` once built.

use super::variable::Variable;
use super::*;

/// A set of variables and the constraints over them. See the [module documentation](crate::modelling).
#[derive(Default)]
pub struct Problem {
    /// Variables of the problem
    variables: Vec<Variable>,
    /// Constraints of the problem.
    constraints: Vec<Box<dyn Constraint + Send + Sync>>,
}

impl Problem {
    /// Adds a variable with the given domain to the problem and returns its index.
    pub fn add_variable(
        &mut self,
        domain: Vec<isize>,
        probabilities: Option<Vec<f64>>,
    ) -> VariableIndex {
        let ret = VariableIndex(self.variables.len());
        self.variables.push(Variable::new(domain, probabilities));
        ret
    }

    /// Adds n variables, with the same domain, to the problem and return their indexes.
    pub fn add_variables(
        &mut self,
        n: usize,
        domain: Vec<isize>,
        probabilities: Option<Vec<f64>>,
    ) -> Vec<VariableIndex> {
        (0..n)
            .map(|_| self.add_variable(domain.clone(), probabilities.clone()))
            .collect()
    }

    /// Adds a constraint to the problem and returns its index. The variables of its scope must
    /// already be in the problem; the constraint is recorded on each of them.
    pub fn add_constraint(&mut self, constraint: impl Constraint + 'static) -> ConstraintIndex {
        self.add_constraint_boxed(Box::new(constraint))
    }

    /// Adds an already-boxed constraint to the problem and returns its index. Used by
    /// `crate::mdd::arena` to rebuild a synthetic problem's constraints from a
    /// `ConstraintShapeKey` (where the concrete constraint type isn't known at the call site).
    pub(crate) fn add_constraint_boxed(
        &mut self,
        constraint: Box<dyn Constraint + Send + Sync>,
    ) -> ConstraintIndex {
        let ret = ConstraintIndex(self.constraints.len());
        for variable in constraint.iter_scope() {
            if variable.0 >= self.variables.len() {
                panic!(
                    "the {} constraint has variable {} in its scope, but the problem has only {} variables: add the variables before the constraints",
                    constraint.name(),
                    variable.0,
                    self.variables.len()
                );
            }
            self[variable].add_constraint(ret);
        }
        self.constraints.push(constraint);
        ret
    }

    /// Returns the number of variables in the problem
    pub fn number_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the number of constraints in the problem
    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Iterates over the constraints
    pub fn iter_constraints(&self) -> impl Iterator<Item = ConstraintIndex> {
        (0..self.constraints.len()).map(ConstraintIndex)
    }

    /// The constraints, in the order they were added.
    pub fn constraints(&self) -> &[Box<dyn Constraint + Send + Sync>] {
        &self.constraints
    }

    /// Iterates over the variables, in the order they were added.
    pub fn iter_variables(&self) -> impl Iterator<Item = VariableIndex> {
        (0..self.variables.len()).map(VariableIndex)
    }

    /// True if `assignment` is a solution: it has one value per variable, each value is in the
    /// domain of its variable, and every constraint is satisfied.
    pub fn is_solution(&self, assignment: &[isize]) -> bool {
        if assignment.len() != self.number_variables() {
            return false;
        }
        for (i, variable) in self.iter_variables().enumerate() {
            if !self[variable].in_domain(assignment[i]) {
                return false;
            }
        }
        for constraint in self.iter_constraints() {
            if !self[constraint].is_satisfied(assignment) {
                return false;
            }
        }
        true
    }
}

impl std::ops::Index<VariableIndex> for Problem {
    type Output = Variable;

    fn index(&self, index: VariableIndex) -> &Self::Output {
        match self.variables.get(index.0) {
            Some(variable) => variable,
            None => panic!(
                "variable {} does not exist: the problem has {} variables",
                index.0,
                self.variables.len()
            ),
        }
    }
}

impl std::ops::IndexMut<VariableIndex> for Problem {
    fn index_mut(&mut self, index: VariableIndex) -> &mut Self::Output {
        let number = self.variables.len();
        match self.variables.get_mut(index.0) {
            Some(variable) => variable,
            None => panic!(
                "variable {} does not exist: the problem has {number} variables",
                index.0
            ),
        }
    }
}

impl std::ops::Index<ConstraintIndex> for Problem {
    type Output = Box<dyn Constraint + Sync + Send>;

    fn index(&self, index: ConstraintIndex) -> &Self::Output {
        match self.constraints.get(index.0) {
            Some(constraint) => constraint,
            None => panic!(
                "constraint {} does not exist: the problem has {} constraints",
                index.0,
                self.constraints.len()
            ),
        }
    }
}

impl std::ops::IndexMut<ConstraintIndex> for Problem {
    fn index_mut(&mut self, index: ConstraintIndex) -> &mut Self::Output {
        let number = self.constraints.len();
        match self.constraints.get_mut(index.0) {
            Some(constraint) => constraint,
            None => panic!(
                "constraint {} does not exist: the problem has {number} constraints",
                index.0
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, not_equals};

    #[test]
    fn variables_and_constraints_are_numbered_in_the_order_they_are_added() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let c = not_equals(&mut problem, x, y);
        let d = not_equals(&mut problem, y, x);
        assert_eq!((x, y), (VariableIndex(0), VariableIndex(1)));
        assert_eq!((c, d), (ConstraintIndex(0), ConstraintIndex(1)));
        assert_eq!(problem.number_variables(), 2);
        assert_eq!(problem.number_constraints(), 2);
        assert_eq!(problem.iter_variables().collect::<Vec<_>>(), vec![x, y]);
        assert_eq!(problem.iter_constraints().collect::<Vec<_>>(), vec![c, d]);
    }

    #[test]
    fn add_variables_gives_consecutive_variables_with_the_same_domain() {
        let mut problem = Problem::default();
        problem.add_variable(vec![9], None);
        let vars = problem.add_variables(3, vec![1, 2], None);
        assert_eq!(
            vars,
            vec![VariableIndex(1), VariableIndex(2), VariableIndex(3)]
        );
        for v in vars {
            assert_eq!(problem[v].iter_domain().collect::<Vec<_>>(), vec![1, 2]);
        }
    }

    #[test]
    fn a_constraint_is_recorded_on_every_variable_of_its_scope() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        let c = all_different(&mut problem, vec![vars[0], vars[2], vars[3]]);
        let d = not_equals(&mut problem, vars[2], vars[1]);
        let on = |v: usize| problem[vars[v]].iter_constraints().collect::<Vec<_>>();
        assert_eq!(on(0), vec![c]);
        assert_eq!(on(1), vec![d]);
        assert_eq!(on(2), vec![c, d]);
        assert_eq!(on(3), vec![c]);
        assert_eq!(problem.constraints().len(), 2);
        assert_eq!(problem[c].name(), problem.constraints()[0].name());
    }

    #[test]
    fn is_solution_matches_a_brute_force_check_of_domains_and_constraints() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![1, 2], None);
        let z = problem.add_variable(vec![0, 2, 3], None);
        all_different(&mut problem, vec![x, y, z]);
        not_equals(&mut problem, x, z);
        let mut accepted = 0;
        for a in -1..5 {
            for b in -1..5 {
                for c in -1..5 {
                    let assignment = [a, b, c];
                    let in_domains =
                        [0, 1, 2].contains(&a) && [1, 2].contains(&b) && [0, 2, 3].contains(&c);
                    let expected = in_domains && a != b && b != c && a != c;
                    assert_eq!(problem.is_solution(&assignment), expected, "{assignment:?}");
                    accepted += expected as usize;
                }
            }
        }
        assert!(accepted > 0);
    }

    #[test]
    fn an_assignment_of_the_wrong_length_is_not_a_solution() {
        let mut problem = Problem::default();
        problem.add_variables(2, vec![0, 1], None);
        assert!(problem.is_solution(&[0, 1]));
        assert!(!problem.is_solution(&[0]));
        assert!(!problem.is_solution(&[0, 1, 0]));
    }

    #[test]
    fn a_problem_without_variables_has_the_empty_solution() {
        let problem = Problem::default();
        assert!(problem.is_solution(&[]));
    }

    #[test]
    #[should_panic(expected = "variable 5 does not exist: the problem has 1 variables")]
    fn reading_a_variable_that_is_not_in_the_problem_panics() {
        let mut problem = Problem::default();
        problem.add_variable(vec![0], None);
        let _ = &problem[VariableIndex(5)];
    }

    #[test]
    #[should_panic(expected = "constraint 0 does not exist: the problem has 0 constraints")]
    fn reading_a_constraint_that_is_not_in_the_problem_panics() {
        let problem = Problem::default();
        let _ = &problem[ConstraintIndex(0)];
    }

    #[test]
    #[should_panic(expected = "has variable 3 in its scope, but the problem has only 2 variables")]
    fn a_constraint_over_a_variable_that_is_not_in_the_problem_panics() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        problem.add_variable(vec![0, 1], None);
        let mut other = Problem::default();
        other.add_variables(4, vec![0, 1], None);
        let constraint = crate::constraints::NotEquals::new(x, VariableIndex(3), &other);
        problem.add_constraint(constraint);
    }

    #[test]
    #[should_panic(expected = "cannot fix variable 1 to 5: it is not in its domain of 2 values")]
    fn equal_refuses_a_value_outside_the_domain() {
        let mut problem = Problem::default();
        problem.add_variables(2, vec![0, 1], None);
        equal(&mut problem, VariableIndex(1), 5);
    }

    #[test]
    fn equal_leaves_a_single_value() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], Some(vec![0.5, 0.25, 0.25]));
        equal(&mut problem, x, 2);
        assert_eq!(problem[x].iter_domain().collect::<Vec<_>>(), vec![2]);
        assert_eq!(problem[x].probability(ValueIndex(0)), 1.0);
        assert!(problem.is_solution(&[2]));
        assert!(!problem.is_solution(&[1]));
    }
}
