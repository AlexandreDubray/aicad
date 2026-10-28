//! The outcome of solving a [`Problem`]: a status, a full solution if one was found, and the cost.

use super::Problem;

/// Status of the solution returned. (Un)satisfiable only if proved, and unknown otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// A solution was found.
    Satisfiable,
    /// The problem was proved to have no solution.
    Unsatisfiable,
    /// Neither: the method reached a limit, or cannot decide.
    Unknown,
}

/// The outcome of solving a problem, whatever the method: compiling a diagram, searching, or a
/// mix of both. A field that a method has no use for keeps its neutral value (no compilation time
/// is 0, no iterations is `None`).
///
/// If the status is [`Status::Satisfiable`] the assignment is a *full* solution of the problem:
/// [`Problem::is_solution`] holds, so there is one value per variable of the problem. A method that
/// works on a part of the variables, such as a diagram over some of the constraints, completes the
/// assignment before building the `Solution`; the constructors check it.
#[derive(Clone, Debug)]
pub struct Solution {
    /// What was established.
    status: Status,
    /// The solution, in variable order, if and only if the status is satisfiable.
    assignment: Option<Vec<isize>>,
    /// Seconds from the start of the solve, compilation included.
    runtime: f64,
    /// Seconds of `runtime` spent compiling diagrams.
    compilation_runtime: f64,
    /// Number of steps of the search, `None` for a method without any.
    iterations: Option<usize>,
}

impl Solution {
    /// A satisfiable outcome with `assignment`, a value for each variable of `problem`.
    ///
    /// # Panics
    ///
    /// If `assignment` is not a solution of `problem` ([`Problem::is_solution`]).
    pub fn satisfiable(problem: &Problem, assignment: Vec<isize>) -> Self {
        if problem.number_variables() != assignment.len() {
            panic!(
                "A solution is being constructed with {} values for a problem with {} variables",
                assignment.len(),
                problem.number_variables()
            );
        }
        if !problem.is_solution(&assignment) {
            panic!("A solution is being constructed, but some constraints are not satisfied");
        }
        Self::new(Status::Satisfiable, Some(assignment))
    }

    /// An outcome that proves that the problem has no solution.
    pub fn unsatisfiable() -> Self {
        Self::new(Status::Unsatisfiable, None)
    }

    /// An outcome that settles nothing: a limit was reached, or the method cannot decide.
    pub fn unknown() -> Self {
        Self::new(Status::Unknown, None)
    }

    /// An outcome with no cost recorded yet.
    fn new(status: Status, assignment: Option<Vec<isize>>) -> Self {
        Self {
            status,
            assignment,
            runtime: 0.0,
            compilation_runtime: 0.0,
            iterations: None,
        }
    }

    /// Records the time of the solve in seconds, and the part of it spent compiling diagrams.
    pub fn with_runtime(mut self, runtime: f64, compilation_runtime: f64) -> Self {
        self.runtime = runtime;
        self.compilation_runtime = compilation_runtime;
        self
    }

    /// Records the number of steps of the search.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = Some(iterations);
        self
    }

    /// What was proved.
    pub fn status(&self) -> Status {
        self.status
    }

    /// The solution, one value per variable in variable order; `None` unless the status is
    /// [`Status::Satisfiable`].
    pub fn assignment(&self) -> Option<&[isize]> {
        self.assignment.as_deref()
    }

    /// Seconds from the start of the solve to its end, compilation included. Sub-second precision.
    pub fn runtime(&self) -> f64 {
        self.runtime
    }

    /// Seconds spent compiling diagrams, already included in [`runtime`](Self::runtime) and not
    /// on top of it. 0 for a method that compiles nothing.
    pub fn compilation_runtime(&self) -> f64 {
        self.compilation_runtime
    }

    /// Number of steps of the search, `None` for a method without any.
    pub fn iterations(&self) -> Option<usize> {
        self.iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::not_equals;

    /// Two variables with the domain `{0, 1}` that must differ.
    fn differ() -> Problem {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        problem
    }

    #[test]
    fn a_satisfiable_solution_holds_the_assignment_and_neutral_costs() {
        let solution = Solution::satisfiable(&differ(), vec![0, 1]);
        assert_eq!(solution.status(), Status::Satisfiable);
        assert_eq!(solution.assignment(), Some(&[0, 1][..]));
        assert_eq!(solution.runtime(), 0.0);
        assert_eq!(solution.compilation_runtime(), 0.0);
        assert_eq!(solution.iterations(), None);
    }

    #[test]
    fn costs_are_recorded() {
        let solution = Solution::unknown()
            .with_runtime(2.5, 1.0)
            .with_iterations(7);
        assert_eq!(solution.status(), Status::Unknown);
        assert_eq!(solution.assignment(), None);
        assert_eq!(solution.runtime(), 2.5);
        assert_eq!(solution.compilation_runtime(), 1.0);
        assert_eq!(solution.iterations(), Some(7));
    }

    #[test]
    fn an_unsatisfiable_solution_has_no_assignment() {
        let solution = Solution::unsatisfiable();
        assert_eq!(solution.status(), Status::Unsatisfiable);
        assert_eq!(solution.assignment(), None);
    }

    #[test]
    #[should_panic]
    fn a_partial_assignment_is_not_a_solution() {
        Solution::satisfiable(&differ(), vec![0]);
    }

    #[test]
    #[should_panic]
    fn an_assignment_that_violates_a_constraint_is_not_a_solution() {
        Solution::satisfiable(&differ(), vec![1, 1]);
    }
}
