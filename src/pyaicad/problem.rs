use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::constraints::*;
use crate::modelling::*;

#[pyclass]
pub struct PyProblem {
    inner: Arc<Problem>,
}

impl PyProblem {
    pub(super) fn arc(&self) -> Arc<Problem> {
        Arc::clone(&self.inner)
    }

    fn mutate(&mut self) -> PyResult<&mut Problem> {
        Arc::get_mut(&mut self.inner).ok_or_else(|| {
            PyRuntimeError::new_err(
                "cannot modify a Problem that has already been shared with a Compiler/Trainer",
            )
        })
    }
}

#[pymethods]
impl PyProblem {
    #[new]
    fn new() -> Self {
        PyProblem {
            inner: Arc::new(Problem::default()),
        }
    }

    fn add_int_var(&mut self, domain: Vec<isize>) -> PyResult<usize> {
        let var = self.mutate()?.add_variable(domain);
        Ok(var.0)
    }

    fn add_bool_var(&mut self) -> PyResult<usize> {
        let var = self.mutate()?.add_variable(vec![0, 1]);
        Ok(var.0)
    }

    fn add_all_different(&mut self, scope: Vec<usize>) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        all_different(self.mutate()?, vars);
        Ok(())
    }

    fn add_not_equals(&mut self, x: usize, y: usize) -> PyResult<()> {
        not_equals(self.mutate()?, VariableIndex(x), VariableIndex(y));
        Ok(())
    }

    fn add_equal(&mut self, x: usize, value: isize) -> PyResult<()> {
        let problem = self.mutate()?;
        if !problem[VariableIndex(x)].in_domain(value) {
            return Err(PyValueError::new_err(format!(
                "cannot fix variable {x} to {value}: it is not in its domain"
            )));
        }
        equal(problem, VariableIndex(x), value);
        Ok(())
    }

    fn add_among(
        &mut self,
        scope: Vec<usize>,
        values: Vec<isize>,
        lb: usize,
        ub: usize,
    ) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        among(self.mutate()?, vars, values, lb, ub);
        Ok(())
    }

    fn add_sum(&mut self, scope: Vec<usize>, target: isize) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        sum(self.mutate()?, vars, target);
        Ok(())
    }

    /// At least `lb` of `scope` must take a value in `values`.
    fn add_at_least(&mut self, scope: Vec<usize>, values: Vec<isize>, lb: usize) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        at_least(self.mutate()?, vars, values, lb);
        Ok(())
    }

    fn add_gcc(&mut self, scope: Vec<usize>, bounds: Vec<(isize, usize, usize)>) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        gcc(self.mutate()?, vars, bounds);
        Ok(())
    }

    fn add_regular(
        &mut self,
        scope: Vec<usize>,
        transitions: Vec<Vec<Option<usize>>>,
        initial_state: usize,
        accepting_states: Vec<usize>,
    ) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        regular(
            self.mutate()?,
            vars,
            transitions,
            initial_state,
            accepting_states,
        );
        Ok(())
    }

    fn add_table(&mut self, scope: Vec<usize>, tuples: Vec<Vec<isize>>) -> PyResult<()> {
        if let Some(tuple) = tuples.iter().find(|tuple| tuple.len() != scope.len()) {
            return Err(PyValueError::new_err(format!(
                "tuples: a tuple has {} values for a scope of {} variables",
                tuple.len(),
                scope.len()
            )));
        }
        let vars = scope.into_iter().map(VariableIndex).collect();
        table(self.mutate()?, vars, tuples);
        Ok(())
    }

    fn add_disjunctive(
        &mut self,
        scope: Vec<usize>,
        release: Vec<isize>,
        deadline: Vec<isize>,
        duration: Vec<isize>,
    ) -> PyResult<()> {
        for (name, list) in [
            ("release", &release),
            ("deadline", &deadline),
            ("duration", &duration),
        ] {
            if list.len() != scope.len() {
                return Err(PyValueError::new_err(format!(
                    "{name}: {} values for a scope of {} variables",
                    list.len(),
                    scope.len()
                )));
            }
        }
        if duration.iter().any(|&p| p < 0) {
            return Err(PyValueError::new_err("duration: a duration is negative"));
        }
        let vars = scope.into_iter().map(VariableIndex).collect();
        disjunctive(self.mutate()?, vars, release, deadline, duration);
        Ok(())
    }

    fn negate(&mut self, x: usize) -> PyResult<usize> {
        let y = self.add_bool_var()?;
        self.add_not_equals(x, y)?;
        Ok(y)
    }

    // --- Model modification --- //

    fn set_variable_position(&mut self, x: usize, position: Vec<usize>) -> PyResult<()> {
        self.mutate()?[VariableIndex(x)].set_position(position);
        Ok(())
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        self.inner.is_solution(&solution)
    }

    fn is_constraint_satisfied(&self, constraint: usize, solution: Vec<isize>) -> PyResult<bool> {
        if constraint >= self.inner.number_constraints() {
            return Err(PyIndexError::new_err(format!(
                "constraint {constraint} out of range: the problem has {} constraints",
                self.inner.number_constraints()
            )));
        }
        if solution.len() != self.inner.number_variables() {
            return Err(PyValueError::new_err(format!(
                "assignment has {} values for {} variables",
                solution.len(),
                self.inner.number_variables()
            )));
        }
        Ok(self.inner[ConstraintIndex(constraint)].is_satisfied(&solution))
    }

    // --- introspection: read-only, always available regardless of sharing --- //

    fn number_variables(&self) -> usize {
        self.inner.number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.inner.number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.inner[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.inner[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.inner[VariableIndex(variable)].iter_domain().collect()
    }
}
