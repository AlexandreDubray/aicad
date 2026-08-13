use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
                "cannot modify a Problem that has already been shared with a Solver/Trainer",
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
        let var = self.mutate()?.add_variable(domain, None);
        Ok(var.0)
    }

    fn add_bool_var(&mut self) -> PyResult<usize> {
        let var = self.mutate()?.add_variable(vec![0, 1], None);
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
        equal(self.mutate()?, VariableIndex(x), value);
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

    fn add_gcc(&mut self, scope: Vec<usize>, bounds: Vec<(isize, usize, usize)>) -> PyResult<()> {
        let vars = scope.into_iter().map(VariableIndex).collect();
        gcc(self.mutate()?, vars, bounds);
        Ok(())
    }

    fn negate(&mut self, x: usize) -> PyResult<usize> {
        let y = self.add_bool_var()?;
        self.add_not_equals(x, y)?;
        Ok(y)
    }
    
    // --- Model modification --- //
    
    fn set_variable_position(&mut self, x: usize, position: Vec<usize>) {
        self.mutate().unwrap()[VariableIndex(x)].set_position(position);
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
