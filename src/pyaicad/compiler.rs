use std::sync::Arc;

use pyo3::prelude::*;

use crate::mdd::heuristics::*;
use crate::mdd::*;
use crate::modelling::*;

use super::heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
use super::problem::PyProblem;

#[pyclass]
pub struct Compiler {
    problem: Arc<Problem>,
}

#[pymethods]
impl Compiler {
    #[new]
    fn new(problem: &PyProblem) -> Self {
        Compiler {
            problem: problem.arc(),
        }
    }

    /// Compiles a fresh `Mdd`. `constraints` selects, by index, the subset of the problem's
    /// constraints to compile over; when omitted, all constraints are used. If `max_width` is
    /// given, the returned `Mdd` is immediately refined to that width; otherwise it is returned
    /// as the universe MDD with constraint propagation applied
    #[pyo3(signature = (constraints=None, max_width=None,
            pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
            pymerge=PyMergeHeuristic::LessRelaxed,
            pyselect=PySelectHeuristic::Greedy))]
    fn compile(
        &self,
        constraints: Option<Vec<usize>>,
        max_width: Option<usize>,
        pyordering: PyOrderingHeuristic,
        pymerge: PyMergeHeuristic,
        pyselect: PySelectHeuristic,
    ) -> PyMdd {
        let ordering: OrderingHeuristic = pyordering.into();
        let merge: MergeHeuristic = pymerge.into();
        let select: SelectHeuristic = pyselect.into();
        let constraints: Vec<ConstraintIndex> = match constraints {
            Some(cs) => cs.into_iter().map(ConstraintIndex).collect(),
            None => self.problem.iter_constraints().collect(),
        };

        let mut mdd = Mdd::new(
            Arc::clone(&self.problem),
            ordering,
            merge,
            select,
            &constraints,
        );
        if let Some(width) = max_width {
            mdd.refine(width);
        }
        PyMdd {
            inner: mdd,
            is_solution_sat: false,
        }
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.problem.number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.problem.number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.problem[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.problem[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.problem[VariableIndex(variable)]
            .iter_domain()
            .collect()
    }
}

/// A compiled `Mdd`, returned by `Compiler::compile`. It owns its `Mdd` outright, so it can be
/// refined further, sampled, or queried independently of the `Compiler` it came from and of any
/// other `Mdd` compiled from the same `Problem`.
#[pyclass]
pub struct PyMdd {
    inner: Mdd,
    is_solution_sat: bool,
}

#[pymethods]
impl PyMdd {
    /// Refines this Mdd in place, allowing up to `max_width` nodes per layer.
    fn refine(&mut self, max_width: usize) {
        self.inner.refine(max_width);
    }

    fn set_probabilities(&mut self, probabilities: Vec<Vec<f64>>) {
        self.inner.set_probabilities(&probabilities);
    }

    // --- SOLUTION INFO --- //

    fn is_unsat(&self) -> bool {
        self.inner.is_unsat()
    }

    fn is_solution_sat(&self) -> bool {
        self.is_solution_sat
    }

    #[pyo3(signature = (sample=false))]
    fn get_solution(&mut self, sample: bool) -> Option<Vec<isize>> {
        if self.inner.is_unsat() {
            return None;
        }
        let solution = if sample {
            Some(self.inner.sample())
        } else {
            self.inner.get_solution()
        };
        if let Some(sol) = solution.as_ref() {
            self.is_solution_sat = self.inner.problem().is_solution(sol);
        }
        solution
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        self.inner.problem().is_solution(&solution)
    }

    fn proportion_satisfied_constraints(&self, solution: Vec<isize>) -> f64 {
        let number_constraints = self.inner.number_constraints() as f64;
        let satisfied = self
            .inner
            .iter_constraints()
            .filter(|constraint| constraint.is_satisfied(&solution))
            .count() as f64;
        satisfied / number_constraints
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.inner.problem().number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.inner.problem().number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.inner.problem()[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.inner.problem()[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.inner.problem()[VariableIndex(variable)]
            .iter_domain()
            .collect()
    }

    // --- Visualisation and structure handling --- //

    fn topological_order(&self) -> Vec<(usize, usize, usize, isize)> {
        self.inner.topological_order()
    }

    fn as_graphviz(&self) -> String {
        self.inner.as_graphviz()
    }

    fn show_memory_info(&self) {
        self.inner.show_memory_footprint();
    }
}
