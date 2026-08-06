use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::mdd::heuristics::*;
use crate::mdd::*;
use crate::modelling::*;

use super::heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
use super::problem::PyProblem;

#[pyclass]
pub struct Solver {
    problem: Option<Arc<Problem>>,
    mdd: Option<Mdd>,
    is_unsat: bool,
    is_solution_sat: bool,
}

impl Solver {
    fn problem_ref(&self) -> &Problem {
        if let Some(mdd) = &self.mdd {
            mdd.problem()
        } else {
            self.problem
                .as_ref()
                .expect("problem is only None once compile() has also set mdd")
        }
    }
}

#[pymethods]
impl Solver {
    #[new]
    fn new(problem: &PyProblem) -> Self {
        Solver {
            problem: Some(problem.arc()),
            mdd: None,
            is_unsat: false,
            is_solution_sat: false,
        }
    }

    #[pyo3(signature = (max_width=None,
            pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
            pymerge=PyMergeHeuristic::LessRelaxed,
            pyselect=PySelectHeuristic::Greedy))]
    fn compile(
        &mut self,
        max_width: Option<usize>,
        pyordering: PyOrderingHeuristic,
        pymerge: PyMergeHeuristic,
        pyselect: PySelectHeuristic,
    ) -> PyResult<()> {
        if self.mdd.is_some() {
            return Err(PyRuntimeError::new_err(
                "already compiled; recompiling with different heuristics isn't supported yet",
            ));
        }

        let width = max_width.unwrap_or(usize::MAX);
        let ordering: OrderingHeuristic = pyordering.into();
        let merge: MergeHeuristic = pymerge.into();
        let select: SelectHeuristic = pyselect.into();

        let arc = self
            .problem
            .take()
            .expect("checked above: not yet compiled");
        let problem = Arc::try_unwrap(arc).map_err(|arc| {
            self.problem = Some(arc);
            PyRuntimeError::new_err(
                "cannot compile: this Problem is also currently shared elsewhere (e.g. with a \
                 Trainer). This restriction goes away once Mdd is updated to borrow \
                 Arc<Problem> instead of owning it outright.",
            )
        })?;

        let mut mdd = Mdd::new(problem, width, ordering, merge, select);
        mdd.refine();
        self.is_unsat = mdd.is_unsat();
        self.mdd = Some(mdd);
        Ok(())
    }

    #[pyo3(signature = (max_width=None,
            pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
            pymerge=PyMergeHeuristic::LessRelaxed,
            pyselect=PySelectHeuristic::Greedy,
            sample=false))]
    fn solve(
        &mut self,
        max_width: Option<usize>,
        pyordering: PyOrderingHeuristic,
        pymerge: PyMergeHeuristic,
        pyselect: PySelectHeuristic,
        sample: bool,
    ) -> PyResult<Option<Vec<isize>>> {
        if self.mdd.is_none() {
            self.compile(max_width, pyordering, pymerge, pyselect)?;
        }
        if self.is_unsat() {
            return Ok(None);
        }
        let solution = if sample {
            Some(self.mdd.as_ref().unwrap().sample())
        } else {
            self.mdd.as_ref().unwrap().get_solution()
        };
        if let Some(sol) = solution.as_ref() {
            self.is_solution_sat = self.is_solution(sol.clone());
        }
        Ok(solution)
    }

    fn set_probabilities(&mut self, probabilities: Vec<Vec<f64>>) {
        if let Some(mdd) = &mut self.mdd {
            mdd.set_probabilities(&probabilities);
        }
    }

    // --- SOLUTION INFO --- //

    fn is_unsat(&self) -> bool {
        self.is_unsat
    }

    fn is_solution_sat(&self) -> bool {
        self.is_solution_sat
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        let problem = self.problem_ref();
        for constraint in problem.iter_constraints() {
            if !problem[constraint].is_satisfied(&solution) {
                return false;
            }
        }
        true
    }

    fn proportion_satisfied_constraints(&self, solution: Vec<isize>) -> f64 {
        let problem = self.problem_ref();
        let number_constraints = problem.number_constraints() as f64;
        let satisfied = problem
            .iter_constraints()
            .filter(|&constraint| problem[constraint].is_satisfied(&solution))
            .count() as f64;
        satisfied / number_constraints
    }

    fn sample_domains(&self) -> Vec<isize> {
        let problem = self.problem_ref();
        problem
            .iter_variables()
            .map(|variable| {
                let domain_size = problem[variable].domain_size();
                let value = ValueIndex(rand::random::<u64>() as usize % domain_size);
                problem[variable].value(value)
            })
            .collect()
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.problem_ref().number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.problem_ref().number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.problem_ref()[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.problem_ref()[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.problem_ref()[VariableIndex(variable)]
            .iter_domain()
            .collect()
    }

    // --- Visualisation and structure handling --- //

    fn topological_order(&self) -> Vec<(usize, usize, usize, isize)> {
        self.mdd.as_ref().unwrap().topological_order()
    }

    fn as_graphviz(&self) -> String {
        self.mdd.as_ref().unwrap().as_graphviz()
    }

    fn show_memory_info(&self) {
        if let Some(mdd) = self.mdd.as_ref() {
            mdd.show_memory_footprint();
        }
    }
}
