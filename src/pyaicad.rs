use pyo3::prelude::*;

use crate::mdd::*;
use crate::mdd::heuristics::*;
use crate::modelling::*;

#[pyclass]
#[derive(Clone)]
pub enum PyOrderingHeuristic {
    MinDomMaxLinked(),
    Custom(Vec<usize>),
}

#[pyclass]
#[derive(Clone)]
pub enum PyMergeHeuristic {
    LessRelaxed,
    MostLikely,
}

#[pyclass]
pub struct Solver {
    problem: Problem,
    mdd: Option<Mdd>,
    is_unsat: bool,
    is_solution_sat: bool,
}

#[pymethods]
impl Solver {

    #[new]
    fn new() -> Self {
        Solver {
            problem: Problem::default(),
            mdd: None,
            is_unsat: false,
            is_solution_sat: false,
        }
    }

    fn add_int_var(&mut self, domain: Vec<isize>) -> usize {
        let var = self.problem.add_variable(domain, None);
        var.0
    }

    fn add_bool_var(&mut self) -> usize {
        let var = self.problem.add_variable(vec![0, 1], None);
        var.0
    }

    fn add_all_different(&mut self, scope: Vec<usize>) {
        let vars = scope.into_iter().map(VariableIndex).collect();
        all_different(&mut self.problem, vars);
    }

    fn add_not_equals(&mut self, x: usize, y: usize) {
        not_equals(&mut self.problem, VariableIndex(x), VariableIndex(y));
    }

    fn add_equal(&mut self, x: usize, value: isize) {
        equal(&mut self.problem, VariableIndex(x), value);
    }

    fn negate(&mut self, x: usize) -> usize {
        let y = self.add_bool_var();
        self.add_not_equals(x, y);
        y
    }

    fn compile(&mut self, max_width: Option<usize>, pyordering: PyOrderingHeuristic, pymerge: PyMergeHeuristic) {
        let width = max_width.unwrap_or(usize::MAX);
        let ordering = match pyordering {
            PyOrderingHeuristic::MinDomMaxLinked() => OrderingHeuristic::MinDomMaxLinked,
            PyOrderingHeuristic::Custom(order) => OrderingHeuristic::Custom(order),
        };

        let merge = match pymerge {
            PyMergeHeuristic::LessRelaxed => MergeHeuristic::LessRelaxed,
            PyMergeHeuristic::MostLikely => MergeHeuristic::MostLikely,
        };

        let mut mdd = Mdd::new(std::mem::take(&mut self.problem), width, ordering, merge);
        mdd.refine();
        self.is_unsat = mdd.is_unsat();
        self.mdd = Some(mdd);
    }

    #[pyo3(signature = (max_width=None,
            pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
            pymerge=PyMergeHeuristic::LessRelaxed,
            recompile=false,
            sample=false))]
    fn solve(&mut self, max_width: Option<usize>, pyordering: PyOrderingHeuristic, pymerge: PyMergeHeuristic, recompile: bool, sample: bool) -> Option<Vec<isize>> {
        if self.mdd.is_none() || recompile {
            self.compile(max_width, pyordering, pymerge);
        }
        if self.is_unsat() {
            return None;
        }
        let solution = if sample {
            Some(self.mdd.as_ref().unwrap().sample())
        }  else {
            self.mdd.as_ref().unwrap().get_solution()
        };
        if let Some(sol) = solution.as_ref() {
            self.is_solution_sat = self.mdd.as_ref().unwrap().is_solution(sol);
        }
        solution
    }

    fn is_unsat(&self) -> bool {
        self.is_unsat
    }

    fn is_solution_sat(&self) -> bool {
        self.is_solution_sat
    }

    fn set_probabilities(&mut self, probabilities: Vec<Vec<f64>>) {
        if let Some(mdd) = &mut self.mdd {
            mdd.set_probabilities(&probabilities);
        }
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        self.mdd.as_ref().unwrap().is_solution(&solution)
    }

    fn proportion_satisfied_constraints(&self, solution: Vec<isize>) -> f64 {
        self.mdd.as_ref().unwrap().proportion_satisfied_constraints(&solution)
    }

    fn topological_order(&self) -> Vec<(usize, usize, usize, isize)> {
        self.mdd.as_ref().unwrap().topological_order()
    }
}

#[pymodule]
fn pyaicad(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    Ok(())
}
