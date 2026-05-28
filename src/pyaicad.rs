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
    solution: Option<Vec<isize>>,
    is_unsat: bool,
}

#[pymethods]
impl Solver {

    #[new]
    fn new() -> Self {
        Solver {
            problem: Problem::default(),
            mdd: None,
            solution: None,
            is_unsat: false,
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

    #[pyo3(signature = (max_width=None, pyordering=PyOrderingHeuristic::MinDomMaxLinked(), pymerge=PyMergeHeuristic::LessRelaxed))]
    fn solve(&mut self, max_width: Option<usize>, pyordering: PyOrderingHeuristic, pymerge: PyMergeHeuristic) -> Option<Vec<isize>> {
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
        let solution = mdd.get_solution();
        self.is_unsat = mdd.is_unsat();
        self.mdd = Some(mdd);
        solution
    }

    fn is_unsat(&self) -> bool {
        self.is_unsat
    }
}

#[pymodule]
fn pyaicad(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    Ok(())
}
