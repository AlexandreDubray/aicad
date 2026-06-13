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

    fn set_probabilities(&mut self, probabilities: Vec<Vec<f64>>) {
        if let Some(mdd) = &mut self.mdd {
            mdd.set_probabilities(&probabilities);
        }
    }

    // --- SOLVE --- //
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
            self.is_solution_sat = self.is_solution(sol.clone());
        }
        solution
    }

    // --- SOLUTION INFO --- //

    fn is_unsat(&self) -> bool {
        self.is_unsat
    }

    fn is_solution_sat(&self) -> bool {
        self.is_solution_sat
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        for constraint in self.problem.iter_constraints() {
            if !self.problem[constraint].is_satisfied(&solution) {
                return false;
            }
        }
        true
    }

    fn proportion_satisfied_constraints(&self, solution: Vec<isize>) -> f64 {
        let number_constraints = self.problem.number_constraints() as f64;
        let satisfied = self.problem.iter_constraints().filter(|&constraint| self.problem[constraint].is_satisfied(&solution)).count() as f64;
        satisfied / number_constraints
    }

    fn topological_order(&self) -> Vec<(usize, usize, usize, isize)> {
        self.mdd.as_ref().unwrap().topological_order()
    }

    fn sample_domains(&self) -> Vec<isize> {
        self.problem.iter_variables().map(|variable| {
            let domain_size = self.problem[variable].domain_size();
            let value = ValueIndex(rand::random::<u64>() as usize % domain_size);
            self.problem[variable].value(value)
        }).collect::<Vec<isize>>()
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.problem.number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.problem.number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.problem[ConstraintIndex(constraint)].iter_scope().map(|v| v.0).collect::<Vec<usize>>()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.problem[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.problem[VariableIndex(variable)].iter_domain().collect()
    }
}

#[pymodule]
fn pyaicad(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    Ok(())
}
