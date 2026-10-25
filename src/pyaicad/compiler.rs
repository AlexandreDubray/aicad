use std::sync::Arc;

use num_bigint::BigUint;
use pyo3::prelude::*;

use crate::mdd::heuristics::*;
use crate::mdd::*;
use crate::modelling::*;

use super::heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
use super::problem::PyProblem;

#[pyclass(name = "Compiler")]
pub struct PyCompiler {
    problem: Arc<Problem>,
}

#[pymethods]
impl PyCompiler {
    #[new]
    fn new(problem: &PyProblem) -> Self {
        PyCompiler {
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
        let merge = pymerge.into_heuristic(&self.problem);
        let select: SelectHeuristic = pyselect.into();
        let constraints: Vec<ConstraintIndex> = match constraints {
            Some(cs) => cs.into_iter().map(ConstraintIndex).collect(),
            None => self.problem.iter_constraints().collect(),
        };

        let mut compiler = Compiler::new(
            Arc::clone(&self.problem),
            CompilationHeuristic {
                ordering,
                merge,
                selection: select,
            },
            &constraints,
        );
        if let Some(width) = max_width {
            compiler.refine(width);
        }
        PyMdd {
            inner: compiler,
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

/// A compiled `Mdd`, returned by `Compiler::compile`. It owns its compilation outright, so it can be
/// refined further, sampled, or queried independently of the `Compiler` it came from and of any
/// other `Mdd` compiled from the same `Problem`.
#[pyclass]
pub struct PyMdd {
    inner: Compiler,
    is_solution_sat: bool,
}

#[pymethods]
impl PyMdd {
    /// Refines this Mdd in place, allowing up to `max_width` nodes per layer.
    fn refine(&mut self, max_width: usize) {
        self.inner.refine(max_width);
    }

    // --- SOLUTION INFO --- //

    fn is_unsat(&self) -> bool {
        self.inner.mdd().is_unsat()
    }

    fn is_solution_sat(&self) -> bool {
        self.is_solution_sat
    }

    #[pyo3(signature = (sample=false))]
    fn get_solution(&mut self, sample: bool) -> Option<Vec<isize>> {
        let mdd = self.inner.mdd();
        if mdd.is_unsat() {
            return None;
        }
        let solution = if sample {
            let weights = crate::sampling::problem_weights(mdd);
            let path = crate::sampling::sample_solution(mdd, &weights);
            Some(mdd.solution_from_path(&path))
        } else {
            mdd.get_solution()
        };
        if let Some(sol) = solution.as_ref() {
            self.is_solution_sat = mdd.problem().is_solution(sol);
        }
        solution
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        self.inner.mdd().problem().is_solution(&solution)
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

    /// Number of root-to-sink paths (encoded assignments) in this Mdd, as an arbitrary-
    /// precision Python int -- a relaxed diagram can encode far more than any fixed-width
    /// integer holds. 0 for an unsat Mdd.
    fn count_solutions(&self) -> BigUint {
        self.inner.mdd().count_solutions()
    }

    /// Number of layers in the compiled Mdd (one per variable in its scope, plus the sink).
    fn number_layers(&self) -> usize {
        self.inner.mdd().number_layers()
    }

    /// Number of (active) nodes at a given layer -- the Mdd's width at that layer. Useful for
    /// checking whether a compile actually reached exactness (every layer's width strictly
    /// below whatever cap was passed to `Compiler.compile`'s `max_width`) versus having been
    /// capped before converging.
    fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.inner.mdd().number_nodes_in_layer(layer)
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.inner.mdd().problem().number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.inner.mdd().problem().number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.inner.mdd().problem()[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.inner.mdd().problem()[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.inner.mdd().problem()[VariableIndex(variable)]
            .iter_domain()
            .collect()
    }

    // --- Visualisation and structure handling --- //

    fn as_graphviz(&self) -> String {
        self.inner.mdd().as_graphviz()
    }

    fn show_memory_info(&self) {
        self.inner.mdd().show_memory_footprint();
    }
}
