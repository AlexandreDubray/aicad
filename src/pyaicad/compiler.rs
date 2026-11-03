use std::sync::Arc;

use num_bigint::BigUint;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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
    /// as the universe MDD with constraint propagation applied. `max_iterations` limits the passes
    /// of the fix-point loop of that refinement; it must be at least 1 and defaults to no limit.
    #[pyo3(signature = (constraints=None, max_width=None,
            pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
            pymerge=PyMergeHeuristic::StateSimilarity,
            pyselect=PySelectHeuristic::Greedy,
            max_iterations=None))]
    fn compile(
        &self,
        constraints: Option<Vec<usize>>,
        max_width: Option<usize>,
        pyordering: PyOrderingHeuristic,
        pymerge: PyMergeHeuristic,
        pyselect: PySelectHeuristic,
        max_iterations: Option<usize>,
    ) -> PyResult<PyMdd> {
        if max_iterations == Some(0) {
            return Err(PyValueError::new_err(
                "max_iterations must be at least 1: with 0 passes the refinement would do nothing",
            ));
        }
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
            compiler.refine(width, max_iterations.unwrap_or(usize::MAX));
        }
        Ok(PyMdd {
            held: Held::Compiling(Box::new(compiler)),
        })
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

/// What a [`PyMdd`] holds: the compilation, which can refine the diagram but also keeps the
/// constraints, their properties and the heuristics alive, or the diagram alone.
enum Held {
    Compiling(Box<Compiler>),
    Frozen(Mdd),
    /// Only while `freeze` moves the compilation out; never observable.
    Moving,
}

/// A compiled `Mdd`, returned by `Compiler::compile`. It owns its compilation outright, so it can be
/// refined further, sampled, or queried independently of the `Compiler` it came from and of any
/// other `Mdd` compiled from the same `Problem`. Once it is not to be refined any more, `freeze`
/// drops everything but the diagram.
#[pyclass]
pub struct PyMdd {
    held: Held,
}

impl PyMdd {
    fn mdd(&self) -> &Mdd {
        match &self.held {
            Held::Compiling(compiler) => compiler.mdd(),
            Held::Frozen(mdd) => mdd,
            Held::Moving => unreachable!("freeze leaves the diagram in place"),
        }
    }
}

#[pymethods]
impl PyMdd {
    /// Refines this Mdd in place, allowing up to `max_width` nodes per layer and at most
    /// `max_iterations` passes of the fix-point loop (at least 1). Fails on a frozen Mdd.
    fn refine(&mut self, max_width: usize, max_iterations: usize) -> PyResult<()> {
        if max_iterations == 0 {
            return Err(PyValueError::new_err(
                "max_iterations must be at least 1: with 0 passes the refinement would do nothing",
            ));
        }
        match &mut self.held {
            Held::Compiling(compiler) => {
                compiler.refine(max_width, max_iterations);
                Ok(())
            }
            _ => Err(PyRuntimeError::new_err(
                "this diagram is frozen: it was released from its compiler and cannot be refined",
            )),
        }
    }

    /// Takes the diagram out of its compilation: the constraints, their properties and the
    /// heuristics are dropped and only the diagram stays, which uses far less memory. Everything
    /// but `refine` still works. Does nothing on a diagram that is already frozen.
    fn freeze(&mut self) {
        self.held = match std::mem::replace(&mut self.held, Held::Moving) {
            Held::Compiling(compiler) => Held::Frozen(compiler.into_mdd()),
            other => other,
        };
    }

    /// Whether `freeze` was called.
    fn is_frozen(&self) -> bool {
        matches!(self.held, Held::Frozen(_))
    }

    // --- SOLUTION INFO --- //

    fn is_unsat(&self) -> bool {
        self.mdd().is_unsat()
    }

    /// A path of the diagram (the first one of exact nodes, or one drawn with every value of a
    /// variable equally likely if `sample`) as the values taken and the indices of the variables they are
    /// the values of, in increasing order of variable: the order of the assignment of a model.
    #[pyo3(signature = (sample=false))]
    fn get_solution(&self, sample: bool) -> Option<(Vec<isize>, Vec<usize>)> {
        let mdd = self.mdd();
        if mdd.is_unsat() {
            return None;
        }
        let (values, variables) = if sample {
            let weights = crate::sampling::uniform_weights(mdd);
            let path = crate::sampling::sample_solution(mdd, &weights);
            mdd.solution_from_path(&path)
        } else {
            mdd.get_solution()?
        };
        let mut assignment: Vec<(usize, isize)> = variables
            .into_iter()
            .map(|variable| variable.0)
            .zip(values)
            .collect();
        assignment.sort_unstable();
        Some(assignment.into_iter().map(|(v, value)| (value, v)).unzip())
    }

    /// The indices of the variables the diagram branches on, in increasing order.
    fn scope(&self) -> Vec<usize> {
        let mdd = self.mdd();
        let mut scope: Vec<usize> = (0..mdd.number_decisions())
            .map(|layer| mdd.decision_at_layer(layer).0)
            .collect();
        scope.sort_unstable();
        scope
    }

    fn is_solution(&self, solution: Vec<isize>) -> bool {
        self.mdd().problem().is_solution(&solution)
    }

    /// Number of root-to-sink paths (encoded assignments) in this Mdd, as an arbitrary-
    /// precision Python int -- a relaxed diagram can encode far more than any fixed-width
    /// integer holds. 0 for an unsat Mdd.
    fn count_solutions(&self) -> BigUint {
        self.mdd().count_solutions()
    }

    /// Number of layers in the compiled Mdd (one per variable in its scope, plus the sink).
    fn number_layers(&self) -> usize {
        self.mdd().number_layers()
    }

    /// Number of (active) nodes at a given layer -- the Mdd's width at that layer. Useful for
    /// checking whether a compile actually reached exactness (every layer's width strictly
    /// below whatever cap was passed to `Compiler.compile`'s `max_width`) versus having been
    /// capped before converging.
    fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.mdd().number_nodes_in_layer(layer)
    }

    // --- MODEL INFO --- //

    fn number_variables(&self) -> usize {
        self.mdd().problem().number_variables()
    }

    fn number_constraints(&self) -> usize {
        self.mdd().problem().number_constraints()
    }

    fn constraint_scope(&self, constraint: usize) -> Vec<usize> {
        self.mdd().problem()[ConstraintIndex(constraint)]
            .iter_scope()
            .map(|v| v.0)
            .collect()
    }

    fn variable_domain_size(&self, variable: usize) -> usize {
        self.mdd().problem()[VariableIndex(variable)].domain_size()
    }

    fn variable_domain(&self, variable: usize) -> Vec<isize> {
        self.mdd().problem()[VariableIndex(variable)]
            .iter_domain()
            .collect()
    }

    // --- Visualisation and structure handling --- //

    fn as_graphviz(&self) -> String {
        self.mdd().as_graphviz()
    }

    fn show_memory_info(&self) {
        self.mdd().show_memory_footprint();
    }
}
