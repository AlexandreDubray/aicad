use pyo3::prelude::*;

use crate::mdd::heuristics::*;
use crate::modelling::Problem;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyOrderingHeuristic {
    MinDomMaxLinked(),
    Custom(Vec<usize>),
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) enum PyMergeHeuristic {
    LessRelaxed,
    MostLikely,
    StateSimilarity,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PySelectHeuristic {
    Greedy,
}

impl From<PyOrderingHeuristic> for OrderingHeuristic {
    fn from(value: PyOrderingHeuristic) -> Self {
        match value {
            PyOrderingHeuristic::MinDomMaxLinked() => OrderingHeuristic::MinDomMaxLinked,
            PyOrderingHeuristic::Custom(order) => OrderingHeuristic::Custom(order),
        }
    }
}

impl PyMergeHeuristic {
    /// The heuristic to compile `problem` with; `MostLikely` reads the probabilities of its
    /// variables.
    pub(crate) fn into_heuristic(self, problem: &Problem) -> MergeHeuristic {
        match self {
            PyMergeHeuristic::LessRelaxed => MergeHeuristic::LessRelaxed,
            PyMergeHeuristic::MostLikely => MergeHeuristic::most_likely(problem),
            PyMergeHeuristic::StateSimilarity => MergeHeuristic::StateSimilarity,
        }
    }
}

impl From<PySelectHeuristic> for SelectHeuristic {
    fn from(value: PySelectHeuristic) -> Self {
        match value {
            PySelectHeuristic::Greedy => SelectHeuristic::Greedy,
        }
    }
}
