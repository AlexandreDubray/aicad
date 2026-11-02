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
    /// The heuristic to compile `problem` with; `MostLikely` takes every value of a variable as
    /// likely as the others, since the Python side gives no probabilities.
    pub(crate) fn into_heuristic(self, problem: &Problem) -> MergeHeuristic {
        match self {
            PyMergeHeuristic::LessRelaxed => MergeHeuristic::LessRelaxed,
            PyMergeHeuristic::MostLikely => MergeHeuristic::MostLikely(
                problem
                    .iter_variables()
                    .map(|variable| {
                        let size = problem[variable].domain_size();
                        vec![1.0 / size as f64; size]
                    })
                    .collect(),
            ),
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
