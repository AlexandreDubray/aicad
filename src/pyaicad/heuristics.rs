use pyo3::prelude::*;

use crate::mdd::heuristics::*;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyOrderingHeuristic {
    MinDomMaxLinked(),
    Custom(Vec<usize>),
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyMergeHeuristic {
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

impl From<PyMergeHeuristic> for MergeHeuristic {
    fn from(value: PyMergeHeuristic) -> Self {
        match value {
            PyMergeHeuristic::LessRelaxed => MergeHeuristic::LessRelaxed,
            PyMergeHeuristic::MostLikely => MergeHeuristic::MostLikely,
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
