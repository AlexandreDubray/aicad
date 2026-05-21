use crate::mdd::*;
use crate::modelling::Problem;

pub enum MergeHeuristic {
    LessRelaxed,
    MostLikely,
}

impl MergeHeuristic {
    pub fn rank_nodes(&self, mdd: &Mdd, layer: usize, _problem: &Problem) -> Vec<f64> {
        let mut scores = vec![0.0; mdd.number_nodes_in_layer(layer)];
        match self {
            Self::LessRelaxed => {
            },
            Self::MostLikely => {
            },
        }
        scores
    }
}
