use crate::mdd::*;
use crate::modelling::Problem;

pub enum MergeHeuristic {
    LessRelaxed,
    MostLikely,
}

impl MergeHeuristic {
    pub fn rank_nodes(&self, mdd: &Mdd, layer: LayerIndex, _problem: &Problem) -> Vec<f64> {
        let mut scores = vec![0.0; mdd[layer].number_nodes()];
        match self {
            Self::LessRelaxed => {
                for (i, _node) in mdd[layer].iter_nodes().enumerate() {
                    scores[i] += 1.0;
                }
            },
            Self::MostLikely => {
                for (i, _node) in mdd[layer].iter_nodes().enumerate() {
                    scores[i] += 1.0;
                }
            },
        }
        scores
    }
}
