use crate::mdd::*;

pub enum MergeHeuristic {
    LessRelaxed,
    MostLikely,
}

impl MergeHeuristic {
    pub fn rank_nodes(&self, mdd: &Mdd, layer: usize) -> Vec<(f64, usize)> {
        let n = mdd.number_nodes_in_layer(layer);
        let mut scores = vec![(0.0, 0); n];
        match self {
            Self::LessRelaxed => {
                for (i, score) in scores.iter_mut().enumerate() {
                    let node = NodeIndex(layer, i);
                    let number_parents = mdd[node].number_parents() as f64;
                    let number_parents_relaxed = mdd[node].iter_parents().map(|edge| mdd[edge].from()).filter(|parent| !mdd[*parent].is_relaxed()).count() as f64;
                    *score = (number_parents_relaxed / number_parents, i);
                }
            },
            Self::MostLikely => {
                for (i, score) in scores.iter_mut().enumerate() {
                    let node = NodeIndex(layer, i);
                    let number_parents = mdd[node].number_parents() as f64;
                    let aggregate_probabilities = mdd[node].iter_parents().map(|edge| mdd.get_edge_probability(edge)).sum::<f64>();
                    *score = (aggregate_probabilities / number_parents, i);
                }
            },
        }
        scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        scores
    }
}
