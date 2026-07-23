use crate::mdd::*;

pub enum MergeHeuristic {
    LessRelaxed,
    MostLikely,
}

impl MergeHeuristic {
    pub fn rank_nodes(&self, mdd: &Mdd, layer: usize) -> Vec<(f64, NodeIndex)> {
        let mut scores: Vec<(f64, NodeIndex)> = vec![];
        match self {
            Self::LessRelaxed => {
                for i in 0..mdd.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, i);
                    if mdd[node].is_active() {
                        let number_parents = mdd[node].number_parents() as f64;
                        let number_parents_relaxed = mdd[node].iter_parents().map(|edge| mdd[edge].from()).filter(|parent| !mdd[*parent].is_relaxed()).count() as f64;
                        scores.push((number_parents_relaxed / number_parents, node));
                    }
                }
            },
            Self::MostLikely => {
                for i in 0..mdd.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, i);
                    if mdd[node].is_active() {
                        let number_parents = mdd[node].number_parents() as f64;
                        let aggregate_probabilities = mdd[node].iter_parents().map(|edge| mdd.get_edge_probability(edge)).sum::<f64>();
                        scores.push((aggregate_probabilities / number_parents, node));
                    }
                }
            },
        }
        scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        scores
    }
}
