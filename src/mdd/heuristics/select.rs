use crate::mdd::*;

pub enum SelectHeuristic {
    Greedy,
}

impl SelectHeuristic {
    pub fn select_node(&self, mdd: &Mdd, layer: usize) -> Option<NodeIndex> {
        match self {
            Self::Greedy => {
                for index in 0..mdd.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, index);
                    if mdd[node].is_relaxed() {
                        return Some(node);
                    }
                }
                None
            },
        }
    }
}
