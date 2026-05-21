use crate::modelling::{Problem, VariableIndex};

pub enum OrderingHeuristic {
    MinDomMaxLinked,
    Custom(Vec<usize>),
}

impl OrderingHeuristic {

    pub fn get_order(&self, problem: &Problem) -> Vec<VariableIndex> {
        match self {
            Self::Custom(order) => return order.iter().copied().map(VariableIndex).collect::<Vec<VariableIndex>>(),
            _ => (),
        };
        let n = problem.number_variables();
        let mut scores = vec![0; n];
        let mut candidates = (0..n).map(VariableIndex).collect::<Vec<VariableIndex>>();
        let mut order: Vec<VariableIndex> = vec![];
        for i in (0..n).rev() {
            let candidate = candidates[i];
            if problem[candidate].domain_size() == 1 {
                order.push(candidate);
                for constraint in problem[candidate].iter_constraints() {
                    for linked_variable in problem[constraint].iter_scope() {
                        scores[linked_variable.0] += 1;
                    }
                }
                candidates.swap_remove(i);
            }
        }
        for _ in 0..candidates.len() {
            let mut best_index = 0;
            let mut best_score = 0;
            let mut best_tie = 0;
            for (index, candidate) in candidates.iter().copied().enumerate() {
                let score = scores[candidate.0];
                let tie = problem[candidate].number_constraints();
                if score > best_score || (score == best_score && tie > best_tie) {
                    best_index = index;
                    best_score = score;
                    best_tie = tie;
                }
            }
            let selected = candidates[best_index];
            order.push(selected);
            for constraint in problem[selected].iter_constraints() {
                for linked_variable in problem[constraint].iter_scope() {
                    scores[linked_variable.0] += 1;
                }
            }
            candidates.swap_remove(best_index);
        }
        order
    }

}
