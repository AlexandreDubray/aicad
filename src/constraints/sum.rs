use super::*;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::Hasher;

#[derive(deepsize::DeepSizeOf)]
pub struct Sum {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Target value the sum of the scope's variables must equal
    target: isize,
    top_down_properties: Vec<Vec<(isize, isize, bool)>>,
    bottom_up_properties: Vec<Vec<(isize, isize, bool)>>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Sum {

    /// Creates a new Sum constraint: the sum of the given variables must equal `target`.
    pub fn new(variables: Vec<VariableIndex>, target: isize) -> Self {
        Self {
            variables,
            target,
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            layer_in_scope: vec![],
        }
    }

}

impl Constraint for Sum {

    fn init(&mut self, vars: &[Variable]) {
        self.top_down_properties = (0..vars.len() + 1).map(|_| {
            vec![(0, 0, false)]
        }).collect::<Vec<Vec<(isize, isize, bool)>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| {
            vec![(0, 0, false)]
        }).collect::<Vec<Vec<(isize, isize, bool)>>>();
        self.layer_in_scope = (0..(vars.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
    }

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        // The layers in the scope of the variable are indicated using a bitvector of 64-bit words.
        // For each layer l its word index is given by l / 64 and the bit index by l % 64
        for variable in self.variables.iter() {
            let layer = ordering[variable.0];
            // Sets the bit of the layer to 1
            self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
        }
    }

    fn reset_property_top_down(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.top_down_properties[layer][index] = (0, 0, false);
    }

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        // The edge (source -> target) belongs to the layer of `source` (the parent): source
        // sits at layer L-1, target at layer L, and the variable branched on for this edge is
        // the one assigned to layer L-1.
        let delta = if self.is_layer_in_scope(source_layer) { assignment } else { 0 };
        let source_property = self.top_down_properties[source_layer][source_index];
        let candidate_min = source_property.0 + delta;
        let candidate_max = source_property.1 + delta;
        let target_property = &mut self.top_down_properties[target_layer][target_index];
        if target_property.2 {
            target_property.0 = target_property.0.min(candidate_min);
            target_property.1 = target_property.1.max(candidate_max);
        } else {
            target_property.0 = candidate_min;
            target_property.1 = candidate_max;
            target_property.2 = true;
        }
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index] = (0, 0, false);
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        // `target` is the node being computed (layer L) and `source` is its child (layer L+1,
        // already computed since layers are processed bottom-up). The edge belongs to layer
        // L, i.e. `target`'s layer, since that's where this edge's branching variable lives.
        let delta = if self.is_layer_in_scope(target_layer) { assignment } else { 0 };
        let source_property = self.bottom_up_properties[source_layer][source_index];
        let candidate_min = source_property.0 + delta;
        let candidate_max = source_property.1 + delta;
        let target_property = &mut self.bottom_up_properties[target_layer][target_index];
        if target_property.2 {
            target_property.0 = target_property.0.min(candidate_min);
            target_property.1 = target_property.1.max(candidate_max);
        } else {
            target_property.0 = candidate_min;
            target_property.1 = candidate_max;
            target_property.2 = true;
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(&self, source: NodeIndex, target: NodeIndex, _decision: VariableIndex, assignment: isize) -> bool {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;

        let local_min = self.top_down_properties[source_layer][source_index].0 + self.bottom_up_properties[target_layer][target_index].0 + assignment;
        let local_max = self.top_down_properties[source_layer][source_index].1 + self.bottom_up_properties[target_layer][target_index].1 + assignment;
        local_min > self.target || local_max < self.target
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        self.top_down_properties[layer].push((0, 0, false));
        self.bottom_up_properties[layer].push((0, 0, false));
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut total: isize = 0;
        for variable in self.variables.iter().copied() {
            total += assignment[variable.0];
        }
        total == self.target
    }

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        state.write_isize(self.top_down_properties[layer][index].0);
        state.write_isize(self.top_down_properties[layer][index].1);
        state.write_isize(self.bottom_up_properties[layer][index].0);
        state.write_isize(self.bottom_up_properties[layer][index].1);
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        // Compare only the substantive min/max bounds - `accumulated`/`ever_computed`
        // are propagation bookkeeping, not part of the node's logical state, and
        // comparing them here could keep otherwise-identical nodes from merging.
        self.top_down_properties[layer][index].0 == self.top_down_properties[olayer][oindex].0 &&
        self.top_down_properties[layer][index].1 == self.top_down_properties[olayer][oindex].1 &&
        self.bottom_up_properties[layer][index].0 == self.bottom_up_properties[olayer][oindex].0 &&
        self.bottom_up_properties[layer][index].1 == self.bottom_up_properties[olayer][oindex].1
    }

    fn name(&self) -> &'static str {
        "Sum"
    }

    fn shrink_layers(&mut self, layers_size: &[usize]) {
        for layer in 0..self.top_down_properties.len() {
            self.top_down_properties[layer].truncate(layers_size[layer]);
            self.bottom_up_properties[layer].truncate(layers_size[layer]);
        }
    }

    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64> {
        let mut scores = vec![0.0; nodes.len()];
        let mut sorted_nodes = (0..nodes.len()).map(|i| {
            let NodeIndex(layer, index) = nodes[i];
            let node_score = (self.top_down_properties[layer][index].0,
                self.top_down_properties[layer][index].1);
            (node_score, i)
        }).collect::<Vec<((isize, isize), usize)>>();
        sorted_nodes.sort_unstable();
        let n = nodes.len() as f64;
        for (rank, (_, i)) in sorted_nodes.iter().copied().enumerate() {
            scores[i] = (rank as f64) / n;
        }
        scores
    }
}

#[cfg(test)]
mod test_sum {

    use crate::modelling::*;
    use crate::constraints::{Sum, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_matches_target() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let sum = Sum::new(vars, 6);
        assert!(sum.is_satisfied(&[1, 2, 3]));
    }

    #[test]
    pub fn test_is_satisfied_does_not_match_target() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let sum = Sum::new(vars, 6);
        assert!(!sum.is_satisfied(&[1, 2, 2]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_zero_target() {
        let sum = Sum::new(vec![], 0);
        assert!(sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_nonzero_target() {
        let sum = Sum::new(vec![], 1);
        assert!(!sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_negative_values() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        let sum = Sum::new(vars, -3);
        assert!(sum.is_satisfied(&[-5, 2]));
        assert!(!sum.is_satisfied(&[5, -2]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_sum() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        sum(&mut problem, vec![x, y], 3);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![1, 2], &solutions));
        assert!(is_solution(vec![2, 1], &solutions));
    }

    #[test]
    pub fn test_sum_brute_force() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2, 3], None);
        let y = problem.add_variable(vec![0, 1, 2, 3], None);
        let z = problem.add_variable(vec![0, 1, 2, 3], None);
        sum(&mut problem, vec![x, y, z], 5);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    if a + b + c == 5 {
                        expected.push(vec![a, b, c]);
                    }
                }
            }
        }

        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }

    #[test]
    pub fn test_unsat_forced_mismatch() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
        // Both variables are forced to 0, so the sum is always 0, never 5.
        sum(&mut problem, vec![x, y], 5);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_unsat_target_above_reachable_range() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        // The maximum reachable sum is 2, so a target of 10 is unreachable.
        sum(&mut problem, vec![x, y], 10);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_unsat_target_below_reachable_range() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        // The minimum reachable sum is 0, so a target of -10 is unreachable.
        sum(&mut problem, vec![x, y], -10);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        sum(&mut problem, vec![x, y], 3);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![1, 2], &solutions));
        assert!(is_solution(vec![2, 1], &solutions));
    }

    #[test]
    pub fn test_negative_domain_values() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![-1, 0, 1], None);
        let y = problem.add_variable(vec![-1, 0, 1], None);
        sum(&mut problem, vec![x, y], 0);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 3);
        assert!(is_solution(vec![-1, 1], &solutions));
        assert!(is_solution(vec![0, 0], &solutions));
        assert!(is_solution(vec![1, -1], &solutions));
    }

    #[test]
    pub fn test_combined_with_all_different() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2, 3], None);
        let y = problem.add_variable(vec![0, 1, 2, 3], None);
        let z = problem.add_variable(vec![0, 1, 2, 3], None);
        all_different(&mut problem, vec![x, y, z]);
        sum(&mut problem, vec![x, y, z], 6);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..4 {
            for b in 0..4 {
                if a == b {
                    continue;
                }
                for c in 0..4 {
                    if c == a || c == b {
                        continue;
                    }
                    if a + b + c == 6 {
                        expected.push(vec![a, b, c]);
                    }
                }
            }
        }
        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }
}
