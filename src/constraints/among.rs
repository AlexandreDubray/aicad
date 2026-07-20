use super::*;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::Hasher;
use rustc_hash::FxHashSet;

#[derive(deepsize::DeepSizeOf)]
pub struct Among {
    variables: Vec<VariableIndex>,
    values: FxHashSet<isize>,
    lb: usize,
    ub: usize,
    /// For each node, (guaranteed count, achievable count, has-been-initialised). The
    /// guaranteed count is the minimum number of value-in-set assignments over all
    /// source-n (top-down) or n-sink (bottom-up) paths; the achievable count is the
    /// maximum over the same paths.
    top_down_properties: Vec<Vec<(usize, usize, bool)>>,
    bottom_up_properties: Vec<Vec<(usize, usize, bool)>>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Among {

    pub fn new(variables: Vec<VariableIndex>, values: FxHashSet<isize>, lb: usize, ub: usize) -> Self {
        Self {
            variables,
            values,
            lb,
            ub,
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            layer_in_scope: vec![],
        }
    }

}

impl Constraint for Among {

    fn init(&mut self, vars: &[Variable]) {
        self.top_down_properties = (0..vars.len() + 1).map(|_| {
            vec![(0, 0, false)]
        }).collect::<Vec<Vec<(usize, usize, bool)>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| {
            vec![(0, 0, false)]
        }).collect::<Vec<Vec<(usize, usize, bool)>>>();
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

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize)  {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        // The edge (source -> target) belongs to the layer of `source` (the parent): layers
        // are decided top-down, source sits at layer L-1, target at layer L, and the variable
        // branched on for this edge is the one assigned to layer L-1.
        let delta = if self.is_layer_in_scope(source_layer) && self.values.contains(&assignment) { 1 } else { 0 };
        let (source_lb, source_ub, _) = self.top_down_properties[source_layer][source_index];
        let candidate_lb = source_lb + delta;
        let candidate_ub = source_ub + delta;
        let (lb, ub, initialized) = &mut self.top_down_properties[target_layer][target_index];
        if *initialized {
            // Several parents (or several parallel edges from the same parent) can reach the
            // same node: the guaranteed count is the min over all of them, the achievable
            // count is the max - mirroring the all_path/some_path (intersection/union)
            // aggregation used by AllDifferent.
            *lb = (*lb).min(candidate_lb);
            *ub = (*ub).max(candidate_ub);
        } else {
            *lb = candidate_lb;
            *ub = candidate_ub;
            *initialized = true;
        }
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index] = (0, 0, false);
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        // Here `target` is the node being computed (layer L) and `source` is its child (layer
        // L+1, already computed since layers are processed bottom-up). The edge belongs to
        // layer L, i.e. `target`'s layer, since that's where this edge's branching variable
        // lives.
        let delta = if self.is_layer_in_scope(target_layer) && self.values.contains(&assignment) { 1 } else { 0 };
        let (source_lb, source_ub, _) = self.bottom_up_properties[source_layer][source_index];
        let candidate_lb = source_lb + delta;
        let candidate_ub = source_ub + delta;
        let (lb, ub, initialized) = &mut self.bottom_up_properties[target_layer][target_index];
        if *initialized {
            *lb = (*lb).min(candidate_lb);
            *ub = (*ub).max(candidate_ub);
        } else {
            *lb = candidate_lb;
            *ub = candidate_ub;
            *initialized = true;
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(&self, source: NodeIndex, target: NodeIndex, _decision: VariableIndex, assignment: isize) -> bool {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;

        let mut local_lb = self.top_down_properties[source_layer][source_index].0 + self.bottom_up_properties[target_layer][target_index].0;
        if self.values.contains(&assignment) {
            local_lb += 1;
        }
        let mut local_ub = self.top_down_properties[source_layer][source_index].1 + self.bottom_up_properties[target_layer][target_index].1;
        if self.values.contains(&assignment) {
            local_ub += 1;
        }
        local_lb > self.ub || local_ub < self.lb
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        self.top_down_properties[layer].push((0, 0, false));
        self.bottom_up_properties[layer].push((0, 0, false));
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut count = 0;
        for variable in self.variables.iter().copied(){
            let value = assignment[variable.0];
            if self.values.contains(&value) {
                count += 1;
            }
        }
        self.lb <= count && count <= self.ub
    }

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        state.write_usize(self.top_down_properties[layer][index].0);
        state.write_usize(self.top_down_properties[layer][index].1);
        state.write_usize(self.bottom_up_properties[layer][index].0);
        state.write_usize(self.bottom_up_properties[layer][index].1);
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        self.top_down_properties[layer][index] == self.top_down_properties[olayer][oindex] &&
        self.bottom_up_properties[layer][index] == self.bottom_up_properties[olayer][oindex]
    }

    fn name(&self) -> &'static str {
        "Among"
    }

    fn shrink_layers(&mut self, layers_size: &[usize]) {
        for layer in 0..self.top_down_properties.len() {
            self.top_down_properties[layer].truncate(layers_size[layer]);
            self.bottom_up_properties[layer].truncate(layers_size[layer]);
        }
    }
}

#[cfg(test)]
mod test_among {

    use crate::modelling::*;
    use crate::constraints::{Among, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use rustc_hash::FxHashSet;

    fn values(vals: &[isize]) -> FxHashSet<isize> {
        FxHashSet::from_iter(vals.iter().copied())
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_within_bounds() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 1, 2);
        // value 1 (in set), 0 (not in set), 2 (in set) -> count = 2, within [1, 2]
        assert!(among.is_satisfied(&[1, 0, 2]));
    }

    #[test]
    pub fn test_is_satisfied_lower_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 1, 2);
        // no variable takes a value in {1, 2} -> count = 0 < lb = 1
        assert!(!among.is_satisfied(&[0, 0, 0]));
    }

    #[test]
    pub fn test_is_satisfied_upper_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 0, 1);
        // three variables take a value in {1, 2} -> count = 3 > ub = 1
        assert!(!among.is_satisfied(&[1, 2, 1]));
    }

    #[test]
    pub fn test_is_satisfied_boundary_lb_eq_ub() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        let among = Among::new(vars, values(&[5]), 1, 1);
        assert!(among.is_satisfied(&[5, 0]));
        assert!(!among.is_satisfied(&[5, 5]));
        assert!(!among.is_satisfied(&[0, 0]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let among = Among::new(vec![], values(&[1]), 0, 0);
        assert!(among.is_satisfied(&[]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_exactly_one() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 1, 1);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_range_bounds() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        among(&mut problem, vec![x, y, z], vec![0, 1], 1, 2);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        // Brute-force the expected solution set: 3^3 assignments, keep those where the
        // number of variables taking a value in {0, 1} is between 1 and 2 (inclusive).
        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let count = [a, b, c].iter().filter(|v| **v == 0 || **v == 1).count();
                    if (1..=2).contains(&count) {
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
    pub fn test_lower_bound_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
        // Neither variable can ever take value 1, so the count is always 0 < lb = 1.
        among(&mut problem, vec![x, y], vec![1], 1, 2);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_upper_bound_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![1], None);
        let y = problem.add_variable(vec![1], None);
        // Both variables are forced to 1, so the count is always 2 > ub = 0.
        among(&mut problem, vec![x, y], vec![1], 0, 0);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 1, 1);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_all_variables_must_take_value() {
        // lb = ub = number of variables: every variable must take a value in the set.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y, z], vec![1], 3, 3);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 1);
        assert!(is_solution(vec![1, 1, 1], &solutions));
    }

    #[test]
    pub fn test_no_bound_restriction() {
        // lb = 0 and ub = number of variables imposes no real restriction: every
        // combination is a valid solution.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 0, 2);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![0, 0], &solutions));
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
        assert!(is_solution(vec![1, 1], &solutions));
    }

    #[test]
    pub fn test_combined_with_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        among(&mut problem, vec![x, y, z], vec![2], 1, 1);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                if a == b {
                    continue;
                }
                for c in 0..3 {
                    let count = [a, b, c].iter().filter(|v| **v == 2).count();
                    if count == 1 {
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
