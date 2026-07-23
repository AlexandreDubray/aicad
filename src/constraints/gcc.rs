use super::*;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::Hasher;
use rustc_hash::{FxHashMap, FxHashSet};

/// Per-node counting property for the GCC. `guaranteed[v]` is the minimum number of times
/// value `v` (identified by its slot in `val_to_bit`) occurs among all source-n (top-down)
/// or n-sink (bottom-up) paths; `achievable[v]` is the maximum over the same paths.
#[derive(Clone, deepsize::DeepSizeOf)]
struct GccProperty {
    guaranteed: Vec<usize>,
    achievable: Vec<usize>,
    initialized: bool,
}

impl GccProperty {

    pub fn new(n: usize) -> Self {
        Self {
            guaranteed: vec![0; n],
            achievable: vec![0; n],
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.guaranteed.iter_mut().for_each(|x| *x = 0);
        self.achievable.iter_mut().for_each(|x| *x = 0);
        self.initialized = false;
    }

}

impl PartialEq for GccProperty {
    fn eq(&self, other: &Self) -> bool {
        self.guaranteed == other.guaranteed && self.achievable == other.achievable
    }
}
impl Eq for GccProperty {}

#[derive(deepsize::DeepSizeOf)]
pub struct Gcc {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Required [lo, hi] occurrence range for each explicitly-bounded value. Values of the
    /// joint domain that are not keys of this map are unconstrained.
    bounds: Vec<(isize, usize, usize)>,
    /// Union of the domain of the variables in the scope, plus the (explicitly bounded)
    /// keys of `bounds` so that a value which is bounded but never assignable is still
    /// tracked (and can correctly drive the constraint to UNSAT if its lower bound is > 0).
    domain: FxHashSet<isize>,
    /// Map each value of `domain` to a slot in the properties' vectors
    val_to_bit: FxHashMap<isize, usize>,
    /// For each value (by its slot), the required lower/upper occurrence bound
    lo: Vec<usize>,
    hi: Vec<usize>,
    /// Top-down properties for each node in the MDD
    top_down_properties: Vec<Vec<GccProperty>>,
    /// Bottom-up properties for each node in the MDD
    bottom_up_properties: Vec<Vec<GccProperty>>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Gcc {

    /// Creates a new GCC constraint over `variables`. `bounds` maps each cardinality
    /// constrained value to its required (lo, hi) occurrence range. Any value of the joint
    /// domain absent from `bounds` is implicitly unconstrained (range [0, |variables|]).
    pub fn new(variables: Vec<VariableIndex>, bounds: Vec<(isize, usize, usize)>) -> Self {
        Self {
            variables,
            bounds,
            domain: FxHashSet::<isize>::default(),
            val_to_bit: FxHashMap::<isize, usize>::default(),
            lo: vec![],
            hi: vec![],
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            layer_in_scope: vec![],
        }
    }

}

impl Constraint for Gcc {

    fn init(&mut self, vars: &[Variable]) {
        for variable in self.variables.iter().copied() {
            for value in vars[*variable].iter_domain() {
                self.domain.insert(value);
            }
        }
        // Values that are explicitly bounded must be tracked even if they never appear in
        // any of the scope's domains (an unreachable value with a strictly positive lower
        // bound must make the constraint UNSAT).
        for value in self.bounds.iter().copied().map(|v| v.0) {
            self.domain.insert(value);
        }
        for value in self.domain.iter().copied() {
            let bit = self.val_to_bit.len();
            self.val_to_bit.insert(value, bit);
        }
        let n = self.variables.len();
        self.lo = vec![0; self.domain.len()];
        self.hi = vec![n; self.domain.len()];
        for (value, lo, hi) in self.bounds.iter().copied() {
            let bit = *self.val_to_bit.get(&value).unwrap();
            self.lo[bit] = lo;
            self.hi[bit] = hi;
        }
        self.top_down_properties = (0..vars.len() + 1).map(|_| vec![GccProperty::new(self.domain.len())]).collect::<Vec<Vec<GccProperty>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| vec![GccProperty::new(self.domain.len())]).collect::<Vec<Vec<GccProperty>>>();
        self.layer_in_scope = (0..(vars.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
    }

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        for variable in self.variables.iter() {
            let layer = ordering[variable.0];
            self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
        }
    }

    fn reset_property_top_down(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.top_down_properties[layer][index].reset();
    }

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let assignment_bit = *self.val_to_bit.get(&assignment).unwrap();
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        let layer_in_scope = self.is_layer_in_scope(source_layer);

        let mut candidate = self.top_down_properties[source_layer][source_index].clone();
        if layer_in_scope {
            candidate.guaranteed[assignment_bit] += 1;
            candidate.achievable[assignment_bit] += 1;
        }

        let target_property = &mut self.top_down_properties[target_layer][target_index];
        if target_property.initialized {
            for v in 0..candidate.guaranteed.len() {
                target_property.guaranteed[v] = target_property.guaranteed[v].min(candidate.guaranteed[v]);
                target_property.achievable[v] = target_property.achievable[v].max(candidate.achievable[v]);
            }
        } else {
            *target_property = candidate;
            target_property.initialized = true;
        }
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index].reset();
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let assignment_bit = *self.val_to_bit.get(&assignment).unwrap();
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        let layer_in_scope = self.is_layer_in_scope(target_layer);

        let mut candidate = self.bottom_up_properties[source_layer][source_index].clone();
        if layer_in_scope {
            candidate.guaranteed[assignment_bit] += 1;
            candidate.achievable[assignment_bit] += 1;
        }

        let target_property = &mut self.bottom_up_properties[target_layer][target_index];
        if target_property.initialized {
            for v in 0..candidate.guaranteed.len() {
                target_property.guaranteed[v] = target_property.guaranteed[v].min(candidate.guaranteed[v]);
                target_property.achievable[v] = target_property.achievable[v].max(candidate.achievable[v]);
            }
        } else {
            *target_property = candidate;
            target_property.initialized = true;
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(&self, source: NodeIndex, target: NodeIndex, _decision: VariableIndex, assignment: isize) -> bool {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        let assignment_bit = *self.val_to_bit.get(&assignment).unwrap();

        let source_property = &self.top_down_properties[source_layer][source_index];
        let target_property = &self.bottom_up_properties[target_layer][target_index];

        for v in 0..self.domain.len() {
            let delta = if v == assignment_bit { 1 } else { 0 };
            let total_guaranteed = source_property.guaranteed[v] + target_property.guaranteed[v] + delta;
            // Every path through this edge is forced to use value v strictly more often
            // than allowed: the edge can never lead to a solution.
            if total_guaranteed > self.hi[v] {
                return true;
            }
            let total_achievable = source_property.achievable[v] + target_property.achievable[v] + delta;
            // Even in the best case (every remaining opportunity to use v is taken), v
            // cannot reach its required lower bound: the edge can never lead to a solution.
            if total_achievable < self.lo[v] {
                return true;
            }
        }
        false
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        self.top_down_properties[layer].push(GccProperty::new(self.domain.len()));
        self.bottom_up_properties[layer].push(GccProperty::new(self.domain.len()));
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut counts: FxHashMap<isize, usize> = FxHashMap::default();
        for variable in self.variables.iter().copied() {
            let value = assignment[*variable];
            *counts.entry(value).or_insert(0) += 1;
        }
        for (value, lo, hi) in self.bounds.iter().copied() {
            let count = counts.get(&value).copied().unwrap_or(0);
            if count < lo || count > hi {
                return false;
            }
        }
        true
    }

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        for count in self.top_down_properties[layer][index].guaranteed.iter().copied() {
            state.write_usize(count);
        }
        for count in self.top_down_properties[layer][index].achievable.iter().copied() {
            state.write_usize(count);
        }
        for count in self.bottom_up_properties[layer][index].guaranteed.iter().copied() {
            state.write_usize(count);
        }
        for count in self.bottom_up_properties[layer][index].achievable.iter().copied() {
            state.write_usize(count);
        }
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        self.top_down_properties[layer][index] == self.top_down_properties[olayer][oindex] &&
        self.bottom_up_properties[layer][index] == self.bottom_up_properties[olayer][oindex]
    }

    fn name(&self) -> &'static str {
        "GCC"
    }

    fn shrink_layers(&mut self, layers_size: &[usize]) {
        for layer in 0..self.top_down_properties.len() {
            self.top_down_properties[layer].truncate(layers_size[layer]);
            self.bottom_up_properties[layer].truncate(layers_size[layer]);
        }
    }
}

#[cfg(test)]
mod test_gcc {

    use crate::modelling::*;
    use crate::constraints::{Gcc, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;

    #[test]
    pub fn test_is_satisfied_within_bounds() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        // value 0 must appear exactly once, value 1 between 1 and 2 times.
        let gcc = Gcc::new(vars, vec![(0, 1, 1), (1, 1, 2)]);
        assert!(gcc.is_satisfied(&[0, 1, 1]));
    }

    #[test]
    pub fn test_is_satisfied_lower_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let gcc = Gcc::new(vars, vec![(2, 1, 3)]);
        // value 2 never appears, but requires at least 1 occurrence.
        assert!(!gcc.is_satisfied(&[0, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_upper_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let gcc = Gcc::new(vars, vec![(1, 0, 1)]);
        // value 1 appears twice, more than the allowed maximum of 1.
        assert!(!gcc.is_satisfied(&[1, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_unbounded_value_is_free() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        // Only value 0 is constrained; value 3 can occur any number of times.
        let gcc = Gcc::new(vars, vec![(0, 1, 1)]);
        assert!(gcc.is_satisfied(&[0, 3]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let gcc = Gcc::new(vec![], vec![]);
        assert!(gcc.is_satisfied(&[]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_all_different_via_gcc() {
        // Each of the 3 values must appear at most once among 3 variables: equivalent to
        // all_different.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        gcc(&mut problem, vars, vec![(0, 0, 1), (1, 0, 1), (2, 0, 1)]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 6);
        for sol in solutions.iter() {
            assert_ne!(sol[0], sol[1]);
            assert_ne!(sol[1], sol[2]);
            assert_ne!(sol[0], sol[2]);
        }
    }

    #[test]
    pub fn test_exact_multiplicity() {
        // 4 binary variables, value 1 must appear exactly twice.
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1], None);
        gcc(&mut problem, vars, vec![(1, 2, 2)]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2, 3]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        let count = [a, b, c, d].iter().filter(|v| **v == 1).count();
                        if count == 2 {
                            expected.push(vec![a, b, c, d]);
                        }
                    }
                }
            }
        }
        assert_eq!(expected.len(), 6);
        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }

    #[test]
    pub fn test_range_bounds_per_value() {
        // 3 ternary variables: value 0 between 1 and 2 times, value 2 at most 1 time.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        gcc(&mut problem, vars, vec![(0, 1, 2), (2, 0, 1)]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let count0 = [a, b, c].iter().filter(|v| **v == 0).count();
                    let count2 = [a, b, c].iter().filter(|v| **v == 2).count();
                    if (1..=2).contains(&count0) && count2 <= 1 {
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
    pub fn test_lower_bound_unsat_unreachable_value() {
        // Value 1 requires at least one occurrence, but no variable can ever be assigned it.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0], None);
        gcc(&mut problem, vars, vec![(1, 1, 2)]);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_upper_bound_unsat() {
        // Both variables forced to 1, but value 1 is capped at 1 occurrence.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![1], None);
        gcc(&mut problem, vars, vec![(1, 0, 1)]);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1], None);
        gcc(&mut problem, vars, vec![(1, 1, 1)]);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_no_bound_restriction() {
        // Unbounded values impose no restriction: every combination is a valid solution.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1], None);
        gcc(&mut problem, vars, vec![]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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
        gcc(&mut problem, vec![x, y, z], vec![(2, 1, 1)]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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
