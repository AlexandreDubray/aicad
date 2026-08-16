use super::*;
use crate::mdd::*;
use crate::modelling::VariableIndex;
use crate::utils::Bitset;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

// Structures for the allDifferent constraint.
//
// References:
//    - Hoda, S., Van Hoeve, W. J., & Hooker, J. N. (2010, September). A systematic approach to MDD-based constraint programming. CP2010

/// Structure that is used to compute the local properties of the MDD's nodes. The same property is
/// used for top-down and bottom-up computation. The property is divided into two sets (A, S)
/// where:
///     - $A(n)$ represents the values appearing on all path to/from node $n$
///     - $S(n)$ represents the value appearing on some path to/from node $n$
/// We use bitset to represent the sets.
///
/// The property operators are defined as follows.
///     1. The integration of a value $v$ into the property $(A, S)$ is defined by $$(A, S) \otimes
///        v = (A \cup \{v\}, S \cup \{v\}$$.
///        implemented using the | operator
///     2. The aggregation of two properties $(A, S)$ and $(A^\prime, S^\prime)$ is computed as $$(A, S) \oplus
///        (A^\prime, S^\prime) = (A \cap A^\prime, S \cup S^\prime)$$
#[derive(Clone, PartialEq, Eq, deepsize::DeepSizeOf)]
struct AllDifferentProperty {
    map: Arc<FxHashMap<isize, usize>>,
    value_all_path: Bitset,
    value_some_path: Bitset,
}

impl AllDifferentProperty {
    /// Creates a new property with bitsets of nb_words 64-bit unsigned integers. `all_path_reset`
    /// picks the starting value of `value_all_path`: `!0` (all-ones, the intersection identity)
    /// for the fold-accumulator seed, `0` (empty) for the empty-path/boundary value. See
    /// `Constraint::identity_property`/`Constraint::empty_property`.
    fn new(n: usize, map: Arc<FxHashMap<isize, usize>>, all_path_reset: u64) -> Self {
        let mut value_all_path = Bitset::new(n);
        value_all_path.reset(all_path_reset);
        let value_some_path = Bitset::new(n);
        Self {
            map,
            value_all_path,
            value_some_path,
        }
    }
}

#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AllDifferent {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Union of the domain of the variables in the scope
    domain: FxHashSet<isize>,
    /// Map each value of the joint domains to a bit in the properties' bitvectors
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    hall_set_bounds: Vec<(usize, usize)>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl AllDifferent {
    /// Creates a new AllDifferent constraint over variables
    pub fn new(variables: Vec<VariableIndex>, problem: &Problem) -> Self {
        let mut domain = FxHashSet::<isize>::default();
        for variable in variables.iter().copied() {
            domain.extend(problem[variable].iter_domain());
        }
        let val_to_bit: Arc<FxHashMap<isize, usize>> = Arc::new(
            domain
                .iter()
                .copied()
                .enumerate()
                .map(|(bit, val)| (val, bit))
                .collect(),
        );
        Self {
            variables,
            domain,
            val_to_bit,
            hall_set_bounds: vec![],
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for AllDifferent {
    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        let mut scope_layers = Vec::with_capacity(self.variables.len());
        self.layer_in_scope = (0..(order.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
        for (layer, &variable) in order.iter().enumerate() {
            if scope.contains(&variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                scope_layers.push(layer);
            }
        }

        self.hall_set_bounds = vec![(0, 0); order.len()];
        let n = scope_layers.len();
        for (pos, layer) in scope_layers.into_iter().enumerate() {
            self.hall_set_bounds[layer] = (pos, n - 1 - pos);
        }
    }

    /// Returns true if the layer is constrained by self
    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut set = FxHashSet::<isize>::default();
        for variable in self.variables.iter().copied() {
            let value = assignment[*variable];
            if set.contains(&value) {
                return false;
            }
            set.insert(value);
        }
        true
    }

    fn name(&self) -> &'static str {
        "AllDifferent"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn rank_nodes(&self, _nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        layer: usize,
        assignment: isize,
    ) -> bool {
        let parent = parent.as_any().downcast_ref::<AllDifferentProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of AllDifferentProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<AllDifferentProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of AllDifferentProperty",
                    child.name()
                );
        });
        let bit = *self.val_to_bit.get(&assignment).unwrap();

        // The value is already forced on every path up to the parent, or forced on every path
        // from the child down to the sink; can not use this value for the variable
        if parent.value_all_path.contains(bit) || child.value_all_path.contains(bit) {
            return true;
        }

        let (hall_set_size_up, hall_set_size_down) = self.hall_set_bounds[layer];
        let combined_capacity = hall_set_size_up + hall_set_size_down;
        let combined_size = parent.value_some_path.size_union(&child.value_some_path);
        combined_size == combined_capacity
            && (parent.value_some_path.contains(bit) || child.value_some_path.contains(bit))
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AllDifferentProperty::new(
            self.domain.len(),
            self.val_to_bit.clone(),
            !0,
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AllDifferentProperty::new(
            self.domain.len(),
            self.val_to_bit.clone(),
            0,
        ))
    }
}

impl ConstraintProperty for AllDifferentProperty {
    fn update(&mut self, parent: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = parent
            .as_any()
            .downcast_ref::<AllDifferentProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    parent.name()
                );
            });

        if in_scope {
            let bit = *self.map.get(&assignment).unwrap();
            self.value_some_path
                .union_with_and_bit(&other.value_some_path, bit);
            self.value_all_path
                .intersect_with_and_bit(&other.value_all_path, bit);
        } else {
            self.value_some_path.union(&other.value_some_path);
            self.value_all_path.intersect(&other.value_all_path);
        }
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for word in self.value_all_path.iter() {
            hasher.write_u64(word);
        }
        for word in self.value_some_path.iter() {
            hasher.write_u64(word);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<AllDifferentProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling eq on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        self.value_all_path == other.value_all_path && self.value_some_path == other.value_some_path
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "AllDifferentProperty"
    }
}

impl std::fmt::Display for AllDifferentProperty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let all_path = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_all_path.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        let some_path = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_some_path.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "all {} - some {}", all_path, some_path,)
    }
}

#[cfg(test)]
mod test_all_diff {

    use crate::constraints::{AllDifferent, Constraint};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use std::sync::Arc;

    #[test]
    pub fn test_basic_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0, 1], None);

        all_different(&mut problem, vec![x, y]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 1);
        assert!(is_solution(vec![0, 1], &solutions));
    }

    #[test]
    pub fn test_no_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);

        all_different(&mut problem, vec![x, y]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![0, 0], &solutions));
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
        assert!(is_solution(vec![1, 1], &solutions));
    }

    #[test]
    pub fn test_basic_hall_set_up() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        all_different(&mut problem, vec![x, y, z]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![0, 0, 2], &solutions));
        assert!(is_solution(vec![0, 1, 2], &solutions));
        assert!(is_solution(vec![1, 0, 2], &solutions));
        assert!(is_solution(vec![1, 1, 2], &solutions));
    }

    #[test]
    pub fn test_basic_hall_set_down() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        all_different(&mut problem, vec![x, y, z]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![2, 0, 0], &solutions));
        assert!(is_solution(vec![2, 0, 1], &solutions));
        assert!(is_solution(vec![2, 1, 0], &solutions));
        assert!(is_solution(vec![2, 1, 1], &solutions));
    }

    #[test]
    pub fn test_hall_set_around() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1], None);
        all_different(&mut problem, vec![x, y, z]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![0, 2, 0], &solutions));
        assert!(is_solution(vec![1, 2, 0], &solutions));
        assert!(is_solution(vec![0, 2, 1], &solutions));
        assert!(is_solution(vec![1, 2, 1], &solutions));
    }

    #[test]
    pub fn test_two_binary() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1], None);
        all_different(&mut problem, vars.clone());

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_value_all_path() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        all_different(&mut problem, vars.clone());
        equal(&mut problem, vars[1], 2);
        equal(&mut problem, vars[2], 0);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![1, 2, 0, 3], &solutions));
        assert!(is_solution(vec![3, 2, 0, 1], &solutions));
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_true() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        let all_diff = AllDifferent::new(vars, &problem);
        assert!(all_diff.is_satisfied(&[0, 1, 2]));
    }

    #[test]
    pub fn test_is_satisfied_false() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        let all_diff = AllDifferent::new(vars, &problem);
        assert!(!all_diff.is_satisfied(&[0, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope; variable 1 can duplicate freely.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        let all_diff = AllDifferent::new(vec![vars[0], vars[2]], &problem);
        assert!(all_diff.is_satisfied(&[0, 0, 1]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let problem = Problem::default();
        let all_diff = AllDifferent::new(vec![], &problem);
        assert!(all_diff.is_satisfied(&[]));
    }

    // --- Additional MDD-level edge cases --- //

    #[test]
    pub fn test_unsat_domain_too_small() {
        // Three variables, each with a 2-value domain, cannot all be pairwise different.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_single_variable_always_sat() {
        // A single variable is trivially all-different from itself.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        all_different(&mut problem, vec![x]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 3);
        assert!(is_solution(vec![0], &solutions));
        assert!(is_solution(vec![1], &solutions));
        assert!(is_solution(vec![2], &solutions));
    }

    #[test]
    pub fn test_combined_hall_condition_should_use_or_not_and() {
        // Scope v0..v3, decided in that order. We probe the decision for v2
        // (hall_set_size_up = 2, hall_set_size_down = 1).
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        let mut all_diff = AllDifferent::new(vars, &problem);
        all_diff.update_variable_ordering(&[
            VariableIndex(0),
            VariableIndex(1),
            VariableIndex(2),
            VariableIndex(3),
        ]);
        let root_td = all_diff.empty_property();
        let mut td1 = all_diff.identity_property();
        td1.update(root_td.as_ref(), 0, all_diff.is_layer_in_scope(0)); // v0 = 0

        let mut td2 = all_diff.identity_property();
        td2.update(td1.as_ref(), 1, all_diff.is_layer_in_scope(1)); // constituent A: v1 = 1
        td2.update(td1.as_ref(), 2, all_diff.is_layer_in_scope(1)); // constituent B: v1 = 2

        let sink_bu = all_diff.empty_property();
        let mut bu3 = all_diff.identity_property();
        bu3.update(sink_bu.as_ref(), 2, all_diff.is_layer_in_scope(3)); // v3 = 2

        assert!(all_diff.is_assignment_invalid(td2.as_ref(), bu3.as_ref(), 2, 1));
    }
}
