use super::*;
use crate::modelling::VariableIndex;
use crate::mdd::*;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::utils::Bitset;
use std::hash::Hasher;

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
    /// Values that appear on all source-n (top-down property) or n-sink (bottom-up
    /// property) path.
    value_all_path: Bitset,
    /// Values that appear on some source-n (top-down property) or n-sink (bottom-up
    /// property) path.
    value_some_path: Bitset,
}

impl AllDifferentProperty {

    /// Creates a new property with bitsiets of nb_words 64-bit unsigned integers
    pub fn new(n: usize) -> Self {
        let value_all_path = Bitset::new(n);
        let value_some_path = Bitset::new(n);
        Self {
            value_all_path,
            value_some_path,
        }
    }

}

#[derive(deepsize::DeepSizeOf)]
pub struct AllDifferent {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Union of the domain of the variables in the scope
    domain: FxHashSet<isize>,
    /// Map each value of the joint domains to a bit in the properties' bitvectors
    val_to_bit: FxHashMap<isize, usize>,
    /// Top-down properties for each node in the MDD
    top_down_properties: Vec<Vec<AllDifferentProperty>>,
    /// Bottom-up properties for each node in the MDD
    bottom_up_properties: Vec<Vec<AllDifferentProperty>>,
    /// For each variable in the scope, indicates how many variables are above and below it in the
    /// MDD.
    map_hall_set: FxHashMap<VariableIndex, (usize, usize)>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl AllDifferent {

    /// Creates a new AllDifferent constraint over variables
    pub fn new(variables: Vec<VariableIndex>) -> Self {
        Self {
            variables,
            domain: FxHashSet::<isize>::default(),
            val_to_bit: FxHashMap::<isize, usize>::default(),
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            map_hall_set: FxHashMap::<VariableIndex, (usize, usize)>::default(),
            layer_in_scope: vec![],
        }
    }

}

impl Constraint for AllDifferent {

    fn init(&mut self, vars: &[Variable]) {
        for variable in self.variables.iter().copied() {
            for value in vars[*variable].iter_domain() {
                self.domain.insert(value);
            }
        }
        for value in self.domain.iter().copied() {
            let bit = self.val_to_bit.len();
            self.val_to_bit.insert(value, bit);
        }
        self.top_down_properties = (0..vars.len() + 1).map(|_| vec![AllDifferentProperty::new(self.domain.len())]).collect::<Vec<Vec<AllDifferentProperty>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| vec![AllDifferentProperty::new(self.domain.len())]).collect::<Vec<Vec<AllDifferentProperty>>>();
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
        // Compute the hall set sizes up and down the mdd. For a given layer l in the scope of the
        // constraint its hall set size up (resp. down) is the number of layer k such that k < l (k
        // > l) and k is in the constraint's scope

        // We sort each variable in the constraint's scope by its position in the ordering
        let mut scope_variable_order = self.variables.iter().copied().map(|v| (ordering[v.0], v)).collect::<Vec<(usize, VariableIndex)>>();
        scope_variable_order.sort_unstable();
        // The hall set sizes are stored as a tuple (size up, size down) and is given, for node i, by (i, n-i)
        let n = self.variables.len();
        for (pos, (_, variable)) in scope_variable_order.iter().copied().enumerate() {
            self.map_hall_set.insert(variable, (pos, n - 1 - pos));
        }
    }

    fn reset_property_top_down(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.top_down_properties[layer][index].value_some_path.reset(0);
        self.top_down_properties[layer][index].value_all_path.reset(!0);
    }

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        // First, we need to map the assignment to its local value as used in the bitsets
        let assignment = *self.val_to_bit.get(&assignment).unwrap();
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        let layer_in_scope = self.is_layer_in_scope(source_layer);

        // For the set A we need to do $A \cap (A^\prime \cup \{assignment\})$. Hence,
        // we can not directly integrate the assignment into A (as is done for the S
        // set, since this is a union of union.
        // Hence, we integrate the assignment into $S^\prime$ and then reverse it.
        let is_in_set = self.top_down_properties[source_layer][source_index].value_all_path.contains(assignment);
        // Only integrate the edge if the layer is in the scope of the constraint.
        if layer_in_scope {
            self.top_down_properties[target_layer][target_index].value_some_path.insert(assignment);
            self.top_down_properties[source_layer][source_index].value_all_path.insert(assignment);
        }

        // Aggregate the source properties into the target properties.
        // Since we need a mutable reference to the properties of layer and a
        // non-mutable references to the source layer we can not directly update the
        // properties. We use the `split_at_mut` method to get two mutable references
        // to non-overlapping slice of the top_down_properties vector. Then, we can use
        // these references to update the properties.
        let (td_properties_above, td_properties_below) = self.top_down_properties.split_at_mut(target_layer);
        td_properties_below[0][target_index].value_all_path.intersect(&td_properties_above[source_layer][source_index].value_all_path);
        td_properties_below[0][target_index].value_some_path.union(&td_properties_above[source_layer][source_index].value_some_path);

        // Reverse the integration of the edge into the $A^\prime$ set.
        if layer_in_scope && !is_in_set{
            self.top_down_properties[source_layer][source_index].value_all_path.remove(assignment);
        }
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index].value_some_path.reset(0);
        self.bottom_up_properties[layer][index].value_all_path.reset(!0);
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let assignment = *self.val_to_bit.get(&assignment).unwrap();
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        let layer_in_scope = self.is_layer_in_scope(target_layer);

        // For the set A we need to do $A \cap (A^\prime \cup \{assignment\})$. Hence,
        // we can not directly integrate the assignment into A (as is done for the S
        // set, since this is a union of union.
        // Hence, we integrate the assignment into $A^\prime$ and then reverse it.
        let is_in_set = self.bottom_up_properties[source_layer][source_index].value_all_path.contains(assignment);
        // Only integrate the edge if the layer is in the scope of the constraint.
        if layer_in_scope {
            self.bottom_up_properties[target_layer][target_index].value_some_path.insert(assignment);
            self.bottom_up_properties[source_layer][source_index].value_all_path.insert(assignment);
        }

        // Aggregate the source properties into the target properties.
        // Since we need a mutable reference to the properties of layer and a
        // non-mutable references to the source layer we can not directly update the
        // properties. We use the `split_at_mut` method to get two mutable references
        // to non-overlapping slice of the top_down_properties vector. Then, we can use
        // these references to update the properties.
        let (bu_properties_above, bu_properties_below) = self.bottom_up_properties.split_at_mut(source_layer);
        bu_properties_above[target_layer][target_index].value_all_path.intersect(&bu_properties_below[0][source_index].value_all_path);
        bu_properties_above[target_layer][target_index].value_some_path.union(&bu_properties_below[0][source_index].value_some_path);

        // Reverse the integration of the edge into the $A^\prime$ set.
        if layer_in_scope && !is_in_set{
            self.bottom_up_properties[source_layer][source_index].value_all_path.remove(assignment);
        }
    }

    /// Returns true if the layer is constrained by self
    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(&self, source: NodeIndex, target: NodeIndex, decision: VariableIndex, assignment: isize) -> bool {
        let assignment = *self.val_to_bit.get(&assignment).unwrap();
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;

        // If the value appears on all path from the source or to the sink, then it will be taken
        // by another variable and can not be assigned to this one.
        if self.top_down_properties[source_layer][source_index].value_all_path.contains(assignment) ||
           self.bottom_up_properties[target_layer][target_index].value_all_path.contains(assignment) {
                return true;
        }
        // If not, we check for Hall-set conditions
        let (hall_set_size_up, hall_set_size_down) = *self.map_hall_set.get(&decision).unwrap();
        let is_on_td_path = self.top_down_properties[source_layer][source_index].value_some_path.contains(assignment);
        let is_on_bu_path = self.bottom_up_properties[target_layer][target_index].value_some_path.contains(assignment);
        if is_on_td_path && hall_set_size_up == self.top_down_properties[source_layer][source_index].value_some_path.size() {
            // First, the variables above are a Hall set: they can take as much values as the union of
            // their domain and this union includes the current assignment.
            return true;
        } else if is_on_bu_path && hall_set_size_down == self.bottom_up_properties[target_layer][target_index].value_some_path.size() {
            // Same but for the variables in later layers.
            return true;
        } else if (is_on_bu_path
            || is_on_td_path)
            && hall_set_size_up + hall_set_size_down == self.top_down_properties[source_layer][source_index].value_some_path.size_union(&self.bottom_up_properties[target_layer][target_index].value_some_path) {
            // Same but for all other variables in the constraint.
            return true;
        }
        false
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        let top_down_property = AllDifferentProperty::new(self.domain.len());
        let bottom_up_property = AllDifferentProperty::new(self.domain.len());
        self.top_down_properties[layer].push(top_down_property);
        self.bottom_up_properties[layer].push(bottom_up_property);
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

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        for word in self.top_down_properties[layer][index].value_all_path.iter() {
            state.write_u64(word);
        }
        for word in self.top_down_properties[layer][index].value_some_path.iter() {
            state.write_u64(word);
        }
        for word in self.bottom_up_properties[layer][index].value_all_path.iter() {
            state.write_u64(word);
        }
        for word in self.bottom_up_properties[layer][index].value_some_path.iter() {
            state.write_u64(word);
        }
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        self.top_down_properties[layer][index].value_all_path == self.top_down_properties[olayer][oindex].value_all_path &&
        self.top_down_properties[layer][index].value_some_path == self.top_down_properties[olayer][oindex].value_some_path &&
        self.bottom_up_properties[layer][index].value_all_path == self.bottom_up_properties[olayer][oindex].value_all_path &&
        self.bottom_up_properties[layer][index].value_some_path == self.bottom_up_properties[olayer][oindex].value_some_path
    }
    
    fn name(&self) -> &'static str {
        "AllDifferent"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
            let node_score = self.top_down_properties[layer][index].value_all_path.size();
            (node_score, i)
        }).collect::<Vec<(usize, usize)>>();
        sorted_nodes.sort_unstable();
        let n = nodes.len() as f64;
        for (rank, (_, i)) in sorted_nodes.iter().copied().enumerate() {
            scores[i] = (rank as f64) / n;
        }
        scores
    }
}

impl std::fmt::Display for AllDifferentProperty {

    fn fmt(&self, f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A: {} - S: {}", self.value_all_path, self.value_some_path)
    }
}

#[cfg(test)]
mod test_all_diff {

    use crate::modelling::*;
    use crate::modelling::variable::Variable;
    use crate::constraints::{AllDifferent, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;

    #[test]
    pub fn test_basic_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0, 1], None);

        all_different(&mut problem, vec![x, y]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
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

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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
        let z = problem.add_variable(vec![0, 1],None);
        all_different(&mut problem, vec![x, y, z]);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
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

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        mdd.to_file("mdd.txt");
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

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![1, 2, 0, 3], &solutions));
        assert!(is_solution(vec![3, 2, 0, 1], &solutions));
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_true() {
        let all_diff = AllDifferent::new(vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)]);
        assert!(all_diff.is_satisfied(&[0, 1, 2]));
    }

    #[test]
    pub fn test_is_satisfied_false() {
        let all_diff = AllDifferent::new(vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)]);
        assert!(!all_diff.is_satisfied(&[0, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope; variable 1 can duplicate freely.
        let all_diff = AllDifferent::new(vec![VariableIndex(0), VariableIndex(2)]);
        assert!(all_diff.is_satisfied(&[0, 0, 1]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let all_diff = AllDifferent::new(vec![]);
        assert!(all_diff.is_satisfied(&[]));
    }

    // --- Additional MDD-level edge cases --- //

    #[test]
    pub fn test_unsat_domain_too_small() {
        // Three variables, each with a 2-value domain, cannot all be pairwise different.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_single_variable_always_sat() {
        // A single variable is trivially all-different from itself.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        all_different(&mut problem, vec![x]);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 3);
        assert!(is_solution(vec![0], &solutions));
        assert!(is_solution(vec![1], &solutions));
        assert!(is_solution(vec![2], &solutions));
    }

    // --- Regression test for the combined (up + down) Hall-set branch of
    // `is_assignment_invalid`. That branch requires the candidate value to appear on
    // BOTH the top-down and the bottom-up "some" path (`is_on_bu_path &&
    // is_on_td_path`), but the necessary condition it implements only needs the
    // value to be a member of the *union* of the two Hall sets, i.e. it should be an
    // OR. As written it silently misses assignments it should reject whenever the
    // value only shows up on one side. This only matters once a node aggregates
    // several top-down/bottom-up histories (as happens once the compiler is forced
    // to merge nodes to respect a width bound); for a fully exact node the leading
    // `value_all_path` check already subsumes it. --- //

    #[test]
    pub fn test_combined_hall_condition_should_use_or_not_and() {
        // Scope v0..v3, decided in that order. We probe the decision for v2
        // (hall_set_size_up = 2, hall_set_size_down = 1).
        let vars = vec![
            Variable::new(vec![0, 1, 2], None),
            Variable::new(vec![0, 1, 2], None),
            Variable::new(vec![0, 1, 2], None),
            Variable::new(vec![0, 1, 2], None),
        ];

        let mut all_diff = AllDifferent::new(vec![VariableIndex(0), VariableIndex(1), VariableIndex(2), VariableIndex(3)]);
        all_diff.init(&vars);
        all_diff.update_variable_ordering(&[0, 1, 2, 3]);

        // Top-down side: merge two constituent histories into the node at v2's
        // layer. Both agree on v0 = 0 but disagree on v1:
        //   - constituent A: v0 = 0, v1 = 1  (still feasible at this point)
        //   - constituent B: v0 = 0, v1 = 2  (will collide with v3 = 2 below, so it
        //     is a dead/spurious prefix -- exactly the kind of state a width-bounded
        //     merge can produce)
        // After merging: top_down some = {0, 1, 2}, all = {0}.
        //
        // Note: we deliberately do NOT call reset_property_top_down on the root
        // (layer 0) or reset_property_bottom_up on the sink (last layer): the real
        // propagate_constraints() never resets those (its passes run over
        // `1..number_layers` and `(0..number_layers-1).rev()` respectively), so they
        // must be left at the all-zero state `init()` gives them. Resetting them here
        // would set their value_all_path to all-ones (the reset's identity value),
        // which turns every downstream intersection into a no-op and corrupts the
        // whole chain -- that was an earlier bug in this test, not in the compiler.
        all_diff.reset_property_top_down(NodeIndex(1, 0));
        all_diff.update_property_top_down(NodeIndex(0, 0), NodeIndex(1, 0), 0);

        all_diff.reset_property_top_down(NodeIndex(2, 0));
        all_diff.update_property_top_down(NodeIndex(1, 0), NodeIndex(2, 0), 1); // constituent A
        all_diff.update_property_top_down(NodeIndex(1, 0), NodeIndex(2, 0), 2); // constituent B

        // Bottom-up side: a single, exact history with v3 = 2.
        all_diff.reset_property_bottom_up(NodeIndex(3, 0));
        all_diff.update_property_bottom_up(NodeIndex(4, 0), NodeIndex(3, 0), 2);

        // For the real constituent (v0=0, v1=1, v3=2), the other 3 variables
        // already use exactly 3 distinct values {0, 1, 2} -- a saturated Hall set
        // over every variable except v2. Assigning v2 = 1 collides with v1 and must
        // be rejected, even though 1 only appears on the top-down side.
        //
        // This currently fails: `is_on_bu_path` is false (1 isn't in {2}), so the
        // buggy `&&` short-circuits and the assignment is wrongly accepted.
        assert!(all_diff.is_assignment_invalid(NodeIndex(2, 0), NodeIndex(3, 0), VariableIndex(2), 1));
    }

}
