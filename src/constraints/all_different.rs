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
    /// Creates a new property with bitsiets of nb_words 64-bit unsigned integers
    pub fn new(n: usize, map: Arc<FxHashMap<isize, usize>>) -> Self {
        let value_all_path = Bitset::new(n);
        let value_some_path = Bitset::new(n);
        Self {
            map,
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
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    /// For each variable in the scope, indicates how many variables are above and below it in the
    /// MDD.
    map_hall_set: FxHashMap<VariableIndex, (usize, usize)>,
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
        let layer_in_scope = (0..(variables.len() / 64 + 1))
            .map(|_| 0)
            .collect::<Vec<u64>>();
        Self {
            variables,
            domain,
            val_to_bit,
            map_hall_set: FxHashMap::<VariableIndex, (usize, usize)>::default(),
            layer_in_scope,
        }
    }
}

impl Constraint for AllDifferent {
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
        let mut scope_variable_order = self
            .variables
            .iter()
            .copied()
            .map(|v| (ordering[v.0], v))
            .collect::<Vec<(usize, VariableIndex)>>();
        scope_variable_order.sort_unstable();
        // The hall set sizes are stored as a tuple (size up, size down) and is given, for node i, by (i, n-i)
        let n = self.variables.len();
        for (pos, (_, variable)) in scope_variable_order.iter().copied().enumerate() {
            self.map_hall_set.insert(variable, (pos, n - 1 - pos));
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

    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
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
        if parent.value_all_path_td.contains(bit) || child.value_all_path_bu.contains(bit) {
            return true;
        }
        // TODO: Hall-set conditions
        false
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AllDifferentProperty::new(
            self.domain.len(),
            self.val_to_bit.clone(),
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
        let tda = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_all_path_td.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        let tds = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_some_path_td.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        let bua = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_all_path_bu.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        let bus = self
            .map
            .iter()
            .filter(|&(_, bit)| self.value_some_path_bu.contains(*bit))
            .map(|(value, _)| format!("{}", value))
            .collect::<Vec<String>>()
            .join(", ");
        write!(
            f,
            "TD: all {} - some {}\nBU: all {} - some {}",
            tda, tds, bua, bus,
        )
    }
}

#[cfg(test)]
mod test_all_diff {

    use crate::constraints::{AllDifferent, Constraint};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::variable::Variable;
    use crate::modelling::*;

    #[test]
    pub fn test_basic_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0, 1], None);

        all_different(&mut problem, vec![x, y]);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
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

        let mdd = Mdd::new(
            problem,
            1,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
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

        let mdd = Mdd::new(
            problem,
            1,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
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

        let mdd = Mdd::new(
            problem,
            1,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
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

        let mdd = Mdd::new(
            problem,
            1,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
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

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::Custom(vec![0, 1]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
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

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![1, 2, 0, 3], &solutions));
        assert!(is_solution(vec![3, 2, 0, 1], &solutions));
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_true() {
        let all_diff =
            AllDifferent::new(vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)]);
        assert!(all_diff.is_satisfied(&[0, 1, 2]));
    }

    #[test]
    pub fn test_is_satisfied_false() {
        let all_diff =
            AllDifferent::new(vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)]);
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

        let mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
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

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
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

        let mut all_diff = AllDifferent::new(vec![
            VariableIndex(0),
            VariableIndex(1),
            VariableIndex(2),
            VariableIndex(3),
        ]);
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
        assert!(all_diff.is_assignment_invalid(
            NodeIndex(2, 0),
            NodeIndex(3, 0),
            VariableIndex(2),
            1
        ));
    }
}
