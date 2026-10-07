//! `NotEquals`: two variables must take different values.
//!
//! # Meaning
//!
//! Constrains $x$ and $y$ to take different values, that is $x \neq y$.
//! We allow `NotEquals::new(x, x, ...)` with a warning, it may be useful to impose such a constraint
//! on the fly, during solving, when learning constraints. This would imply that the problem is
//! UNSAT and stop compilation.
//!
//! # How it compiles
//!
//! A node carries the set of values taken by `x` or `y` on the paths that reach it (top-down) or
//! leave it (bottom-up), stored as a bitset.
//!
//! ## Property update
//!
//! Updating a property is done by unioning the parent (top-down) or child (bottom-up) property
//! with the target node's property. If the layer is in scope, the edge value is also added to the
//! target bitset.
//! The intuition for doing a union is the following. Each value appearing in the source property
//! represents a possible value for `x` or `y` in the corresponding top-down or bottom-up paths; hence,
//! all these values are still possible for the target node, and we add them to the node's
//! property.
//!
//! ## Node merging
//!
//! In the same manner as the update, when merging nodes, their properties are unioned. Each value
//! that was possible for either node is still possible, so this gives a correct relaxation.
//!
//! ## Edge filtering
//!
//! Because the constraint has only two variables, an edge can be removed in two situations only
//! (assuming the layer of `x` is less than the layer of `y`):
//!
//! 1. An edge can be filtered for a node in the layer of `y` if each path reaching it has a single
//!    value for `x` with value $\{v\}$. Then, the edge associated with $v$ can be removed.
//! 2. An edge can be filtered for a node in the layer of `x` if each path leaving it has a single
//!    value for `y` with value $\{v\}$. Then, the edge associated with $v$ can be removed.
//!
//! # Design notes
//!
//! - **One bit per value of the union of both domains**, shared by every property of a constraint
//!   through an [`Arc`]. The bit numbers are arbitrary and only meaningful inside one constraint;
//!   two `NotEquals` never compare their bitsets.
//! - **Domains are read once, in [`NotEquals::new`].** Shrinking a variable's domain afterwards
//!   is fine; adding a value that was not in either domain at construction time makes
//!   compilation panic, because that value has no bit.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, NotEquals};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let x = problem.add_variable(vec![0, 1, 2], None);
//! let y = problem.add_variable(vec![0, 1, 2], None);
//! let ne = NotEquals::new(x, y, &problem);
//!
//! assert!(ne.is_satisfied(&[0, 1]));
//! assert!(!ne.is_satisfied(&[2, 2]));
//! ```
use super::*;
use crate::modelling::*;
use crate::utils::Bitset;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`NotEquals`]: the set of values taken by `x` or `y` along the paths that
/// reach the node (top-down) or leave it (bottom-up).
///
/// Bit `b` is set when some such path assigns the value that `map` sends to `b`. The empty set
/// means no path has assigned either variable yet, which is the identity of `merge`. `merge` is
/// the union, so a merged node over-approximates both parents.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct NotEqualsProperty {
    set: Bitset,
    map: Arc<FxHashMap<isize, usize>>,
}

impl NotEqualsProperty {
    fn new(map: Arc<FxHashMap<isize, usize>>) -> Self {
        Self {
            set: Bitset::new(map.len()),
            map,
        }
    }
}

/// The constraint $x \neq y$. See the [module documentation](self) for the semantics and for how
/// it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct NotEquals {
    x: VariableIndex,
    y: VariableIndex,
    /// Map each value in the union of the domains of `x` and `y` to a bit.
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    /// Layer of variable `x`, set to `usize::MAX` until `update_variable_ordering` is called.
    layer_x: usize,
    /// Layer of variable `y`, set to `usize::MAX` until `update_variable_ordering` is called.
    layer_y: usize,
}

impl NotEquals {
    /// Builds `x != y`.
    ///
    /// The bit numbering is taken from the values in the domains of `x` and `y` *at this point*;
    /// see the module notes about domains that change later. `x == y` is accepted: a warning is
    /// emitted and the constraint has no satisfying assignment.
    pub fn new(x: VariableIndex, y: VariableIndex, problem: &Problem) -> Self {
        if x == y {
            log::warn!("Building the constraint NotEquals(x, y) with x == y.");
        }
        let mut domains = FxHashSet::<isize>::default();
        domains.extend(problem[x].iter_domain());
        domains.extend(problem[y].iter_domain());
        let val_to_bit = Arc::new(
            domains
                .iter()
                .copied()
                .enumerate()
                .map(|(bit, value)| (value, bit))
                .collect(),
        );
        Self {
            x,
            y,
            val_to_bit,
            layer_x: usize::MAX,
            layer_y: usize::MAX,
        }
    }
}

impl Constraint for NotEquals {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        ConstraintShapeKey::NotEquals
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        // Two independent `if`s, not `else if`: when `x == y` both layers must be set.
        for (layer, &variable) in order.iter().enumerate() {
            if variable == self.x {
                self.layer_x = layer;
            }
            if variable == self.y {
                self.layer_y = layer;
            }
        }
        debug_assert!(self.layer_x != usize::MAX);
        debug_assert!(self.layer_y != usize::MAX);
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(self.layer_x != usize::MAX);
        debug_assert!(self.layer_y != usize::MAX);
        layer == self.layer_x || layer == self.layer_y
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        _layer: usize,
        assignment: isize,
    ) -> bool {
        // `x != x` has no solution: every edge of its layer is invalid. The sets cannot see this,
        // since there is no second variable to compare against.
        if self.x == self.y {
            return true;
        }
        let parent = parent.as_any().downcast_ref::<NotEqualsProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of NotEqualsProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<NotEqualsProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of NotEqualsProperty",
                    child.name()
                );
        });

        let bit = *self.val_to_bit.get(&assignment).unwrap();
        // The edge belongs to the layer of x or of y. Say x comes first. On an x edge the parent
        // set is empty (nothing assigned above), so only the child set can matter: does every
        // path below give y this very value? On a y edge the child set is empty (nothing
        // assigned below), so only the parent set matters: does every path above give x this
        // very value? The test is the same in both cases, and the ordering of x and y needs no
        // special handling.
        (parent.set.contains(bit) && parent.set.size() == 1)
            || (child.set.contains(bit) && child.set.size() == 1)
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new([self.x, self.y].into_iter())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover both variables, otherwise
        // this panics on the out-of-range index.
        assignment[*self.x] != assignment[*self.y]
    }

    fn name(&self) -> &'static str {
        "Not Equals"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(NotEqualsProperty::new(self.val_to_bit.clone()))
    }
}

impl ConstraintProperty for NotEqualsProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<NotEqualsProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        if in_scope {
            let bit = *self.map.get(&assignment).unwrap();
            self.set.insert(bit);
        }
        self.set.union(&other.set);
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<NotEqualsProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        self.set.union(&other.set);
    }

    fn order_key(&self) -> Vec<f64> {
        vec![self.set.size() as f64]
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for word in self.set.iter() {
            hasher.write_u64(word);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<NotEqualsProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling eq on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        self.set == other.set
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "NotEqualsProperty"
    }
}

#[cfg(test)]
mod test_not_equals {
    use super::NotEqualsProperty;
    use crate::constraints::{
        AllDifferent, Constraint, ConstraintProperty, ConstraintShapeKey, NotEquals,
    };
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHasher;
    use std::hash::Hasher;
    use std::sync::Arc;

    // ----------------------------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------------------------

    /// Two variables with the given domains, and the constraint between them.
    fn pair(dom_x: Vec<isize>, dom_y: Vec<isize>) -> (Problem, VariableIndex, VariableIndex) {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom_x, None);
        let y = problem.add_variable(dom_y, None);
        not_equals(&mut problem, x, y);
        (problem, x, y)
    }

    /// Compiles every constraint of `problem` with the given variable order. `width` is the
    /// refinement budget: `usize::MAX` refines to an exact MDD, `1` keeps the initial relaxation.
    fn compile(problem: Problem, order: OrderingHeuristic, width: usize) -> Mdd {
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            order,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        if width > 1 {
            mdd.refine(width);
        }
        mdd
    }

    /// All assignments accepted by the MDD, sorted so they can be compared as sets.
    fn accepted(mdd: &Mdd) -> Vec<Vec<isize>> {
        let mut solutions = get_all_solutions(mdd);
        solutions.sort();
        solutions
    }

    /// The expected solutions, written independently of `NotEquals`: every pair from the two
    /// domains whose entries differ.
    fn expected_pairs(dom_x: &[isize], dom_y: &[isize]) -> Vec<Vec<isize>> {
        let mut pairs = vec![];
        for &a in dom_x {
            for &b in dom_y {
                if a != b {
                    pairs.push(vec![a, b]);
                }
            }
        }
        pairs.sort();
        pairs
    }

    /// A property whose set holds exactly the given values, built the way the compiler builds
    /// them: by `update` calls that are in scope.
    fn property_with(ne: &NotEquals, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let empty = ne.identity_property();
        let mut property = ne.identity_property();
        for &value in values {
            property.update(&*empty, value, true);
        }
        property
    }

    fn set_of(property: &dyn ConstraintProperty) -> Vec<u64> {
        property
            .as_any()
            .downcast_ref::<NotEqualsProperty>()
            .unwrap()
            .set
            .iter()
            .collect()
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    fn ne_with_domain(domain: Vec<isize>) -> NotEquals {
        let (problem, x, y) = pair(domain.clone(), domain);
        NotEquals::new(x, y, &problem)
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_truth_table() {
        // Written out by hand over a domain with a negative and a non-contiguous value.
        let ne = ne_with_domain(vec![-1, 0, 5]);
        let table: [([isize; 2], bool); 9] = [
            ([-1, -1], false),
            ([-1, 0], true),
            ([-1, 5], true),
            ([0, -1], true),
            ([0, 0], false),
            ([0, 5], true),
            ([5, -1], true),
            ([5, 0], true),
            ([5, 5], false),
        ];
        for (assignment, expected) in table {
            assert_eq!(ne.is_satisfied(&assignment), expected, "{assignment:?}");
        }
    }

    #[test]
    fn is_satisfied_with_disjoint_domains_always_holds() {
        let (problem, x, y) = pair(vec![0, 1], vec![2, 3]);
        let ne = NotEquals::new(x, y, &problem);
        for a in [0, 1] {
            for b in [2, 3] {
                assert!(ne.is_satisfied(&[a, b]));
            }
        }
    }

    #[test]
    fn is_satisfied_reads_the_variables_by_index() {
        // Constraint between the first and the third variable: the middle one must be ignored.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let _middle = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let ne = NotEquals::new(x, y, &problem);
        assert!(ne.is_satisfied(&[1, 9, 2]));
        assert!(!ne.is_satisfied(&[1, 9, 1]));
        assert!(!ne.is_satisfied(&[1, 2, 1]));
    }

    #[test]
    fn is_satisfied_is_never_true_for_x_not_equal_x() {
        let (problem, x, _y) = pair(vec![0, 1, 2], vec![0, 1, 2]);
        let ne = NotEquals::new(x, x, &problem);
        for a in 0..3 {
            assert!(!ne.is_satisfied(&[a, 0]));
        }
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        let (problem, x, y) = pair(vec![0, 1], vec![0, 1]);
        NotEquals::new(x, y, &problem).is_satisfied(&[0]);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_and_structural_key() {
        let (problem, x, y) = pair(vec![0, 1], vec![0, 1]);
        let ne = NotEquals::new(x, y, &problem);
        assert_eq!(ne.iter_scope().collect::<Vec<_>>(), vec![x, y]);
        assert_eq!(ne.structural_key(&problem), ConstraintShapeKey::NotEquals);
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        let mut ne = NotEquals::new(x, y, &problem);
        // y first, then z (not in scope), then x.
        ne.update_variable_ordering(&[y, z, x]);
        assert!(ne.is_layer_in_scope(0));
        assert!(!ne.is_layer_in_scope(1));
        assert!(ne.is_layer_in_scope(2));
    }

    #[test]
    fn layers_in_scope_with_a_repeated_variable() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        let mut ne = NotEquals::new(x, x, &problem);
        ne.update_variable_ordering(&[z, x]);
        assert!(!ne.is_layer_in_scope(0));
        assert!(ne.is_layer_in_scope(1));
    }

    // The layer sentinel is only checked in debug builds.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let (problem, x, y) = pair(vec![0, 1], vec![0, 1]);
        NotEquals::new(x, y, &problem).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let (problem, x, y) = pair(vec![0, 1], vec![0, 1]);
        let mut ne = NotEquals::new(x, y, &problem);
        // `y` does not appear in the order.
        ne.update_variable_ordering(&[x]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn identity_property_is_empty() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let identity = ne.identity_property();
        assert!(set_of(&*identity).iter().all(|&word| word == 0));
        assert_eq!(identity.order_key(), vec![0.0]);
        assert!(same(&*identity, &*ne.identity_property()));
    }

    #[test]
    fn update_in_scope_adds_the_assigned_value() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let parent = ne.identity_property();
        let mut child = ne.identity_property();
        child.update(&*parent, 2, true);
        assert_eq!(child.order_key(), vec![1.0]);
        let bit = ne.val_to_bit[&2];
        assert!(
            child
                .as_any()
                .downcast_ref::<NotEqualsProperty>()
                .unwrap()
                .set
                .contains(bit)
        );
    }

    #[test]
    fn update_out_of_scope_only_passes_the_parent_state_through() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let parent = property_with(&ne, &[1]);
        let mut child = ne.identity_property();
        child.update(&*parent, 2, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn update_in_scope_keeps_the_parent_values_too() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let parent = property_with(&ne, &[1]);
        let mut child = ne.identity_property();
        child.update(&*parent, 2, true);
        assert!(same(&*child, &*property_with(&ne, &[1, 2])));
    }

    #[test]
    fn merge_is_the_union() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let mut merged = property_with(&ne, &[0]);
        merged.merge(&*property_with(&ne, &[2]));
        assert!(same(&*merged, &*property_with(&ne, &[0, 2])));
        // Merging with itself or with the identity changes nothing.
        let before = property_with(&ne, &[0, 2]);
        merged.merge(&*before);
        merged.merge(&*ne.identity_property());
        assert!(same(&*merged, &*before));
    }

    #[test]
    fn eq_and_hash_depend_only_on_the_set() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        // Same set reached in two different orders.
        let a = property_with(&ne, &[0, 2]);
        let b = property_with(&ne, &[2, 0]);
        assert!(same(&*a, &*b));
        assert_eq!(hash_of(&*a), hash_of(&*b));
        // A different set is a different state.
        let c = property_with(&ne, &[0]);
        assert!(!same(&*a, &*c));
        assert_ne!(hash_of(&*a), hash_of(&*c));
    }

    #[test]
    fn order_key_is_the_number_of_values() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        assert_eq!(property_with(&ne, &[]).order_key(), vec![0.0]);
        assert_eq!(property_with(&ne, &[1]).order_key(), vec![1.0]);
        assert_eq!(property_with(&ne, &[0, 1, 2]).order_key(), vec![3.0]);
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid, called directly
    // ----------------------------------------------------------------------------------------

    #[test]
    fn an_edge_is_invalid_when_the_other_side_forces_the_same_value() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let empty = ne.identity_property();
        let only_1 = property_with(&ne, &[1]);

        // Parent (above) forces x = 1: assigning y = 1 is invalid, y = 2 is fine.
        assert!(ne.is_assignment_invalid(&*only_1, &*empty, 1, 1));
        assert!(!ne.is_assignment_invalid(&*only_1, &*empty, 1, 2));
        // Child (below) forces y = 1: assigning x = 1 is invalid, x = 2 is fine.
        assert!(ne.is_assignment_invalid(&*empty, &*only_1, 0, 1));
        assert!(!ne.is_assignment_invalid(&*empty, &*only_1, 0, 2));
    }

    #[test]
    fn an_edge_is_kept_when_the_other_side_is_empty_or_not_forced() {
        let ne = ne_with_domain(vec![0, 1, 2]);
        let empty = ne.identity_property();
        let one_or_two = property_with(&ne, &[1, 2]);

        // Nothing is known on either side.
        for value in 0..3 {
            assert!(!ne.is_assignment_invalid(&*empty, &*empty, 0, value));
        }
        // Two possible values above: some path can still use any assignment, so the edge stays
        // even for a value in the set. This is where the relaxation comes from.
        for value in 0..3 {
            assert!(!ne.is_assignment_invalid(&*one_or_two, &*empty, 1, value));
            assert!(!ne.is_assignment_invalid(&*empty, &*one_or_two, 0, value));
        }
    }

    #[test]
    fn is_assignment_invalid_is_always_true_for_x_not_equal_x() {
        let (problem, x, _y) = pair(vec![0, 1], vec![0, 1]);
        let ne = NotEquals::new(x, x, &problem);
        let empty = ne.identity_property();
        for value in 0..2 {
            assert!(ne.is_assignment_invalid(&*empty, &*empty, 0, value));
        }
    }

    #[test]
    #[should_panic(expected = "instead of NotEqualsProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let ne = NotEquals::new(x, y, &problem);
        let other = AllDifferent::new(vec![x, y], &problem).identity_property();
        let own = ne.identity_property();
        ne.is_assignment_invalid(&*other, &*own, 0, 0);
    }

    #[test]
    fn a_domain_wider_than_one_word_works() {
        // 70 values need more than the 64 bits of the inline bitset.
        let domain: Vec<isize> = (0..70).collect();
        let ne = ne_with_domain(domain);
        let empty = ne.identity_property();
        let only_high = property_with(&ne, &[69]);
        assert!(ne.is_assignment_invalid(&*only_high, &*empty, 1, 69));
        assert!(!ne.is_assignment_invalid(&*only_high, &*empty, 1, 3));
        let mut both = property_with(&ne, &[3]);
        both.merge(&*only_high);
        assert_eq!(both.order_key(), vec![2.0]);
        assert!(!ne.is_assignment_invalid(&*both, &*empty, 1, 69));
    }

    // ----------------------------------------------------------------------------------------
    // Compiled into an MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn exact_mdd_accepts_exactly_the_different_pairs() {
        for (dom_x, dom_y) in [
            (vec![0, 1], vec![0, 1]),
            (vec![0, 1, 2], vec![0, 1, 2]),
            // Different, overlapping domains.
            (vec![0, 1, 2], vec![1, 2, 3]),
            // Disjoint domains: nothing is removed.
            (vec![0, 1], vec![2, 3]),
            // Negative and non-contiguous values.
            (vec![-2, 0, 7], vec![-2, 7]),
        ] {
            for order in [vec![0, 1], vec![1, 0]] {
                let (problem, _, _) = pair(dom_x.clone(), dom_y.clone());
                let mdd = compile(
                    problem,
                    OrderingHeuristic::Custom(order.clone()),
                    usize::MAX,
                );
                assert_eq!(
                    accepted(&mdd),
                    expected_pairs(&dom_x, &dom_y),
                    "domains {dom_x:?} / {dom_y:?}, order {order:?}"
                );
            }
        }
    }

    #[test]
    fn exact_mdd_with_a_default_ordering() {
        let (problem, _, _) = pair(vec![0, 1, 2], vec![0, 1, 2]);
        let mdd = compile(problem, OrderingHeuristic::MinDomMaxLinked, usize::MAX);
        assert_eq!(accepted(&mdd), expected_pairs(&[0, 1, 2], &[0, 1, 2]));
    }

    #[test]
    fn a_wide_domain_compiles_exactly() {
        // More than 64 values, so the bitset spans two words.
        let domain: Vec<isize> = (0..65).collect();
        let (problem, _, _) = pair(domain.clone(), domain.clone());
        let mdd = compile(problem, OrderingHeuristic::Custom(vec![0, 1]), usize::MAX);
        assert_eq!(accepted(&mdd).len(), 65 * 64);
    }

    #[test]
    fn two_singletons_with_different_values_are_satisfiable() {
        // Regression test. Bottom-up propagation used to look at the wrong layer, folded x's
        // value into y's downstream set, and then removed the only edge out of the root, so a
        // trivially satisfiable problem came out UNSAT. Both orders exercise both directions.
        for order in [vec![0, 1], vec![1, 0]] {
            let (problem, _, _) = pair(vec![0], vec![1]);
            let mdd = compile(problem, OrderingHeuristic::Custom(order), 1);
            assert!(!mdd.is_unsat());
            assert_eq!(mdd.get_solution(), Some(vec![0, 1]));
        }
    }

    #[test]
    fn two_singletons_with_the_same_value_are_unsat() {
        // The mirror of the regression test above: a real conflict must still be detected.
        for order in [vec![0, 1], vec![1, 0]] {
            let (problem, _, _) = pair(vec![5], vec![5]);
            let mdd = compile(problem, OrderingHeuristic::Custom(order), 1);
            assert!(mdd.is_unsat());
            assert_eq!(mdd.get_solution(), None);
        }
    }

    #[test]
    fn x_not_equal_x_is_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let ne = NotEquals::new(x, x, &problem);
        problem.add_constraint(ne);
        let mdd = compile(problem, OrderingHeuristic::Custom(vec![0]), usize::MAX);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    fn relaxations_never_lose_a_solution() {
        // For every width budget the MDD may accept too much but never too little.
        let dom: Vec<isize> = (0..4).collect();
        let expected = expected_pairs(&dom, &dom);
        for width in [1, 2, 3, 4, usize::MAX] {
            let (problem, _, _) = pair(dom.clone(), dom.clone());
            let mdd = compile(problem, OrderingHeuristic::Custom(vec![0, 1]), width);
            let solutions = accepted(&mdd);
            for pair in &expected {
                assert!(
                    is_solution(pair.clone(), &solutions),
                    "width {width} lost {pair:?}"
                );
            }
        }
    }

    #[test]
    fn a_domain_shrunk_after_construction_still_works() {
        let (mut problem, x, _y) = pair(vec![0, 1, 2], vec![0, 1, 2]);
        problem[x].set_domain(vec![1]);
        let mdd = compile(problem, OrderingHeuristic::Custom(vec![0, 1]), usize::MAX);
        assert_eq!(accepted(&mdd), vec![vec![1, 0], vec![1, 2]]);
    }

    #[test]
    #[should_panic]
    fn a_value_added_after_construction_panics() {
        // Documented limitation: the value 9 has no bit.
        let (mut problem, x, _y) = pair(vec![0, 1], vec![0, 1]);
        problem[x].set_domain(vec![0, 1, 9]);
        let mdd = compile(problem, OrderingHeuristic::Custom(vec![0, 1]), usize::MAX);
        drop(mdd);
    }

    #[test]
    fn a_chain_of_inequalities_is_all_different() {
        // x != y, y != z and x != z over three values leave the 3! permutations.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);
        let mdd = compile(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            usize::MAX,
        );

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    if a != b && b != c && a != c {
                        expected.push(vec![a, b, c]);
                    }
                }
            }
        }
        expected.sort();
        assert_eq!(expected.len(), 6);
        assert_eq!(accepted(&mdd), expected);
    }
}
