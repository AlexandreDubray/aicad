//! `AllDifferent`: the variables of the scope must take pairwise different values.
//!
//! # Meaning
//!
//! Constrains $x_1, \dots, x_n$ so that $x_i \neq x_j$ for every $i \neq j$. An empty scope or a
//! scope with a single variable is always satisfied.
//!
//! A variable that appears twice in the scope can never differ from itself, so such a constraint
//! has no solution. As for [`NotEquals`] with `x == y`, it is accepted with a
//! warning: it is useful to be able to add it on the fly (for instance when learning constraints
//! on an UNSAT problem) and let the compilation conclude that the problem is UNSAT.
//!
//! # How it compiles
//!
//! A node carries a pair of sets of values, $(A, S)$, stored as bitsets (one bit per value of the
//! union of the domains of the scope). They describe the paths that reach the node (top-down) or
//! leave it (bottom-up), counting only the assignments of variables of the scope:
//!
//! - $A$ holds the values taken on *every* such path;
//! - $S$ holds the values taken on *some* such path.
//!
//! The same property is used in both directions. For a single path, $A = S$ is the set of its
//! values.
//!
//! ## Property update
//!
//! A node folds the properties of its parents (top-down) or children (bottom-up), each extended
//! with the value of the edge when the layer is in scope:
//!
//! $$(A, S) \otimes v = (A \cup \{v\}, S \cup \{v\})$$
//!
//! The results of the different edges are combined with $A \leftarrow A \cap A'$ and
//! $S \leftarrow S \cup S'$. Out of scope layers copy the property unchanged.
//!
//! ## Node merging
//!
//! Merging two nodes combines their properties the same way,
//! $(A, S) \oplus (A', S') = (A \cap A', S \cup S')$. A value is on all the paths of the merged
//! node only if it was on all the paths of both nodes, and on some path if it was on some path of
//! either, so the merged node over-approximates both.
//!
//! ## Edge filtering
//!
//! An edge with value $v$ at a layer of the scope, between a parent with property $(A_p, S_p)$ and
//! a child with property $(A_c, S_c)$, is removed in two situations:
//!
//! 1. $v \in A_p \cup A_c$: $v$ is taken on every path above, or on every path below, so no
//!    assignment going through this edge can take it again.
//! 2. **Hall set**: let $u$ and $d$ be the numbers of scope variables above and below the
//!    layer. If $|S_p \cup S_c| = u + d$ and $v \in S_p \cup S_c$, the other $u + d$ variables use
//!    at least all of these $u + d$ values together, so $v$ is not available.
//!
//! # Design notes
//!
//! - **One bit per value of the union of the domains of the scope**, shared by every property of
//!   the constraint through an [`Arc`]. Bit numbers are arbitrary and only meaningful inside one
//!   constraint.
//! - **Domains are read once, in [`AllDifferent::new`].** Shrinking a domain afterwards is fine;
//!   adding a value that was not there at construction time makes compilation panic, because
//!   that value has no bit.
//! - **Two neutral properties.** [`identity_property`](Constraint::identity_property) is the seed
//!   of a fold: $A$ is full (the neutral element of the intersection), $S$ is empty.
//!   [`empty_property`](Constraint::empty_property) is the property of the root and of the sink:
//!   no path has assigned anything, so both sets are empty.
//! - **Layers that are not in the scope are ignored.** The Hall-set counts are only about the
//!   variables of the scope, whatever their position in the order.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{AllDifferent, Constraint};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2], None);
//! let all_diff = AllDifferent::new(vars, &problem);
//!
//! assert!(all_diff.is_satisfied(&[2, 0, 1]));
//! assert!(!all_diff.is_satisfied(&[2, 0, 2]));
//! ```
//!
//! # References
//!
//! - Hoda, van Hoeve and Hooker, *A systematic approach to MDD-based constraint programming*, CP 2010.
//!
use super::*;
use crate::modelling::VariableIndex;
use crate::utils::Bitset;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`AllDifferent`]: the pair $(A, S)$ of the values taken on all, respectively
/// on some, of the paths that reach the node (top-down) or leave it (bottom-up).
///
/// Bit `b` is set when `map` sends a value to `b` that satisfies the condition. `merge` is
/// $A \cap A'$ and $S \cup S'$, so a merged node over-approximates both parents. See the
/// [module documentation](self).
#[derive(Clone, PartialEq, Eq, deepsize::DeepSizeOf)]
struct AllDifferentProperty {
    map: Arc<FxHashMap<isize, usize>>,
    /// $A$: values taken on all paths.
    value_all_path: Bitset,
    /// $S$: values taken on some path.
    value_some_path: Bitset,
}

impl AllDifferentProperty {
    /// Creates a property whose bitsets have room for `n` bits. `all_path_reset` is the starting
    /// word of `value_all_path`: `!0` (all ones, the identity of the intersection) for the seed of
    /// a fold, `0` (empty) for the property of an empty path. See
    /// `Constraint::identity_property` and `Constraint::empty_property`.
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

/// The constraint that the variables of its scope take pairwise different values. See the
/// [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AllDifferent {
    /// Scope of the constraint.
    variables: Vec<VariableIndex>,
    /// Union of the domains of the variables in the scope, read at construction time.
    domain: FxHashSet<isize>,
    /// Map each value of the joint domains to a bit in the properties' bitsets.
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    /// For each layer of the scope, the numbers of scope variables above and below it. Layers out
    /// of the scope hold `(0, 0)` and are never read. Empty until `update_variable_ordering`.
    hall_set_bounds: Vec<(usize, usize)>,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
    /// True if a variable appears more than once in the scope: no assignment is accepted.
    has_duplicate: bool,
}

impl AllDifferent {
    /// Builds the constraint over `variables`.
    ///
    /// The bit numbering is taken from the values in the domains of the variables *at this
    /// point*; see the module notes about domains that change later. A variable repeated in the
    /// scope is accepted: a warning is emitted and the constraint has no satisfying assignment.
    pub fn new(variables: Vec<VariableIndex>, problem: &Problem) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        let has_duplicate = distinct.len() != variables.len();
        if has_duplicate {
            log::warn!(
                "Building the constraint AllDifferent with a variable repeated in its scope."
            );
        }
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
            has_duplicate,
        }
    }
}

impl Constraint for AllDifferent {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        ConstraintShapeKey::AllDifferent {
            arity: self.variables.len(),
        }
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        let mut scope_layers = Vec::with_capacity(scope.len());
        self.layer_in_scope = vec![0; order.len() / 64 + 1];
        for (layer, &variable) in order.iter().enumerate() {
            if scope.contains(&variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                scope_layers.push(layer);
            }
        }
        debug_assert_eq!(
            scope_layers.len(),
            scope.len(),
            "a variable of the scope is missing from the ordering"
        );

        // The i-th variable of the scope (by layer) has i variables above and n - 1 - i below.
        self.hall_set_bounds = vec![(0, 0); order.len()];
        let n = scope_layers.len();
        for (pos, layer) in scope_layers.into_iter().enumerate() {
            self.hall_set_bounds[layer] = (pos, n - 1 - pos);
        }
    }

    /// Returns true if the layer is constrained by self
    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(
            !self.layer_in_scope.is_empty(),
            "update_variable_ordering has not been called"
        );
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover the scope, otherwise this
        // panics on the out-of-range index. A repeated variable repeats its value: not satisfied.
        let mut set = FxHashSet::<isize>::default();
        for variable in self.variables.iter().copied() {
            let value = assignment[*variable];
            if !set.insert(value) {
                return false;
            }
        }
        true
    }

    fn name(&self) -> &'static str {
        "AllDifferent"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        layer: usize,
        assignment: isize,
    ) -> bool {
        // A repeated variable has no solution: every edge of its layer is invalid. The sets
        // cannot see this, since they only record values.
        if self.has_duplicate {
            return true;
        }
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

        // Hall set: the other variables of the scope (`up` above, `down` below) use at least as
        // many different values as they are, and these are exactly the values in S.
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

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<AllDifferentProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        self.value_some_path.union(&other.value_some_path);
        self.value_all_path.intersect(&other.value_all_path);
    }

    fn order_key(&self) -> Vec<f64> {
        vec![
            self.value_all_path.size() as f64,
            self.value_some_path.size() as f64,
        ]
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
    use super::AllDifferentProperty;
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

    /// One variable per domain, and an AllDifferent over the variables listed in `scope`.
    fn scoped(domains: &[Vec<isize>], scope: &[usize]) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        all_different(&mut problem, scope.iter().map(|&i| vars[i]).collect());
        (problem, vars)
    }

    /// One variable per domain, all in the scope.
    fn full(domains: &[Vec<isize>]) -> (Problem, Vec<VariableIndex>) {
        scoped(domains, &(0..domains.len()).collect::<Vec<_>>())
    }

    /// The constraint over variables `0..domains.len()` with the identity ordering already set.
    fn ad(domains: &[Vec<isize>]) -> AllDifferent {
        let (problem, vars) = full(domains);
        let mut ad = AllDifferent::new(vars.clone(), &problem);
        ad.update_variable_ordering(&vars);
        ad
    }

    /// Compiles every constraint of `problem` with the given variable order. `width` is the
    /// refinement budget: `usize::MAX` refines as far as possible, `1` keeps the initial
    /// relaxation.
    fn compile(problem: Problem, order: Vec<usize>, width: usize) -> Mdd {
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(order),
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
        if mdd.is_unsat() {
            return vec![];
        }
        let mut solutions = get_all_solutions(mdd);
        solutions.sort();
        solutions
    }

    /// The expected solutions, written independently of `AllDifferent`: the cartesian product of
    /// the domains, keeping the tuples whose entries at the `scope` positions are pairwise
    /// different (compared two by two, not through a set).
    fn expected(domains: &[Vec<isize>], scope: &[usize]) -> Vec<Vec<isize>> {
        expected_all(domains, &[scope])
    }

    /// Same as `expected`, for several AllDifferent constraints at once.
    fn expected_all(domains: &[Vec<isize>], scopes: &[&[usize]]) -> Vec<Vec<isize>> {
        let mut tuples: Vec<Vec<isize>> = vec![vec![]];
        for domain in domains {
            let mut next = vec![];
            for prefix in &tuples {
                for &value in domain {
                    let mut tuple = prefix.clone();
                    tuple.push(value);
                    next.push(tuple);
                }
            }
            tuples = next;
        }
        tuples.retain(|t| {
            scopes.iter().all(|scope| {
                (0..scope.len()).all(|i| (i + 1..scope.len()).all(|j| t[scope[i]] != t[scope[j]]))
            })
        });
        tuples.sort();
        tuples
    }

    /// The property of a single path that assigns `values` in this order, built the way the
    /// compiler builds it: one in-scope `update` per edge.
    fn path_property(ad: &AllDifferent, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = ad.empty_property();
        for &value in values {
            let mut next = ad.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        property
    }

    /// Merge of several properties, starting from the first.
    fn merged(properties: &[Box<dyn ConstraintProperty>]) -> Box<dyn ConstraintProperty> {
        let mut result = ad_clone(&*properties[0]);
        for other in &properties[1..] {
            result.merge(&**other);
        }
        result
    }

    fn ad_clone(property: &dyn ConstraintProperty) -> Box<dyn ConstraintProperty> {
        Box::new(
            property
                .as_any()
                .downcast_ref::<AllDifferentProperty>()
                .unwrap()
                .clone(),
        )
    }

    /// The values (not the bits) of $A$ and of $S$, sorted.
    fn sets_of(property: &dyn ConstraintProperty) -> (Vec<isize>, Vec<isize>) {
        let property = property
            .as_any()
            .downcast_ref::<AllDifferentProperty>()
            .unwrap();
        let mut all: Vec<isize> = vec![];
        let mut some: Vec<isize> = vec![];
        for (&value, &bit) in property.map.iter() {
            if property.value_all_path.contains(bit) {
                all.push(value);
            }
            if property.value_some_path.contains(bit) {
                some.push(value);
            }
        }
        all.sort();
        some.sort();
        (all, some)
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_agrees_with_the_pairwise_definition() {
        // Every assignment over a domain with a negative and a non-contiguous value.
        let values = [-1, 0, 5];
        let ad = ad(&[values.to_vec(), values.to_vec(), values.to_vec()]);
        for a in values {
            for b in values {
                for c in values {
                    let want = a != b && a != c && b != c;
                    assert_eq!(ad.is_satisfied(&[a, b, c]), want, "{:?}", [a, b, c]);
                }
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        let ad = ad(&[dom(3), dom(3), dom(3)]);
        assert!(ad.is_satisfied(&[0, 1, 2]));
        assert!(ad.is_satisfied(&[2, 0, 1]));
        assert!(!ad.is_satisfied(&[0, 1, 0]));
        assert!(!ad.is_satisfied(&[1, 1, 2]));
        assert!(!ad.is_satisfied(&[2, 2, 2]));
    }

    #[test]
    fn is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope; variable 1 can repeat any value.
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2)], &[0, 2]);
        let ad = AllDifferent::new(vec![vars[0], vars[2]], &problem);
        assert!(ad.is_satisfied(&[0, 0, 1]));
        assert!(ad.is_satisfied(&[1, 1, 0]));
        assert!(!ad.is_satisfied(&[1, 0, 1]));
    }

    #[test]
    fn empty_and_single_scopes_are_always_satisfied() {
        let problem = Problem::default();
        assert!(AllDifferent::new(vec![], &problem).is_satisfied(&[]));
        let ad = ad(&[dom(3)]);
        assert!(ad.is_satisfied(&[0]));
        assert!(ad.is_satisfied(&[2]));
    }

    #[test]
    fn a_repeated_variable_is_never_satisfied() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        let ad = AllDifferent::new(vec![x, y, x], &problem);
        for a in 0..3 {
            for b in 0..3 {
                assert!(!ad.is_satisfied(&[a, b]));
            }
        }
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        ad(&[dom(2), dom(2), dom(2)]).is_satisfied(&[0, 1]);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_name_and_structural_key() {
        let (problem, vars) = full(&[dom(2), dom(2), dom(2)]);
        let ad = AllDifferent::new(vars.clone(), &problem);
        assert_eq!(ad.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(ad.name(), "AllDifferent");
        assert_eq!(
            ad.structural_key(&problem),
            ConstraintShapeKey::AllDifferent { arity: 3 }
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2), dom(2)], &[0, 2, 3]);
        let mut ad = AllDifferent::new(vec![vars[0], vars[2], vars[3]], &problem);
        // Layer order: v3, v1, v0, v2.
        ad.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[2]]);
        assert!(ad.is_layer_in_scope(0));
        assert!(!ad.is_layer_in_scope(1));
        assert!(ad.is_layer_in_scope(2));
        assert!(ad.is_layer_in_scope(3));
    }

    #[test]
    fn hall_set_bounds_count_scope_variables_above_and_below() {
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2), dom(2)], &[0, 2, 3]);
        let mut ad = AllDifferent::new(vec![vars[0], vars[2], vars[3]], &problem);
        ad.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[2]]);
        // Scope layers are 0, 2 and 3.
        assert_eq!(ad.hall_set_bounds[0], (0, 2));
        assert_eq!(ad.hall_set_bounds[2], (1, 1));
        assert_eq!(ad.hall_set_bounds[3], (2, 0));
    }

    #[test]
    fn layers_beyond_the_first_word_are_tracked() {
        // 130 variables, the scope is at layers 0, 63, 64, 65 and 129: it crosses the 64 and the
        // 128 bit boundaries of the layer bitset.
        let mut problem = Problem::default();
        let vars = problem.add_variables(130, dom(2), None);
        let scope_layers = [0usize, 63, 64, 65, 129];
        let scope: Vec<VariableIndex> = scope_layers.iter().map(|&l| vars[l]).collect();
        let mut ad = AllDifferent::new(scope, &problem);
        ad.update_variable_ordering(&vars);
        for layer in 0..130 {
            assert_eq!(
                ad.is_layer_in_scope(layer),
                scope_layers.contains(&layer),
                "layer {layer}"
            );
        }
        assert_eq!(ad.hall_set_bounds[64], (2, 2));
        assert_eq!(ad.hall_set_bounds[129], (4, 0));
    }

    #[test]
    fn a_new_ordering_replaces_the_previous_one() {
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2)], &[0, 1]);
        let mut ad = AllDifferent::new(vec![vars[0], vars[1]], &problem);
        ad.update_variable_ordering(&[vars[0], vars[1], vars[2]]);
        assert!(ad.is_layer_in_scope(0) && ad.is_layer_in_scope(1) && !ad.is_layer_in_scope(2));
        ad.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        assert!(!ad.is_layer_in_scope(0) && ad.is_layer_in_scope(1) && ad.is_layer_in_scope(2));
        assert_eq!(ad.hall_set_bounds[1], (0, 1));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let (problem, vars) = full(&[dom(2), dom(2)]);
        AllDifferent::new(vars, &problem).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let (problem, vars) = full(&[dom(2), dom(2)]);
        let mut ad = AllDifferent::new(vars.clone(), &problem);
        // `vars[1]` does not appear in the order.
        ad.update_variable_ordering(&[vars[0]]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn empty_property_has_both_sets_empty() {
        let ad = ad(&[dom(3), dom(3)]);
        let empty = ad.empty_property();
        assert_eq!(sets_of(&*empty), (vec![], vec![]));
        assert_eq!(empty.order_key(), vec![0.0, 0.0]);
        assert!(same(&*empty, &*ad.empty_property()));
    }

    #[test]
    fn identity_property_is_not_the_empty_property() {
        // A is full in the identity, so that the first intersection keeps the other operand.
        let ad = ad(&[dom(3), dom(3)]);
        assert!(!same(&*ad.identity_property(), &*ad.empty_property()));
        assert_eq!(sets_of(&*ad.identity_property()).1, Vec::<isize>::new());
        assert_eq!(sets_of(&*ad.identity_property()).0, vec![0, 1, 2]);
    }

    #[test]
    fn update_in_scope_from_the_empty_property_gives_the_value() {
        let ad = ad(&[dom(3), dom(3)]);
        let property = path_property(&ad, &[1]);
        assert_eq!(sets_of(&*property), (vec![1], vec![1]));
    }

    #[test]
    fn a_path_accumulates_its_values() {
        let ad = ad(&[dom(4), dom(4), dom(4)]);
        let property = path_property(&ad, &[2, 0, 3]);
        assert_eq!(sets_of(&*property), (vec![0, 2, 3], vec![0, 2, 3]));
        assert_eq!(property.order_key(), vec![3.0, 3.0]);
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let ad = ad(&[dom(4), dom(4)]);
        let parent = path_property(&ad, &[1, 3]);
        let mut child = ad.identity_property();
        // The value 99 is not in any domain: it must not even be looked up.
        child.update(&*parent, 99, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn folding_two_parents_intersects_a_and_unions_s() {
        let ad = ad(&[dom(4), dom(4), dom(4)]);
        let left = path_property(&ad, &[0, 1]);
        let right = path_property(&ad, &[0, 2]);
        let mut node = ad.identity_property();
        node.update(&*left, 3, true);
        node.update(&*right, 3, true);
        // 0 and 3 are on both paths; 1 and 2 on one of them each.
        assert_eq!(sets_of(&*node), (vec![0, 3], vec![0, 1, 2, 3]));
    }

    #[test]
    fn folding_does_not_depend_on_the_order_of_the_parents() {
        let ad = ad(&[dom(4), dom(4), dom(4)]);
        let left = path_property(&ad, &[0, 1]);
        let right = path_property(&ad, &[2]);
        let mut ab = ad.identity_property();
        ab.update(&*left, 3, true);
        ab.update(&*right, 3, true);
        let mut ba = ad.identity_property();
        ba.update(&*right, 3, true);
        ba.update(&*left, 3, true);
        assert!(same(&*ab, &*ba));
    }

    #[test]
    fn merge_intersects_a_and_unions_s() {
        let ad = ad(&[dom(5), dom(5)]);
        let a = path_property(&ad, &[0, 1]);
        let b = path_property(&ad, &[1, 4]);
        let m = merged(&[a, b]);
        assert_eq!(sets_of(&*m), (vec![1], vec![0, 1, 4]));
        assert_eq!(m.order_key(), vec![1.0, 3.0]);
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let ad = ad(&[dom(5), dom(5)]);
        let a = path_property(&ad, &[0, 1]);
        let b = path_property(&ad, &[1, 4]);

        let mut twice = ad_clone(&*a);
        twice.merge(&*a);
        assert!(same(&*twice, &*a));

        let mut ab = ad_clone(&*a);
        ab.merge(&*b);
        let mut ba = ad_clone(&*b);
        ba.merge(&*a);
        assert!(same(&*ab, &*ba));

        let mut with_identity = ad_clone(&*a);
        with_identity.merge(&*ad.identity_property());
        assert!(same(&*with_identity, &*a));
    }

    #[test]
    fn eq_and_hash_compare_both_sets() {
        let ad = ad(&[dom(3), dom(3)]);
        let p01 = path_property(&ad, &[0, 1]);
        let p10 = path_property(&ad, &[1, 0]);
        // Same set of values in a different order: the same property, the same hash.
        assert!(same(&*p01, &*p10));
        assert_eq!(hash_of(&*p01), hash_of(&*p10));

        // Same S = {0, 1}, different A.
        let a0 = merged(&[path_property(&ad, &[0]), path_property(&ad, &[0, 1])]);
        let a1 = merged(&[path_property(&ad, &[1]), path_property(&ad, &[0, 1])]);
        assert_eq!(sets_of(&*a0).1, sets_of(&*a1).1);
        assert!(!same(&*a0, &*a1));
        assert_ne!(hash_of(&*a0), hash_of(&*a1));

        // Same A = {}, different S.
        let s01 = merged(&[path_property(&ad, &[0]), path_property(&ad, &[1])]);
        let s02 = merged(&[path_property(&ad, &[0]), path_property(&ad, &[2])]);
        assert_eq!(sets_of(&*s01).0, sets_of(&*s02).0);
        assert!(!same(&*s01, &*s02));
        assert_ne!(hash_of(&*s01), hash_of(&*s02));
    }

    #[test]
    fn a_domain_wider_than_one_word_works() {
        // 70 values need more than the 64 bits of the inline bitset.
        let ad = ad(&[dom(70), dom(70)]);
        let low_high = path_property(&ad, &[3, 69]);
        assert_eq!(sets_of(&*low_high), (vec![3, 69], vec![3, 69]));
        let other = path_property(&ad, &[3, 64]);
        let m = merged(&[low_high, other]);
        assert_eq!(sets_of(&*m), (vec![3], vec![3, 64, 69]));
        assert_eq!(m.order_key(), vec![1.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        let ad = ad(&[dom(2), dom(2)]);
        let (problem, vars) = full(&[dom(2), dom(2)]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        ad.identity_property().update(&*foreign, 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        let ad = ad(&[dom(2), dom(2)]);
        let (problem, vars) = full(&[dom(2), dom(2)]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        ad.identity_property().merge(&*foreign);
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    /// Three variables over 0..4, ordered 0, 1, 2. The probed layer is 1 (one variable above,
    /// one below), so the Hall capacity is 2.
    fn probe() -> AllDifferent {
        ad(&[dom(4), dom(4), dom(4)])
    }

    #[test]
    fn a_value_on_every_path_above_is_invalid() {
        let ad = probe();
        let parent = path_property(&ad, &[2]);
        let child = ad.empty_property();
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 2));
        // |S| = 1 is below the capacity 2: nothing else is filtered.
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 3));
    }

    #[test]
    fn a_value_on_every_path_below_is_invalid() {
        let ad = probe();
        let parent = ad.empty_property();
        let child = path_property(&ad, &[1]);
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 1));
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn a_value_on_some_paths_only_is_valid_below_the_capacity() {
        let ad = probe();
        // A = {}, S = {0}: |S| = 1 < 2.
        let parent = merged(&[path_property(&ad, &[0]), ad.empty_property()]);
        let child = ad.empty_property();
        for value in 0..4 {
            assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, value));
        }
    }

    #[test]
    fn the_hall_set_filters_the_values_that_fill_the_capacity() {
        let ad = probe();
        // A = {}, S = {0, 1} = capacity 2: both values are needed by the other variables.
        let parent = merged(&[path_property(&ad, &[0]), path_property(&ad, &[1])]);
        let child = ad.empty_property();
        assert_eq!(sets_of(&*parent), (vec![], vec![0, 1]));
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 0));
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 1));
        // Values outside the Hall set stay available.
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 2));
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 3));
    }

    #[test]
    fn the_hall_set_adds_up_the_parent_and_the_child() {
        let ad = probe();
        // S_parent = {0}, S_child = {2}: the union has the capacity 2 although neither side does.
        let parent = merged(&[path_property(&ad, &[0]), ad.empty_property()]);
        let child = merged(&[path_property(&ad, &[2]), ad.empty_property()]);
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 0));
        assert!(ad.is_assignment_invalid(&*parent, &*child, 1, 2));
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 1));
    }

    #[test]
    fn the_hall_set_counts_a_shared_value_once() {
        let ad = probe();
        // The same value on both sides: |S_parent U S_child| = 1, below the capacity.
        let parent = merged(&[path_property(&ad, &[0]), ad.empty_property()]);
        let child = merged(&[path_property(&ad, &[0]), ad.empty_property()]);
        for value in 0..4 {
            assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, value));
        }
    }

    #[test]
    fn the_hall_set_needs_the_value_to_be_in_the_union() {
        // Regression guard on the final `&&`: a full capacity does not filter other values. A
        // union of the right size but a value outside it is valid, and conversely.
        let ad = probe();
        let parent = merged(&[path_property(&ad, &[0]), path_property(&ad, &[1])]);
        let child = ad.empty_property();
        assert!(!ad.is_assignment_invalid(&*parent, &*child, 1, 3));
    }

    #[test]
    fn the_hall_condition_looks_at_either_side_not_both() {
        // Scope v0..v3 decided in that order; we probe v2 (two variables above, one below). The
        // value 1 is only in S of the parent, not in the child, and the capacity 3 is reached.
        let ad = ad(&[dom(3), dom(3), dom(3), dom(3)]);
        let td1 = path_property(&ad, &[0]);
        let mut td2 = ad.identity_property();
        td2.update(&*td1, 1, true); // constituent A: v1 = 1
        td2.update(&*td1, 2, true); // constituent B: v1 = 2
        let bu3 = path_property(&ad, &[2]); // v3 = 2
        assert_eq!(sets_of(&*td2), (vec![0], vec![0, 1, 2]));
        assert!(ad.is_assignment_invalid(&*td2, &*bu3, 2, 1));
    }

    #[test]
    fn a_repeated_variable_invalidates_every_edge() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        let mut ad = AllDifferent::new(vec![x, y, x], &problem);
        ad.update_variable_ordering(&[x, y]);
        let empty = ad.empty_property();
        for value in 0..3 {
            assert!(ad.is_assignment_invalid(&*empty, &*empty, 0, value));
        }
    }

    #[test]
    #[should_panic(expected = "instead of AllDifferentProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let ad = probe();
        let (problem, vars) = full(&[dom(2), dom(2)]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        let own = ad.identity_property();
        ad.is_assignment_invalid(&*foreign, &*own, 0, 0);
    }

    // ----------------------------------------------------------------------------------------
    // Compiled MDDs against a brute-force oracle
    // ----------------------------------------------------------------------------------------

    #[test]
    fn exact_mdd_for_small_cases_and_every_ordering() {
        let cases: Vec<Vec<Vec<isize>>> = vec![
            vec![vec![0, 1], vec![0, 1]],
            vec![vec![0], vec![0, 1]],
            vec![vec![0, 1], vec![0, 1], vec![0, 1, 2]],
            vec![vec![0, 1, 2], vec![0, 1], vec![0, 1]],
            vec![vec![0, 1], vec![0, 1, 2], vec![0, 1]],
            vec![vec![-1, 4], vec![4, 7], vec![-1, 7]],
            vec![dom(3), dom(3), dom(3)],
        ];
        for domains in cases {
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            for order in permutations(n) {
                let (problem, _) = full(&domains);
                let mdd = compile(problem, order.clone(), usize::MAX);
                assert_eq!(
                    accepted(&mdd),
                    expected(&domains, &scope),
                    "domains {domains:?} order {order:?}"
                );
            }
        }
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut result = vec![];
        for rest in permutations(n - 1) {
            for position in 0..=rest.len() {
                let mut order = rest.clone();
                order.insert(position, n - 1);
                result.push(order);
            }
        }
        result
    }

    #[test]
    fn the_hall_set_prunes_without_any_search() {
        // The two binary variables use up 0 and 1, so z = 2 on every solution. The initial MDD,
        // before any refinement, already shows it.
        for order in [vec![0, 1, 2], vec![2, 0, 1], vec![0, 2, 1]] {
            let (problem, _) = full(&[vec![0, 1], vec![0, 1], vec![0, 1, 2]]);
            let mdd = compile(problem, order.clone(), 1);
            for solution in accepted(&mdd) {
                assert_eq!(solution[2], 2, "order {order:?}");
            }
        }
    }

    /// Two AllDifferent constraints, over `first` and `second`, on one variable per domain.
    fn two_scopes(domains: &[Vec<isize>], first: &[usize], second: &[usize]) -> Problem {
        let (mut problem, vars) = scoped(domains, first);
        all_different(&mut problem, second.iter().map(|&i| vars[i]).collect());
        problem
    }

    #[test]
    fn a_constraint_ignores_the_layers_of_the_other_one() {
        // Variable 1 is not in the first scope: its layer is out of scope for that constraint.
        let domains = [dom(2), dom(2), dom(2), dom(2)];
        let problem = two_scopes(&domains, &[0, 2], &[1, 3]);
        let mdd = compile(problem, vec![0, 1, 2, 3], usize::MAX);
        assert_eq!(accepted(&mdd), expected_all(&domains, &[&[0, 2], &[1, 3]]));
        assert_eq!(accepted(&mdd).len(), 4);
    }

    #[test]
    fn a_scope_spread_over_the_order_is_exact() {
        let domains = [dom(3), dom(2), dom(3), dom(2), dom(3)];
        for order in [
            vec![0, 1, 2, 3, 4],
            vec![3, 4, 1, 2, 0],
            vec![1, 3, 2, 0, 4],
        ] {
            let problem = two_scopes(&domains, &[0, 2, 4], &[1, 3]);
            let mdd = compile(problem, order.clone(), usize::MAX);
            assert_eq!(
                accepted(&mdd),
                expected_all(&domains, &[&[0, 2, 4], &[1, 3]]),
                "order {order:?}"
            );
        }
    }

    #[test]
    fn too_few_values_is_unsat() {
        let (problem, _) = full(&[dom(2), dom(2), dom(2)]);
        let mdd = compile(problem, vec![0, 1, 2], usize::MAX);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    fn forced_values_propagate_through_the_whole_scope() {
        // Three variables fixed to 0, 1 and 2 leave only 3 for the last one.
        let (mut problem, vars) = full(&[dom(4), dom(4), dom(4), dom(4)]);
        equal(&mut problem, vars[1], 2);
        equal(&mut problem, vars[2], 0);
        let mdd = compile(problem, vec![0, 1, 2, 3], usize::MAX);
        assert_eq!(accepted(&mdd), vec![vec![1, 2, 0, 3], vec![3, 2, 0, 1]]);
    }

    #[test]
    fn a_single_variable_keeps_its_whole_domain() {
        let (problem, _) = full(&[dom(3)]);
        let mdd = compile(problem, vec![0], usize::MAX);
        assert_eq!(accepted(&mdd), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn a_repeated_variable_makes_the_problem_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(2), None);
        let y = problem.add_variable(dom(3), None);
        all_different(&mut problem, vec![x, x, y]);
        let mdd = compile(problem, vec![0, 1], usize::MAX);
        assert!(mdd.is_unsat());
    }

    #[test]
    fn a_wide_domain_is_exact() {
        // 70 values in the joint domain: the bitsets use two words.
        let domains = [vec![0, 64, 69], vec![64, 69], vec![0, 64, 69]];
        let (problem, _) = full(&domains);
        let mdd = compile(problem, vec![0, 1, 2], usize::MAX);
        assert_eq!(accepted(&mdd), expected(&domains, &[0, 1, 2]));
    }

    /// The tiny deterministic generator used for the random instances below.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
    }

    #[test]
    fn relaxed_mdds_never_lose_a_solution() {
        // For random instances and several width budgets, every solution of the oracle must be
        // accepted: a relaxation may accept more, never less.
        let mut rng = Lcg(12345);
        for _ in 0..200 {
            let n = 2 + rng.below(4) as usize;
            let domains: Vec<Vec<isize>> = (0..n)
                .map(|_| {
                    let mut d: Vec<isize> = (0..5).filter(|_| rng.below(2) == 1).collect();
                    if d.is_empty() {
                        d.push(rng.below(5) as isize);
                    }
                    d
                })
                .collect();
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &scope);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = full(&domains);
                let mdd = compile(problem, order.clone(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {domains:?} order {order:?} width {width}"
                    );
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Domains that change after construction
    // ----------------------------------------------------------------------------------------

    #[test]
    fn a_domain_that_shrinks_after_construction_is_fine() {
        let (mut problem, vars) = full(&[dom(3), dom(3)]);
        problem[vars[0]].set_domain(vec![1]);
        let mdd = compile(problem, vec![0, 1], usize::MAX);
        assert_eq!(accepted(&mdd), vec![vec![1, 0], vec![1, 2]]);
    }

    #[test]
    #[should_panic]
    fn a_value_added_after_construction_panics() {
        // Documented limitation: the value 9 has no bit.
        let (mut problem, vars) = full(&[dom(2), dom(2)]);
        problem[vars[0]].set_domain(vec![0, 1, 9]);
        let mdd = compile(problem, vec![0, 1], usize::MAX);
        drop(mdd);
    }
}
