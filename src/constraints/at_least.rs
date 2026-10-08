//! `AtLeast`: at least `lb` variables of the scope take a value of a given set.
//!
//! # Meaning
//!
//! Given a set of values $V$ and a bound $lb$, constrains $x_1, \dots, x_n$ so that
//!
//! $$|\{ i \mid x_i \in V \}| \geq lb.$$
//!
//! Values of $V$ that are in no domain are never counted. With $lb = 0$ the constraint always
//! holds, and with $lb > n$ it has no solution. An empty scope counts 0, so it is satisfied
//! exactly when $lb = 0$; as for [`Sum`] and [`Among`], the MDD has no layer at which to filter
//! anything, so an empty `AtLeast` is not seen by the compilation.
//!
//! This is [`Among`] without an upper bound: the usual "at least $K$ of these" cover (for example
//! at least $K$ nurses on a shift, with no ceiling). It is modelled on its own, instead of as an
//! `Among` whose upper bound is pinned to $n$, so that the compiled state can stop growing at
//! $lb$ (see below).
//!
//! A variable may appear only once in the scope: the compiled property cannot know how many times
//! a layer is counted, so [`AtLeast::new`] panics on a repeated variable instead of compiling
//! something different from [`is_satisfied`](Constraint::is_satisfied).
//!
//! # How it compiles
//!
//! A node carries the interval $[m, M]$ of the number of scope variables that take a value of $V$
//! on the paths that reach it (top-down) or leave it (bottom-up), **capped at $lb$**: $m$ is the
//! smallest such count and $M$ the largest, and both are replaced by $lb$ when they exceed it. A
//! node with no path at all holds the sentinel $[+\infty, 0]$, which is recognised by $m > M$.
//!
//! ## Property update
//!
//! On an edge of the scope with a value in $V$, the interval of the parent (or child) is shifted
//! by 1 and then capped at $lb$; any other edge, and every out of scope layer, copies it. When a
//! node has several parents, the results are combined by taking the smallest $m$ and the largest
//! $M$. A parent with no path is ignored.
//!
//! The cap is an exact reduction of the state space, not a relaxation. There is no upper bound,
//! so once $lb$ variables are counted nothing that happens later can matter: every count from
//! $lb$ up behaves the same for every future decision. With the cap a layer has at most $lb + 1$
//! different counts, whatever the number of variables, where [`Among`] would keep every count up
//! to $n$. ([`Gcc`] caps the counts of the values whose upper bound can never bind for the same
//! reason.)
//!
//! ## Node merging
//!
//! Merging two nodes takes the smallest $m$ and the largest $M$. The merged interval contains the
//! counts of both nodes, so the merged node over-approximates them. Both bounds are at most the
//! cap, so the merge needs no capping of its own.
//!
//! ## Edge filtering
//!
//! An edge with value $v$ between a parent $[m_p, M_p]$ and a child $[m_c, M_c]$ lies on paths
//! whose largest count is $M_p + M_c + \delta$, with $\delta = 1$ if $v \in V$ and 0 otherwise.
//! It is removed when that count is below $lb$. An edge next to a node with no path at all is
//! removed too.
//!
//! Only the largest count matters, because there is no upper bound to exceed. A prefix that
//! reaches the parent with $M_p$ and a suffix that leaves the child with $M_c$ can always be
//! joined through the edge, so the test is exact on the diagram it looks at: unlike [`Among`] or
//! [`Sum`] there is no hole in an interval to be fooled by. The capped values keep the test
//! valid: a capped side already brings the sum to $lb$ or more.
//!
//! # Design notes
//!
//! - **The lower bound $m$ is carried but never read by the filtering.** It is part of the state
//!   that decides whether two nodes are the same (equality, hash and the merge heuristics' key),
//!   so two nodes with the same $M$ and a different $m$ are kept apart although they filter the
//!   same edges. See the task "AtLeast keeps a bound it never uses".
//! - **Only the set $V$ is stored**, in an [`Arc`] shared by the constraint and all its
//!   properties. No domain is read, so domains may change freely after construction.
//! - **The structural key is self-contained**: it holds the arity, the sorted values of $V$ and
//!   $lb$. Two `AtLeast` built with the values in a different order get the same key.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{AtLeast, Constraint};
//! use aicad::modelling::*;
//! use rustc_hash::FxHashSet;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2], None);
//! // At least two variables take a value in {1, 2}.
//! let at_least = AtLeast::new(vars, FxHashSet::from_iter([1, 2]), 2);
//!
//! assert!(at_least.is_satisfied(&[1, 0, 2]));
//! assert!(at_least.is_satisfied(&[2, 2, 2]));
//! assert!(!at_least.is_satisfied(&[1, 0, 0]));
//! ```
use super::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`AtLeast`]: the smallest and the largest number of scope variables that
/// take a value of the set along the paths that reach the node (top-down) or leave it
/// (bottom-up), both capped at `lb`.
///
/// `(usize::MAX, 0)` means that no path has been folded in yet, which is the identity of `merge`;
/// it is recognised by `min > max`. `merge` takes the smallest minimum and the largest maximum, so
/// a merged node over-approximates both parents. See the [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AtLeastProperty {
    /// The set $V$ of counted values.
    values: Arc<FxHashSet<isize>>,
    /// The bound at which the counts are capped.
    lb: usize,
    min: usize,
    max: usize,
}

impl AtLeastProperty {
    fn new(values: Arc<FxHashSet<isize>>, lb: usize, min: usize, max: usize) -> Self {
        Self {
            values,
            lb,
            min,
            max,
        }
    }

    /// True if no path has been folded into this property.
    fn has_no_path(&self) -> bool {
        self.min > self.max
    }
}

/// The constraint that at least `lb` variables take a value in a set. See the
/// [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AtLeast {
    /// Scope of the constraint, without repetition.
    variables: Vec<VariableIndex>,
    /// The set $V$ of counted values.
    values: Arc<FxHashSet<isize>>,
    lb: usize,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
}

impl AtLeast {
    /// Builds the constraint that at least `lb` of `variables` take a value in `values`.
    ///
    /// # Panics
    ///
    /// If a variable appears more than once in `variables`.
    pub fn new(variables: Vec<VariableIndex>, values: FxHashSet<isize>, lb: usize) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        assert!(
            distinct.len() == variables.len(),
            "AtLeast does not support a variable repeated in its scope"
        );
        Self {
            variables,
            values: Arc::new(values),
            lb,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for AtLeast {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let mut values: Vec<isize> = self.values.iter().copied().collect();
        values.sort_unstable();
        ConstraintShapeKey::AtLeast {
            arity: self.variables.len(),
            values,
            lb: self.lb,
        }
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        let mut found = 0;
        self.layer_in_scope = vec![0; order.len() / 64 + 1];
        for (layer, variable) in order.iter().enumerate() {
            if scope.contains(variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                found += 1;
            }
        }
        debug_assert_eq!(
            found,
            scope.len(),
            "a variable of the scope is missing from the ordering"
        );
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(
            !self.layer_in_scope.is_empty(),
            "update_variable_ordering has not been called"
        );
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        _layer: usize,
        assignment: isize,
    ) -> bool {
        let parent = parent.as_any().downcast_ref::<AtLeastProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of AtLeastProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<AtLeastProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of AtLeastProperty",
                    child.name()
                );
        });
        // A node without any path cannot be crossed by an edge. Checking it also keeps a bogus
        // count out of the sum below.
        if parent.has_no_path() || child.has_no_path() {
            return true;
        }

        // No upper bound exists, so the edge can only be pruned if even the best completion
        // (the largest count reaching the parent, the largest count leaving the child, and this
        // edge's own contribution) cannot reach `lb`.
        let delta = usize::from(self.values.contains(&assignment));
        parent.max + child.max + delta < self.lb
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover the scope, otherwise this
        // panics on the out-of-range index.
        let count = self
            .variables
            .iter()
            .filter(|variable| self.values.contains(&assignment[variable.0]))
            .count();
        count >= self.lb
    }

    fn name(&self) -> &'static str {
        "AtLeast"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AtLeastProperty::new(
            self.values.clone(),
            self.lb,
            usize::MAX,
            0,
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AtLeastProperty::new(self.values.clone(), self.lb, 0, 0))
    }
}

impl ConstraintProperty for AtLeastProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        // A parent without paths adds none: shifting its sentinels would overflow `min` and
        // would turn the sentinel `max` of 0 into a real count.
        if other.has_no_path() {
            return;
        }

        let delta = usize::from(in_scope && self.values.contains(&assignment));
        // Cap at `lb`: once a count is already >= lb, every further increment is provably
        // interchangeable for every future decision (there is no upper bound left to violate).
        self.min = self.min.min(other.min + delta).min(self.lb);
        self.max = self.max.max(other.max + delta).min(self.lb);
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        // Both operands are already capped at `lb` by `update`, so a plain min/max fold
        // preserves that invariant without needing to re-cap here.
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn order_key(&self) -> Vec<f64> {
        vec![self.min as f64, self.max as f64]
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.min);
        hasher.write_usize(self.max);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling eq on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        self.min == other.min && self.max == other.max
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "AtLeastProperty"
    }
}

#[cfg(test)]
mod test_at_least {
    use super::AtLeastProperty;
    use crate::constraints::{
        AllDifferent, AtLeast, Constraint, ConstraintProperty, ConstraintShapeKey,
    };
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::{FxHashSet, FxHasher};
    use std::hash::Hasher;
    use std::sync::Arc;

    // ----------------------------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------------------------

    fn set(values: &[isize]) -> FxHashSet<isize> {
        FxHashSet::from_iter(values.iter().copied())
    }

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    /// One variable per domain, and an AtLeast over the variables listed in `scope`.
    fn scoped(
        domains: &[Vec<isize>],
        scope: &[usize],
        values: &[isize],
        lb: usize,
    ) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        at_least(
            &mut problem,
            scope.iter().map(|&i| vars[i]).collect(),
            values.to_vec(),
            lb,
        );
        (problem, vars)
    }

    /// One variable per domain, all in the scope.
    fn full(domains: &[Vec<isize>], values: &[isize], lb: usize) -> (Problem, Vec<VariableIndex>) {
        scoped(domains, &(0..domains.len()).collect::<Vec<_>>(), values, lb)
    }

    /// The constraint over variables `0..n` with the identity ordering already set.
    fn at_least_of(n: usize, values: &[isize], lb: usize) -> AtLeast {
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(3), None);
        let mut constraint = AtLeast::new(vars.clone(), set(values), lb);
        constraint.update_variable_ordering(&vars);
        constraint
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

    /// Like `compile` with `usize::MAX`, then runs the propagation again until nothing changes.
    ///
    /// `Mdd::refine` propagates only one pass after its last split, so an MDD can keep paths that
    /// a second pass would remove (see the task "MDD propagation is not iterated to a fixpoint").
    /// The exactness tests need the fixpoint, because they check what the constraint rules out,
    /// not how many passes the engine runs.
    fn settled(problem: Problem, order: Vec<usize>) -> Mdd {
        let mut mdd = compile(problem, order, usize::MAX);
        for _ in 0..10 {
            if mdd.is_unsat() {
                break;
            }
            mdd.propagate_constraints();
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

    /// One test case: domains, counted values, lower bound.
    type Case = (Vec<Vec<isize>>, Vec<isize>, usize);

    /// One AtLeast of the oracle: scope positions, counted values, lower bound.
    type Spec<'a> = (&'a [usize], &'a [isize], usize);

    /// The expected solutions, written independently of `AtLeast`: the cartesian product of the
    /// domains, keeping the tuples that satisfy every spec.
    fn expected(domains: &[Vec<isize>], specs: &[Spec]) -> Vec<Vec<isize>> {
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
            specs.iter().all(|(scope, values, lb)| {
                let mut count = 0;
                for &position in scope.iter() {
                    if values.contains(&t[position]) {
                        count += 1;
                    }
                }
                count >= *lb
            })
        });
        tuples.sort();
        tuples
    }

    /// The property of a single path that assigns `values`, built the way the compiler builds
    /// it: one in-scope `update` per edge.
    fn path_property(constraint: &AtLeast, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = constraint.empty_property();
        for &value in values {
            let mut next = constraint.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        property
    }

    /// A property whose interval is exactly `[lo, hi]`, for a constraint counting the value 1.
    /// `constraint.lb` must be at least `hi`, otherwise the counts are capped.
    fn interval(constraint: &AtLeast, lo: usize, hi: usize) -> Box<dyn ConstraintProperty> {
        let mut property = path_property(constraint, &vec![1; lo]);
        property.merge(&*path_property(constraint, &vec![1; hi]));
        property
    }

    fn clone_of(property: &dyn ConstraintProperty) -> Box<dyn ConstraintProperty> {
        Box::new(
            property
                .as_any()
                .downcast_ref::<AtLeastProperty>()
                .unwrap()
                .clone(),
        )
    }

    /// The `(min, max)` of a property.
    fn bounds(property: &dyn ConstraintProperty) -> (usize, usize) {
        let property = property.as_any().downcast_ref::<AtLeastProperty>().unwrap();
        (property.min, property.max)
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    fn foreign() -> Box<dyn ConstraintProperty> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        AllDifferent::new(vars, &problem).identity_property()
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_agrees_with_the_definition() {
        // Every assignment over {0, 1, 2}, for every bound, counting the values {1, 2}.
        for lb in 0..=4 {
            let constraint = at_least_of(3, &[1, 2], lb);
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let count = [a, b, c].iter().filter(|&&v| v >= 1).count();
                        assert_eq!(
                            constraint.is_satisfied(&[a, b, c]),
                            count >= lb,
                            "{:?} with lb {lb}",
                            [a, b, c]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        let constraint = at_least_of(3, &[1, 2], 2);
        // 1 and 2 are in the set, 0 is not: counts 2, 1 and 3.
        assert!(constraint.is_satisfied(&[1, 0, 2]));
        assert!(!constraint.is_satisfied(&[1, 0, 0]));
        assert!(constraint.is_satisfied(&[2, 2, 2]));
        // There is no ceiling: every variable may take a value of the set.
        let at_least_one = at_least_of(2, &[5], 1);
        assert!(at_least_one.is_satisfied(&[5, 0]));
        assert!(at_least_one.is_satisfied(&[5, 5]));
        assert!(!at_least_one.is_satisfied(&[0, 0]));
    }

    #[test]
    fn a_zero_bound_always_holds() {
        let constraint = at_least_of(2, &[1], 0);
        assert!(constraint.is_satisfied(&[0, 0]));
        assert!(constraint.is_satisfied(&[1, 1]));
    }

    #[test]
    fn values_outside_the_domains_are_never_counted() {
        let constraint = at_least_of(2, &[7, 8], 1);
        assert!(!constraint.is_satisfied(&[0, 1]));
    }

    #[test]
    fn is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        let constraint = AtLeast::new(vec![vars[0], vars[2]], set(&[1]), 2);
        assert!(constraint.is_satisfied(&[1, 0, 1]));
        assert!(!constraint.is_satisfied(&[1, 1, 0]));
    }

    #[test]
    fn empty_scope_counts_zero() {
        assert!(AtLeast::new(vec![], set(&[1]), 0).is_satisfied(&[]));
        assert!(!AtLeast::new(vec![], set(&[1]), 1).is_satisfied(&[]));
    }

    #[test]
    fn a_bound_above_the_number_of_variables_is_never_satisfied() {
        let constraint = at_least_of(3, &[1], 4);
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    assert!(!constraint.is_satisfied(&[a, b, c]));
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        at_least_of(3, &[1], 0).is_satisfied(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        AtLeast::new(vec![x, y, x], set(&[1]), 1);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_name_and_structural_key() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(5), None);
        let constraint = AtLeast::new(vars.clone(), set(&[4, 1, 3]), 2);
        assert_eq!(constraint.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(constraint.name(), "AtLeast");
        assert_eq!(
            constraint.structural_key(&problem),
            ConstraintShapeKey::AtLeast {
                arity: 3,
                values: vec![1, 3, 4],
                lb: 2
            }
        );
    }

    #[test]
    fn the_structural_key_does_not_depend_on_the_order_of_the_values() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(5), None);
        let a = AtLeast::new(vars.clone(), set(&[1, 2, 3]), 1);
        let b = AtLeast::new(vars.clone(), set(&[3, 1, 2]), 1);
        let different_bound = AtLeast::new(vars.clone(), set(&[1, 2, 3]), 2);
        let different_values = AtLeast::new(vars, set(&[1, 2]), 1);
        assert_eq!(a.structural_key(&problem), b.structural_key(&problem));
        assert_ne!(
            a.structural_key(&problem),
            different_bound.structural_key(&problem)
        );
        assert_ne!(
            a.structural_key(&problem),
            different_values.structural_key(&problem)
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, dom(2), None);
        let mut constraint = AtLeast::new(vec![vars[0], vars[2], vars[3]], set(&[1]), 1);
        // Layer order: v3, v1, v0, v2.
        constraint.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[2]]);
        assert!(constraint.is_layer_in_scope(0));
        assert!(!constraint.is_layer_in_scope(1));
        assert!(constraint.is_layer_in_scope(2));
        assert!(constraint.is_layer_in_scope(3));
    }

    #[test]
    fn layers_beyond_the_first_word_are_tracked() {
        // 130 variables, the scope is at layers 0, 63, 64, 65 and 129.
        let mut problem = Problem::default();
        let vars = problem.add_variables(130, dom(2), None);
        let scope_layers = [0usize, 63, 64, 65, 129];
        let scope: Vec<VariableIndex> = scope_layers.iter().map(|&l| vars[l]).collect();
        let mut constraint = AtLeast::new(scope, set(&[1]), 1);
        constraint.update_variable_ordering(&vars);
        for layer in 0..130 {
            assert_eq!(
                constraint.is_layer_in_scope(layer),
                scope_layers.contains(&layer),
                "layer {layer}"
            );
        }
    }

    #[test]
    fn a_new_ordering_replaces_the_previous_one() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        let mut constraint = AtLeast::new(vec![vars[0], vars[1]], set(&[1]), 1);
        constraint.update_variable_ordering(&[vars[0], vars[1], vars[2]]);
        assert!(
            constraint.is_layer_in_scope(0)
                && constraint.is_layer_in_scope(1)
                && !constraint.is_layer_in_scope(2)
        );
        constraint.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        assert!(
            !constraint.is_layer_in_scope(0)
                && constraint.is_layer_in_scope(1)
                && constraint.is_layer_in_scope(2)
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        AtLeast::new(vars, set(&[1]), 1).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        let mut constraint = AtLeast::new(vars.clone(), set(&[1]), 1);
        // `vars[1]` does not appear in the order.
        constraint.update_variable_ordering(&[vars[0]]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn empty_property_is_the_zero_interval() {
        let constraint = at_least_of(2, &[1], 2);
        assert_eq!(bounds(&*constraint.empty_property()), (0, 0));
        assert_eq!(constraint.empty_property().order_key(), vec![0.0, 0.0]);
    }

    #[test]
    fn identity_property_holds_the_no_path_sentinels() {
        let constraint = at_least_of(2, &[1], 2);
        assert_eq!(bounds(&*constraint.identity_property()), (usize::MAX, 0));
        assert!(!same(
            &*constraint.identity_property(),
            &*constraint.empty_property()
        ));
    }

    #[test]
    fn update_counts_only_the_values_of_the_set() {
        let constraint = at_least_of(3, &[1, 2], 3);
        assert_eq!(bounds(&*path_property(&constraint, &[0])), (0, 0));
        assert_eq!(bounds(&*path_property(&constraint, &[1])), (1, 1));
        assert_eq!(bounds(&*path_property(&constraint, &[2, 0, 1])), (2, 2));
        assert_eq!(bounds(&*path_property(&constraint, &[0, 0, 0])), (0, 0));
    }

    #[test]
    fn update_caps_the_counts_at_the_bound() {
        let constraint = at_least_of(5, &[1], 2);
        assert_eq!(bounds(&*path_property(&constraint, &[1])), (1, 1));
        assert_eq!(bounds(&*path_property(&constraint, &[1, 1])), (2, 2));
        // From the third counted value on, the state no longer changes.
        assert_eq!(bounds(&*path_property(&constraint, &[1, 1, 1])), (2, 2));
        assert_eq!(bounds(&*path_property(&constraint, &[1; 5])), (2, 2));
        // A value out of the set does not move a capped count either.
        assert_eq!(bounds(&*path_property(&constraint, &[1, 1, 0])), (2, 2));
    }

    #[test]
    fn a_zero_bound_keeps_every_count_at_zero() {
        let constraint = at_least_of(3, &[1], 0);
        assert_eq!(bounds(&*path_property(&constraint, &[1, 1, 1])), (0, 0));
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let constraint = at_least_of(2, &[1], 5);
        let parent = interval(&constraint, 1, 2);
        let mut child = constraint.identity_property();
        // The value 1 is in the set but the layer is out of scope: nothing is counted.
        child.update(&*parent, 1, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn folding_two_parents_takes_the_extreme_counts() {
        let constraint = at_least_of(3, &[1], 10);
        let mut node = constraint.identity_property();
        node.update(&*interval(&constraint, 0, 1), 1, true);
        node.update(&*interval(&constraint, 2, 3), 1, true);
        assert_eq!(bounds(&*node), (1, 4));
    }

    #[test]
    fn folding_applies_the_cap_to_the_result() {
        // Bound 2: [0, 1] + 1 = [1, 2] and [2, 2] + 1 = [3, 3], capped to [2, 2].
        let constraint = at_least_of(3, &[1], 2);
        let mut node = constraint.identity_property();
        node.update(&*interval(&constraint, 0, 1), 1, true);
        node.update(&*interval(&constraint, 2, 2), 1, true);
        assert_eq!(bounds(&*node), (1, 2));
    }

    #[test]
    fn folding_does_not_depend_on_the_order_of_the_parents() {
        let constraint = at_least_of(3, &[1], 10);
        let left = interval(&constraint, 0, 1);
        let right = interval(&constraint, 2, 3);
        let mut lr = constraint.identity_property();
        lr.update(&*left, 1, true);
        lr.update(&*right, 1, true);
        let mut rl = constraint.identity_property();
        rl.update(&*right, 1, true);
        rl.update(&*left, 1, true);
        assert!(same(&*lr, &*rl));
    }

    #[test]
    fn a_parent_without_paths_stays_without_paths() {
        // Folding the identity must neither overflow nor turn the sentinel maximum 0 into a count.
        let constraint = at_least_of(2, &[1], 2);
        let mut node = constraint.identity_property();
        node.update(&*constraint.identity_property(), 1, true);
        assert!(same(&*node, &*constraint.identity_property()));
        // And it does not disturb a real parent folded in afterwards.
        node.update(&*path_property(&constraint, &[0]), 1, true);
        assert_eq!(bounds(&*node), (1, 1));
    }

    #[test]
    fn merge_takes_the_smallest_min_and_the_largest_max() {
        let constraint = at_least_of(3, &[1], 10);
        let mut a = interval(&constraint, 1, 2);
        a.merge(&*interval(&constraint, 0, 1));
        assert_eq!(bounds(&*a), (0, 2));
        assert_eq!(a.order_key(), vec![0.0, 2.0]);
        let mut disjoint = interval(&constraint, 0, 0);
        disjoint.merge(&*interval(&constraint, 3, 3));
        assert_eq!(bounds(&*disjoint), (0, 3));
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let constraint = at_least_of(3, &[1], 10);
        let a = interval(&constraint, 1, 2);
        let b = interval(&constraint, 0, 3);

        let mut twice = clone_of(&*a);
        twice.merge(&*a);
        assert!(same(&*twice, &*a));

        let mut ab = clone_of(&*a);
        ab.merge(&*b);
        let mut ba = clone_of(&*b);
        ba.merge(&*a);
        assert!(same(&*ab, &*ba));

        let mut with_identity = clone_of(&*a);
        with_identity.merge(&*constraint.identity_property());
        assert!(same(&*with_identity, &*a));
    }

    #[test]
    fn merging_capped_nodes_stays_below_the_bound() {
        let constraint = at_least_of(4, &[1], 2);
        let mut node = path_property(&constraint, &[1, 1, 1]);
        node.merge(&*path_property(&constraint, &[0]));
        assert_eq!(bounds(&*node), (0, 2));
    }

    #[test]
    fn eq_and_hash_compare_both_bounds() {
        let constraint = at_least_of(3, &[1], 10);
        let a = interval(&constraint, 1, 2);
        assert!(same(&*a, &*interval(&constraint, 1, 2)));
        assert_eq!(hash_of(&*a), hash_of(&*interval(&constraint, 1, 2)));
        assert!(!same(&*a, &*interval(&constraint, 1, 3)));
        assert!(!same(&*a, &*interval(&constraint, 0, 2)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&constraint, 1, 3)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&constraint, 0, 2)));
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        let constraint = at_least_of(2, &[1], 1);
        constraint.identity_property().update(&*foreign(), 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        let constraint = at_least_of(2, &[1], 1);
        constraint.identity_property().merge(&*foreign());
    }

    #[test]
    #[should_panic(expected = "Calling eq on property")]
    fn eq_with_a_property_of_another_constraint_is_rejected() {
        let constraint = at_least_of(2, &[1], 1);
        same(&*constraint.identity_property(), &*foreign());
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_bound_is_inclusive() {
        // Parent [1, 2], child [0, 1]: the largest count is 4 with an edge in the set, 3 with an
        // edge out of the set. The intervals are built with a large bound so they are not capped.
        let helper = at_least_of(4, &[1], 10);
        let parent = interval(&helper, 1, 2);
        let child = interval(&helper, 0, 1);
        // (lb, edge value, invalid?)
        let cases = [
            (5, 1, true),  // max 4 < lb 5
            (4, 1, false), // max 4 == lb 4
            (4, 0, true),  // max 3 < lb 4
            (3, 0, false), // max 3 == lb 3
            (0, 0, false),
        ];
        for (lb, value, invalid) in cases {
            let constraint = at_least_of(4, &[1], lb);
            assert_eq!(
                constraint.is_assignment_invalid(&*parent, &*child, 1, value),
                invalid,
                "lb {lb} edge value {value}"
            );
        }
    }

    #[test]
    fn only_edges_in_the_set_add_to_the_count() {
        // One path with count 1, bound 2: an edge in the set reaches 2, an edge out of the set
        // stays at 1.
        let constraint = at_least_of(2, &[1], 2);
        let parent = path_property(&constraint, &[1]);
        let child = constraint.empty_property();
        assert!(!constraint.is_assignment_invalid(&*parent, &*child, 1, 1));
        assert!(constraint.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn the_parent_and_the_child_both_count() {
        let constraint = at_least_of(3, &[1], 2);
        let one = path_property(&constraint, &[1]);
        let none = constraint.empty_property();
        // Parent only: 1 + 1 = 2 with an edge in the set. Child only: the same.
        assert!(!constraint.is_assignment_invalid(&*one, &*none, 1, 1));
        assert!(!constraint.is_assignment_invalid(&*none, &*one, 1, 1));
        // Both, with an edge out of the set: 1 + 1 = 2.
        assert!(!constraint.is_assignment_invalid(&*one, &*one, 1, 0));
        // Neither: 0 + 0 + 1 = 1 < 2.
        assert!(constraint.is_assignment_invalid(&*none, &*none, 1, 1));
    }

    #[test]
    fn the_smallest_count_plays_no_part() {
        // Only the largest count of a node matters: the same maximum with a different minimum
        // gives the same verdict, for every bound and every edge value.
        let helper = at_least_of(4, &[1], 10);
        let child = helper.empty_property();
        for lb in 0..=4 {
            let constraint = at_least_of(4, &[1], lb);
            for value in 0..2 {
                let verdicts: Vec<bool> = [0, 1, 2]
                    .iter()
                    .map(|&lo| {
                        let parent = interval(&helper, lo, 2);
                        constraint.is_assignment_invalid(&*parent, &*child, 1, value)
                    })
                    .collect();
                assert!(
                    verdicts.windows(2).all(|w| w[0] == w[1]),
                    "lb {lb} value {value}: {verdicts:?}"
                );
            }
        }
    }

    #[test]
    fn a_capped_count_keeps_every_edge() {
        // A node already at the bound can follow any edge, in the set or not.
        let constraint = at_least_of(4, &[1], 2);
        let capped = path_property(&constraint, &[1, 1, 1]);
        let none = constraint.empty_property();
        for value in 0..3 {
            assert!(!constraint.is_assignment_invalid(&*capped, &*none, 1, value));
            assert!(!constraint.is_assignment_invalid(&*none, &*capped, 1, value));
        }
    }

    #[test]
    fn a_zero_bound_never_invalidates_an_edge() {
        let constraint = at_least_of(2, &[1], 0);
        let zero = constraint.empty_property();
        for value in 0..3 {
            assert!(!constraint.is_assignment_invalid(&*zero, &*zero, 1, value));
        }
    }

    #[test]
    fn an_edge_next_to_a_node_without_paths_is_invalid() {
        // Such a node is about to be removed; the sentinels must not be added to anything. With a
        // zero bound a bogus count could not matter, so the bound is set to 1.
        let constraint = at_least_of(2, &[1], 1);
        let identity = constraint.identity_property();
        let real = path_property(&constraint, &[1]);
        assert!(constraint.is_assignment_invalid(&*identity, &*identity, 0, 0));
        assert!(constraint.is_assignment_invalid(&*identity, &*real, 0, 1));
        assert!(constraint.is_assignment_invalid(&*real, &*identity, 0, 1));
    }

    #[test]
    #[should_panic(expected = "instead of AtLeastProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let constraint = at_least_of(2, &[1], 1);
        let own = constraint.empty_property();
        constraint.is_assignment_invalid(&*foreign(), &*own, 0, 0);
    }

    // ----------------------------------------------------------------------------------------
    // Compiled MDDs against a brute-force oracle
    // ----------------------------------------------------------------------------------------

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
    fn exact_mdd_for_small_cases_and_every_ordering() {
        // (domains, counted values, lb)
        let cases: Vec<Case> = vec![
            (vec![dom(2), dom(2)], vec![1], 1),
            (vec![dom(3), dom(3), dom(3)], vec![0, 1], 2),
            (vec![dom(2), dom(2), dom(2)], vec![1], 3),
            (vec![dom(2), dom(2), dom(2)], vec![1], 0),
            (vec![dom(3), dom(3), dom(3), dom(3)], vec![2], 2),
            (vec![vec![-1, 4], vec![4, 7], vec![-1, 7]], vec![4, 7], 2),
            (vec![dom(3), vec![5, 6], dom(3)], vec![5, 0], 2),
        ];
        for (domains, values, lb) in cases {
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            for order in permutations(n) {
                let (problem, _) = full(&domains, &values, lb);
                let mdd = settled(problem, order.clone());
                assert_eq!(
                    accepted(&mdd),
                    expected(&domains, &[(&scope, &values, lb)]),
                    "domains {domains:?} values {values:?} lb {lb} order {order:?}"
                );
            }
        }
    }

    #[test]
    fn unreachable_bounds_are_unsat() {
        // Never in the set; lb above the arity; a set that no domain contains.
        let cases: Vec<Case> = vec![
            (vec![vec![0], vec![0]], vec![1], 1),
            (vec![dom(2), dom(2)], vec![1], 3),
            (vec![dom(2), dom(2)], vec![9], 1),
        ];
        for (domains, values, lb) in cases {
            let (problem, _) = full(&domains, &values, lb);
            let mdd = settled(problem, vec![0, 1]);
            assert!(accepted(&mdd).is_empty(), "{domains:?} {values:?} lb {lb}");
        }
    }

    #[test]
    fn a_zero_bound_accepts_every_tuple() {
        let domains = [dom(2), dom(3), dom(2)];
        for values in [vec![1], vec![9]] {
            let (problem, _) = full(&domains, &values, 0);
            let mdd = settled(problem, vec![0, 1, 2]);
            assert_eq!(accepted(&mdd).len(), 12, "values {values:?}");
        }
    }

    /// Two AtLeast constraints on one variable per domain.
    fn two_scopes(domains: &[Vec<isize>], first: Spec, second: Spec) -> Problem {
        let (mut problem, vars) = scoped(domains, first.0, first.1, first.2);
        at_least(
            &mut problem,
            second.0.iter().map(|&i| vars[i]).collect(),
            second.1.to_vec(),
            second.2,
        );
        problem
    }

    #[test]
    fn a_constraint_ignores_the_layers_of_the_other_one() {
        let domains = [dom(3), dom(3), dom(3), dom(3)];
        let first: Spec = (&[0, 2], &[1], 1);
        let second: Spec = (&[1, 3], &[2], 2);
        for order in [vec![0, 1, 2, 3], vec![3, 1, 2, 0], vec![1, 3, 0, 2]] {
            let problem = two_scopes(&domains, first, second);
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &[first, second]),
                "order {order:?}"
            );
        }
    }

    #[test]
    fn overlapping_scopes_are_exact() {
        let domains = [dom(3), dom(3), dom(3)];
        let first: Spec = (&[0, 1], &[1, 2], 2);
        let second: Spec = (&[1, 2], &[0], 1);
        let problem = two_scopes(&domains, first, second);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want = expected(&domains, &[first, second]);
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn combined_with_not_equals_is_exact() {
        let domains = [dom(3), dom(3), dom(3)];
        let (mut problem, vars) = full(&domains, &[2], 1);
        not_equals(&mut problem, vars[0], vars[1]);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want: Vec<Vec<isize>> = expected(&domains, &[(&[0, 1, 2], &[2], 1)])
            .into_iter()
            .filter(|t| t[0] != t[1])
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn a_layer_holds_at_most_lb_plus_one_nodes() {
        // At least 2 of 6 variables in {0, 1} take the value 1. A layer only needs the counts
        // 0, 1 and "2 or more", and the counts that can no longer reach 2 are removed.
        let domains = vec![dom(2); 6];
        let (problem, _) = full(&domains, &[1], 2);
        let mdd = settled(problem, (0..6).collect());
        let sizes: Vec<usize> = (0..=6).map(|l| mdd.number_nodes_in_layer(l)).collect();
        assert_eq!(sizes, vec![1, 2, 3, 3, 3, 2, 1]);
    }

    #[test]
    fn it_accepts_what_among_without_an_upper_bound_accepts_with_fewer_nodes() {
        let domains = vec![dom(2); 6];
        let (problem, vars) = full(&domains, &[1], 2);
        let compact = settled(problem, (0..6).collect());

        let mut problem = Problem::default();
        let vars_among: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        among(&mut problem, vars_among, vec![1], 2, vars.len());
        let verbose = settled(problem, (0..6).collect());

        assert_eq!(accepted(&compact), accepted(&verbose));
        let nodes = |mdd: &Mdd| -> usize { (0..=6).map(|l| mdd.number_nodes_in_layer(l)).sum() };
        assert!(
            nodes(&compact) < nodes(&verbose),
            "{} nodes against {}",
            nodes(&compact),
            nodes(&verbose)
        );
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

    /// A random instance: domains over {0, ..., 3}, a non-empty set of counted values and a bound
    /// from 0 to `n` (so that both loose and unreachable bounds show up).
    fn random_instance(rng: &mut Lcg) -> (Vec<Vec<isize>>, Vec<isize>, usize) {
        let n = 2 + rng.below(4) as usize;
        let domains: Vec<Vec<isize>> = (0..n)
            .map(|_| {
                let mut d: Vec<isize> = (0..4).filter(|_| rng.below(2) == 1).collect();
                if d.is_empty() {
                    d.push(rng.below(4) as isize);
                }
                d
            })
            .collect();
        let mut values: Vec<isize> = (0..4).filter(|_| rng.below(2) == 1).collect();
        if values.is_empty() {
            values.push(rng.below(4) as isize);
        }
        let lb = rng.below(n as u64 + 2) as usize;
        (domains, values, lb)
    }

    #[test]
    fn random_instances_are_exact_once_settled() {
        let mut rng = Lcg(2024);
        for _ in 0..200 {
            let (domains, values, lb) = random_instance(&mut rng);
            let n = domains.len();
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let (problem, _) = full(&domains, &values, lb);
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &[(&scope, &values, lb)]),
                "domains {domains:?} values {values:?} lb {lb} order {order:?}"
            );
        }
    }

    #[test]
    fn the_initial_relaxation_never_loses_a_solution() {
        // Width 1 keeps the first relaxation, with no splitting and no merging.
        let mut rng = Lcg(77);
        for _ in 0..200 {
            let (domains, values, lb) = random_instance(&mut rng);
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &values, lb)]);
            let (problem, _) = full(&domains, &values, lb);
            let got = accepted(&compile(problem, (0..n).collect(), 1));
            for solution in &want {
                assert!(
                    got.contains(solution),
                    "lost {solution:?}: domains {domains:?} values {values:?} lb {lb}"
                );
            }
        }
    }

    #[test]
    fn relaxed_mdds_never_lose_a_solution() {
        // For random instances and several width budgets, every solution of the oracle must be
        // accepted: a relaxation may accept more, never less.
        let mut rng = Lcg(77);
        for _ in 0..200 {
            let (domains, values, lb) = random_instance(&mut rng);
            let n = domains.len();
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &values, lb)]);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = full(&domains, &values, lb);
                let mdd = compile(problem, order.clone(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {domains:?} values {values:?} lb {lb} order {order:?} width {width}"
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
        let (mut problem, vars) = full(&[dom(3), dom(3)], &[1], 1);
        problem[vars[0]].set_domain(vec![1]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![1, 0], vec![1, 1], vec![1, 2]]);
    }

    #[test]
    fn a_domain_that_grows_after_construction_is_fine() {
        // Only the set of counted values is stored, so a new value is simply counted.
        let (mut problem, vars) = full(&[dom(2), dom(2)], &[9], 1);
        problem[vars[0]].set_domain(vec![0, 9]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![9, 0], vec![9, 1]]);
    }
}
