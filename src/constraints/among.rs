//! `Among`: between `lb` and `ub` variables of the scope take a value of a given set.
//!
//! # Meaning
//!
//! Given a set of values $V$ and bounds $lb \leq ub$, constrains $x_1, \dots, x_n$ so that
//!
//! $$lb \leq |\{ i \mid x_i \in V \}| \leq ub.$$
//!
//! Values of $V$ that are in no domain are never counted. With $lb > ub$, or $lb > n$, the
//! constraint has no solution; the MDD detects it only once the intervals below are single
//! values, since wider intervals are kept. An empty scope counts 0, so it is satisfied exactly when $lb = 0$;
//! as for [`Sum`], the MDD has no layer at which to filter anything, so an empty
//! `Among` is not seen by the compilation.
//!
//! A variable may appear only once in the scope: the compiled property cannot know how many times
//! a layer is counted, so [`Among::new`] panics on a repeated variable instead of compiling
//! something different from [`is_satisfied`](Constraint::is_satisfied).
//!
//! # How it compiles
//!
//! A node carries the interval $[m, M]$ of the number of scope variables that take a value of $V$
//! on the paths that reach it (top-down) or leave it (bottom-up): $m$ is the smallest such count
//! and $M$ the largest. A node with no path at all holds the sentinel $[+\infty, 0]$, which is
//! recognised by $m > M$.
//!
//! ## Property update
//!
//! On an edge of the scope with a value in $V$, the interval of the parent (or child) is shifted
//! by 1; any other edge, and every out of scope layer, copies it. When a node has several
//! parents, the results are combined by taking the smallest $m$ and the largest $M$. A parent with
//! no path is ignored.
//!
//! ## Node merging
//!
//! Merging two nodes takes the smallest $m$ and the largest $M$. The merged interval contains the
//! counts of both nodes, so the merged node over-approximates them (it also covers the counts in
//! between, which may not be reachable).
//!
//! ## Edge filtering
//!
//! An edge with value $v$ between a parent $[m_p, M_p]$ and a child $[m_c, M_c]$ lies on paths
//! whose count is somewhere in $[m_p + m_c + \delta, M_p + M_c + \delta]$, with $\delta = 1$ if
//! $v \in V$ and 0 otherwise. It is removed when that interval does not intersect $[lb, ub]$,
//! that is when $m_p + m_c + \delta > ub$ or $M_p + M_c + \delta < lb$. An edge next to a node
//! with no path at all is removed too. This is bounds reasoning, as for `Sum`: a count that is
//! inside the interval but not reachable is not detected, so a compiled MDD may keep paths that
//! violate the constraint until nodes are split enough for the intervals to become single values.
//!
//! # Design notes
//!
//! - **Only the set $V$ is stored**, in an [`Arc`] shared by the constraint and all its
//!   properties. No domain is read, so domains may change freely after construction.
//! - **The structural key is self-contained**: it holds the arity, the sorted values of $V$, and
//!   both bounds. Two `Among` built with the values in a different order get the same key.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Among, Constraint};
//! use aicad::modelling::*;
//! use rustc_hash::FxHashSet;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2], None);
//! // Between one and two variables take a value in {1, 2}.
//! let among = Among::new(vars, FxHashSet::from_iter([1, 2]), 1, 2);
//!
//! assert!(among.is_satisfied(&[1, 0, 2]));
//! assert!(!among.is_satisfied(&[0, 0, 0]));
//! assert!(!among.is_satisfied(&[1, 2, 1]));
//! ```
use super::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`Among`]: the smallest and the largest number of scope variables that take
/// a value of the set along the paths that reach the node (top-down) or leave it (bottom-up).
///
/// `(usize::MAX, 0)` means that no path has been folded in yet, which is the identity of `merge`;
/// it is recognised by `min > max`. `merge` takes the smallest minimum and the largest maximum, so
/// a merged node over-approximates both parents. See the [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
struct AmongProperty {
    /// The set $V$ of counted values.
    values: Arc<FxHashSet<isize>>,
    min: usize,
    max: usize,
}

impl AmongProperty {
    fn new(values: Arc<FxHashSet<isize>>, min: usize, max: usize) -> Self {
        Self { values, min, max }
    }

    /// True if no path has been folded into this property.
    fn has_no_path(&self) -> bool {
        self.min > self.max
    }
}

/// The constraint that between `lb` and `ub` variables take a value in a set. See the
/// [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Among {
    /// Scope of the constraint, without repetition.
    variables: Vec<VariableIndex>,
    /// The set $V$ of counted values.
    values: Arc<FxHashSet<isize>>,
    lb: usize,
    ub: usize,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
}

impl Among {
    /// Builds the constraint that between `lb` and `ub` of `variables` take a value in `values`.
    ///
    /// # Panics
    ///
    /// If a variable appears more than once in `variables`.
    pub fn new(
        variables: Vec<VariableIndex>,
        values: FxHashSet<isize>,
        lb: usize,
        ub: usize,
    ) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        assert!(
            distinct.len() == variables.len(),
            "Among does not support a variable repeated in its scope"
        );
        Self {
            variables,
            values: Arc::new(values),
            lb,
            ub,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Among {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let mut values: Vec<isize> = self.values.iter().copied().collect();
        values.sort_unstable();
        ConstraintShapeKey::Among {
            arity: self.variables.len(),
            values,
            lb: self.lb,
            ub: self.ub,
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
        let parent = parent.as_any().downcast_ref::<AmongProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of AmongProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<AmongProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of AmongProperty",
                    child.name()
                );
        });
        // A node without any path cannot be crossed by an edge. This also keeps the additions
        // below away from the sentinel `usize::MAX`, which would overflow.
        if parent.has_no_path() || child.has_no_path() {
            return true;
        }

        let delta = usize::from(self.values.contains(&assignment));
        let local_lb = parent.min + child.min + delta;
        let local_ub = parent.max + child.max + delta;
        local_lb > self.ub || local_ub < self.lb
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
        self.lb <= count && count <= self.ub
    }

    fn name(&self) -> &'static str {
        "Among"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AmongProperty::new(self.values.clone(), usize::MAX, 0))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AmongProperty::new(self.values.clone(), 0, 0))
    }
}

impl ConstraintProperty for AmongProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<AmongProperty>()
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
        self.min = self.min.min(other.min + delta);
        self.max = self.max.max(other.max + delta);
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<AmongProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

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
            .downcast_ref::<AmongProperty>()
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
        "AmongProperty"
    }
}

#[cfg(test)]
mod test_among {
    use super::AmongProperty;
    use crate::constraints::{
        AllDifferent, Among, Constraint, ConstraintProperty, ConstraintShapeKey,
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

    /// One variable per domain, and an Among over the variables listed in `scope`.
    fn scoped(
        domains: &[Vec<isize>],
        scope: &[usize],
        values: &[isize],
        lb: usize,
        ub: usize,
    ) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        among(
            &mut problem,
            scope.iter().map(|&i| vars[i]).collect(),
            values.to_vec(),
            lb,
            ub,
        );
        (problem, vars)
    }

    /// One variable per domain, all in the scope.
    fn full(
        domains: &[Vec<isize>],
        values: &[isize],
        lb: usize,
        ub: usize,
    ) -> (Problem, Vec<VariableIndex>) {
        scoped(
            domains,
            &(0..domains.len()).collect::<Vec<_>>(),
            values,
            lb,
            ub,
        )
    }

    /// The constraint over variables `0..n` with the identity ordering already set.
    fn among_of(n: usize, values: &[isize], lb: usize, ub: usize) -> Among {
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(3), None);
        let mut among = Among::new(vars.clone(), set(values), lb, ub);
        among.update_variable_ordering(&vars);
        among
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

    /// One test case: domains, counted values, lower and upper bound.
    type Case = (Vec<Vec<isize>>, Vec<isize>, usize, usize);

    /// One Among of the oracle: scope positions, counted values, bounds.
    type Spec<'a> = (&'a [usize], &'a [isize], usize, usize);

    /// The expected solutions, written independently of `Among`: the cartesian product of the
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
            specs.iter().all(|(scope, values, lb, ub)| {
                let mut count = 0;
                for &position in scope.iter() {
                    if values.contains(&t[position]) {
                        count += 1;
                    }
                }
                *lb <= count && count <= *ub
            })
        });
        tuples.sort();
        tuples
    }

    /// The property of a single path that assigns `values`, built the way the compiler builds
    /// it: one in-scope `update` per edge.
    fn path_property(among: &Among, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = among.empty_property();
        for &value in values {
            let mut next = among.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        property
    }

    /// A property whose interval is exactly `[lo, hi]`, for a constraint counting the value 1.
    fn interval(among: &Among, lo: usize, hi: usize) -> Box<dyn ConstraintProperty> {
        let mut property = path_property(among, &vec![1; lo]);
        property.merge(&*path_property(among, &vec![1; hi]));
        property
    }

    fn clone_of(property: &dyn ConstraintProperty) -> Box<dyn ConstraintProperty> {
        Box::new(
            property
                .as_any()
                .downcast_ref::<AmongProperty>()
                .unwrap()
                .clone(),
        )
    }

    /// The `(min, max)` of a property.
    fn bounds(property: &dyn ConstraintProperty) -> (usize, usize) {
        let property = property.as_any().downcast_ref::<AmongProperty>().unwrap();
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
        // Every assignment over {0, 1, 2}, for every pair of bounds, counting the values {1, 2}.
        for lb in 0..=4 {
            for ub in 0..=4 {
                let among = among_of(3, &[1, 2], lb, ub);
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            let count = [a, b, c].iter().filter(|&&v| v >= 1).count();
                            assert_eq!(
                                among.is_satisfied(&[a, b, c]),
                                lb <= count && count <= ub,
                                "{:?} in [{lb}, {ub}]",
                                [a, b, c]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        let among = among_of(3, &[1, 2], 1, 2);
        // 1 and 2 are in the set, 0 is not: counts 2, 0 and 3.
        assert!(among.is_satisfied(&[1, 0, 2]));
        assert!(!among.is_satisfied(&[0, 0, 0]));
        assert!(!among.is_satisfied(&[1, 2, 1]));
        // lb == ub
        let exactly_one = among_of(2, &[5], 1, 1);
        assert!(exactly_one.is_satisfied(&[5, 0]));
        assert!(!exactly_one.is_satisfied(&[5, 5]));
        assert!(!exactly_one.is_satisfied(&[0, 0]));
    }

    #[test]
    fn values_outside_the_domains_are_never_counted() {
        let among = among_of(2, &[7, 8], 1, 2);
        assert!(!among.is_satisfied(&[0, 1]));
    }

    #[test]
    fn is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        let among = Among::new(vec![vars[0], vars[2]], set(&[1]), 1, 1);
        assert!(among.is_satisfied(&[1, 1, 0]));
        assert!(!among.is_satisfied(&[0, 1, 0]));
    }

    #[test]
    fn empty_scope_counts_zero() {
        assert!(Among::new(vec![], set(&[1]), 0, 0).is_satisfied(&[]));
        assert!(Among::new(vec![], set(&[1]), 0, 3).is_satisfied(&[]));
        assert!(!Among::new(vec![], set(&[1]), 1, 2).is_satisfied(&[]));
    }

    #[test]
    fn bounds_that_cannot_be_met_are_never_satisfied() {
        // lb > ub, and lb above the number of variables.
        let crossed = among_of(3, &[1], 2, 1);
        let too_many = among_of(3, &[1], 4, 5);
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    assert!(!crossed.is_satisfied(&[a, b, c]));
                    assert!(!too_many.is_satisfied(&[a, b, c]));
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        among_of(3, &[1], 0, 3).is_satisfied(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        Among::new(vec![x, y, x], set(&[1]), 1, 2);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_name_and_structural_key() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(5), None);
        let among = Among::new(vars.clone(), set(&[4, 1, 3]), 1, 2);
        assert_eq!(among.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(among.name(), "Among");
        assert_eq!(
            among.structural_key(&problem),
            ConstraintShapeKey::Among {
                arity: 3,
                values: vec![1, 3, 4],
                lb: 1,
                ub: 2
            }
        );
    }

    #[test]
    fn the_structural_key_does_not_depend_on_the_order_of_the_values() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(5), None);
        let a = Among::new(vars.clone(), set(&[1, 2, 3]), 0, 1);
        let b = Among::new(vars.clone(), set(&[3, 1, 2]), 0, 1);
        let different_bound = Among::new(vars, set(&[1, 2, 3]), 0, 2);
        assert_eq!(a.structural_key(&problem), b.structural_key(&problem));
        assert_ne!(
            a.structural_key(&problem),
            different_bound.structural_key(&problem)
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, dom(2), None);
        let mut among = Among::new(vec![vars[0], vars[2], vars[3]], set(&[1]), 1, 1);
        // Layer order: v3, v1, v0, v2.
        among.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[2]]);
        assert!(among.is_layer_in_scope(0));
        assert!(!among.is_layer_in_scope(1));
        assert!(among.is_layer_in_scope(2));
        assert!(among.is_layer_in_scope(3));
    }

    #[test]
    fn layers_beyond_the_first_word_are_tracked() {
        // 130 variables, the scope is at layers 0, 63, 64, 65 and 129.
        let mut problem = Problem::default();
        let vars = problem.add_variables(130, dom(2), None);
        let scope_layers = [0usize, 63, 64, 65, 129];
        let scope: Vec<VariableIndex> = scope_layers.iter().map(|&l| vars[l]).collect();
        let mut among = Among::new(scope, set(&[1]), 1, 2);
        among.update_variable_ordering(&vars);
        for layer in 0..130 {
            assert_eq!(
                among.is_layer_in_scope(layer),
                scope_layers.contains(&layer),
                "layer {layer}"
            );
        }
    }

    #[test]
    fn a_new_ordering_replaces_the_previous_one() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        let mut among = Among::new(vec![vars[0], vars[1]], set(&[1]), 1, 1);
        among.update_variable_ordering(&[vars[0], vars[1], vars[2]]);
        assert!(
            among.is_layer_in_scope(0) && among.is_layer_in_scope(1) && !among.is_layer_in_scope(2)
        );
        among.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        assert!(
            !among.is_layer_in_scope(0) && among.is_layer_in_scope(1) && among.is_layer_in_scope(2)
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        Among::new(vars, set(&[1]), 1, 1).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        let mut among = Among::new(vars.clone(), set(&[1]), 1, 1);
        // `vars[1]` does not appear in the order.
        among.update_variable_ordering(&[vars[0]]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn empty_property_is_the_zero_interval() {
        let among = among_of(2, &[1], 0, 2);
        assert_eq!(bounds(&*among.empty_property()), (0, 0));
        assert_eq!(among.empty_property().order_key(), vec![0.0, 0.0]);
    }

    #[test]
    fn identity_property_holds_the_no_path_sentinels() {
        let among = among_of(2, &[1], 0, 2);
        assert_eq!(bounds(&*among.identity_property()), (usize::MAX, 0));
        assert!(!same(&*among.identity_property(), &*among.empty_property()));
    }

    #[test]
    fn update_counts_only_the_values_of_the_set() {
        let among = among_of(3, &[1, 2], 0, 3);
        assert_eq!(bounds(&*path_property(&among, &[0])), (0, 0));
        assert_eq!(bounds(&*path_property(&among, &[1])), (1, 1));
        assert_eq!(bounds(&*path_property(&among, &[2, 0, 1])), (2, 2));
        assert_eq!(bounds(&*path_property(&among, &[0, 0, 0])), (0, 0));
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let among = among_of(2, &[1], 0, 2);
        let parent = interval(&among, 1, 2);
        let mut child = among.identity_property();
        // The value 1 is in the set but the layer is out of scope: nothing is counted.
        child.update(&*parent, 1, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn folding_two_parents_takes_the_extreme_counts() {
        let among = among_of(3, &[1], 0, 3);
        let mut node = among.identity_property();
        node.update(&*interval(&among, 0, 1), 1, true);
        node.update(&*interval(&among, 2, 3), 1, true);
        assert_eq!(bounds(&*node), (1, 4));
    }

    #[test]
    fn folding_does_not_depend_on_the_order_of_the_parents() {
        let among = among_of(3, &[1], 0, 3);
        let left = interval(&among, 0, 1);
        let right = interval(&among, 2, 3);
        let mut lr = among.identity_property();
        lr.update(&*left, 1, true);
        lr.update(&*right, 1, true);
        let mut rl = among.identity_property();
        rl.update(&*right, 1, true);
        rl.update(&*left, 1, true);
        assert!(same(&*lr, &*rl));
    }

    #[test]
    fn a_parent_without_paths_stays_without_paths() {
        // Folding the identity must neither overflow nor turn the sentinel maximum 0 into a count.
        let among = among_of(2, &[1], 0, 2);
        let mut node = among.identity_property();
        node.update(&*among.identity_property(), 1, true);
        assert!(same(&*node, &*among.identity_property()));
        // And it does not disturb a real parent folded in afterwards.
        node.update(&*path_property(&among, &[0]), 1, true);
        assert_eq!(bounds(&*node), (1, 1));
    }

    #[test]
    fn merge_takes_the_smallest_min_and_the_largest_max() {
        let among = among_of(3, &[1], 0, 3);
        let mut a = interval(&among, 1, 2);
        a.merge(&*interval(&among, 0, 1));
        assert_eq!(bounds(&*a), (0, 2));
        assert_eq!(a.order_key(), vec![0.0, 2.0]);
        let mut disjoint = interval(&among, 0, 0);
        disjoint.merge(&*interval(&among, 3, 3));
        assert_eq!(bounds(&*disjoint), (0, 3));
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let among = among_of(3, &[1], 0, 3);
        let a = interval(&among, 1, 2);
        let b = interval(&among, 0, 3);

        let mut twice = clone_of(&*a);
        twice.merge(&*a);
        assert!(same(&*twice, &*a));

        let mut ab = clone_of(&*a);
        ab.merge(&*b);
        let mut ba = clone_of(&*b);
        ba.merge(&*a);
        assert!(same(&*ab, &*ba));

        let mut with_identity = clone_of(&*a);
        with_identity.merge(&*among.identity_property());
        assert!(same(&*with_identity, &*a));
    }

    #[test]
    fn eq_and_hash_compare_both_bounds() {
        let among = among_of(3, &[1], 0, 3);
        let a = interval(&among, 1, 2);
        assert!(same(&*a, &*interval(&among, 1, 2)));
        assert_eq!(hash_of(&*a), hash_of(&*interval(&among, 1, 2)));
        assert!(!same(&*a, &*interval(&among, 1, 3)));
        assert!(!same(&*a, &*interval(&among, 0, 2)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&among, 1, 3)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&among, 0, 2)));
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        let among = among_of(2, &[1], 0, 2);
        among.identity_property().update(&*foreign(), 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        let among = among_of(2, &[1], 0, 2);
        among.identity_property().merge(&*foreign());
    }

    #[test]
    #[should_panic(expected = "Calling eq on property")]
    fn eq_with_a_property_of_another_constraint_is_rejected() {
        let among = among_of(2, &[1], 0, 2);
        same(&*among.identity_property(), &*foreign());
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn an_edge_is_kept_when_the_intervals_meet() {
        // Counting the value 1, bounds [3, 4]. Parent [1, 2] and child [0, 1]: with an edge in the
        // set the count is in [2, 4], which meets the bounds.
        let among = among_of(4, &[1], 3, 4);
        let parent = interval(&among, 1, 2);
        let child = interval(&among, 0, 1);
        assert!(!among.is_assignment_invalid(&*parent, &*child, 1, 1));
    }

    #[test]
    fn the_bounds_of_the_interval_are_inclusive() {
        // Parent [1, 2], child [0, 1]. An edge in the set gives the counts [2, 4], an edge out of
        // the set gives [1, 3].
        let helper = among_of(4, &[1], 0, 4);
        let parent = interval(&helper, 1, 2);
        let child = interval(&helper, 0, 1);
        // (lb, ub, edge value, invalid?)
        let cases = [
            (5, 5, 1, true),  // max 4 < lb 5
            (4, 4, 1, false), // max 4 == lb 4
            (0, 2, 1, false), // min 2 == ub 2
            (0, 1, 1, true),  // min 2 > ub 1
            (4, 4, 0, true),  // max 3 < lb 4
            (3, 3, 0, false), // max 3 == lb 3
            (0, 1, 0, false), // min 1 == ub 1
            (0, 0, 0, true),  // min 1 > ub 0
        ];
        for (lb, ub, value, invalid) in cases {
            let among = among_of(4, &[1], lb, ub);
            assert_eq!(
                among.is_assignment_invalid(&*parent, &*child, 1, value),
                invalid,
                "bounds [{lb}, {ub}] edge value {value}"
            );
        }
    }

    #[test]
    fn only_edges_in_the_set_shift_the_interval() {
        // One path with count 1, bounds [2, 2]: an edge in the set reaches 2, an edge out of the
        // set stays at 1.
        let among = among_of(2, &[1], 2, 2);
        let parent = path_property(&among, &[1]);
        let child = among.empty_property();
        assert!(!among.is_assignment_invalid(&*parent, &*child, 1, 1));
        assert!(among.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn the_parent_and_the_child_both_count() {
        let among = among_of(3, &[1], 2, 2);
        let one = path_property(&among, &[1]);
        let none = among.empty_property();
        // Parent only: 1 + 1 = 2 with an edge in the set. Child only: the same.
        assert!(!among.is_assignment_invalid(&*one, &*none, 1, 1));
        assert!(!among.is_assignment_invalid(&*none, &*one, 1, 1));
        // Both: 1 + 1 + 1 = 3 > ub.
        assert!(among.is_assignment_invalid(&*one, &*one, 1, 1));
    }

    #[test]
    fn an_edge_is_kept_inside_a_hole_of_the_interval() {
        // The reachable counts are only 0 and 2 but the interval is [0, 2]: bounds reasoning keeps
        // the bound [1, 1]. This is the documented relaxation.
        let among = among_of(2, &[1], 1, 1);
        let parent = interval(&among, 0, 2);
        let child = among.empty_property();
        assert!(!among.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn crossed_bounds_invalidate_every_edge_of_a_single_count() {
        // With lb > ub no count is acceptable. Bounds reasoning sees it only on intervals that
        // are single values; a wider interval is kept (it may straddle the empty range).
        let among = among_of(2, &[1], 2, 1);
        let child = among.empty_property();
        for ones in 0..=2 {
            let parent = path_property(&among, &vec![1; ones]);
            for value in 0..3 {
                assert!(
                    among.is_assignment_invalid(&*parent, &*child, 1, value),
                    "count {ones} edge value {value}"
                );
            }
        }
    }

    #[test]
    fn an_edge_next_to_a_node_without_paths_is_invalid() {
        // Such a node is about to be removed; the sentinels must not be added to anything.
        let among = among_of(2, &[1], 0, 2);
        let identity = among.identity_property();
        let real = path_property(&among, &[1]);
        assert!(among.is_assignment_invalid(&*identity, &*identity, 0, 0));
        assert!(among.is_assignment_invalid(&*identity, &*real, 0, 1));
        assert!(among.is_assignment_invalid(&*real, &*identity, 0, 1));
    }

    #[test]
    #[should_panic(expected = "instead of AmongProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let among = among_of(2, &[1], 0, 2);
        let own = among.empty_property();
        among.is_assignment_invalid(&*foreign(), &*own, 0, 0);
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
        // (domains, counted values, lb, ub)
        let cases: Vec<Case> = vec![
            (vec![dom(2), dom(2)], vec![1], 1, 1),
            (vec![dom(3), dom(3), dom(3)], vec![0, 1], 1, 2),
            (vec![dom(2), dom(2), dom(2)], vec![1], 3, 3),
            (vec![dom(2), dom(2), dom(2)], vec![1], 0, 3),
            (vec![dom(2), dom(2), dom(2)], vec![1], 0, 0),
            (vec![dom(3), dom(3), dom(3), dom(3)], vec![2], 2, 3),
            (vec![vec![-1, 4], vec![4, 7], vec![-1, 7]], vec![4, 7], 1, 1),
            (vec![dom(3), vec![5, 6], dom(3)], vec![5, 0], 1, 2),
        ];
        for (domains, values, lb, ub) in cases {
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            for order in permutations(n) {
                let (problem, _) = full(&domains, &values, lb, ub);
                let mdd = settled(problem, order.clone());
                assert_eq!(
                    accepted(&mdd),
                    expected(&domains, &[(&scope, &values, lb, ub)]),
                    "domains {domains:?} values {values:?} [{lb}, {ub}] order {order:?}"
                );
            }
        }
    }

    #[test]
    fn unreachable_bounds_are_unsat() {
        // Never in the set; always in the set; lb above the arity; crossed bounds.
        let cases: Vec<Case> = vec![
            (vec![vec![0], vec![0]], vec![1], 1, 2),
            (vec![vec![1], vec![1]], vec![1], 0, 0),
            (vec![dom(2), dom(2)], vec![1], 3, 3),
            (vec![dom(2), dom(2)], vec![1], 2, 1),
        ];
        for (domains, values, lb, ub) in cases {
            let (problem, _) = full(&domains, &values, lb, ub);
            let mdd = settled(problem, vec![0, 1]);
            assert!(
                accepted(&mdd).is_empty(),
                "{domains:?} {values:?} [{lb}, {ub}]"
            );
        }
    }

    #[test]
    fn a_set_with_no_value_of_the_domains_counts_zero() {
        let domains = [dom(2), dom(2)];
        let (problem, _) = full(&domains, &[9], 0, 0);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd).len(), 4);
        let (problem, _) = full(&domains, &[9], 1, 2);
        let mdd = settled(problem, vec![0, 1]);
        assert!(accepted(&mdd).is_empty());
    }

    /// Two Among constraints on one variable per domain.
    fn two_scopes(domains: &[Vec<isize>], first: Spec, second: Spec) -> Problem {
        let (mut problem, vars) = scoped(domains, first.0, first.1, first.2, first.3);
        among(
            &mut problem,
            second.0.iter().map(|&i| vars[i]).collect(),
            second.1.to_vec(),
            second.2,
            second.3,
        );
        problem
    }

    #[test]
    fn a_constraint_ignores_the_layers_of_the_other_one() {
        let domains = [dom(3), dom(3), dom(3), dom(3)];
        let first: Spec = (&[0, 2], &[1], 1, 1);
        let second: Spec = (&[1, 3], &[2], 0, 1);
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
        let first: Spec = (&[0, 1], &[1, 2], 1, 1);
        let second: Spec = (&[1, 2], &[0], 1, 2);
        let problem = two_scopes(&domains, first, second);
        let mdd = settled(problem, vec![0, 1, 2]);
        assert_eq!(accepted(&mdd), expected(&domains, &[first, second]));
    }

    #[test]
    fn combined_with_not_equals_is_exact() {
        let domains = [dom(3), dom(3), dom(3)];
        let (mut problem, vars) = full(&domains, &[2], 1, 1);
        not_equals(&mut problem, vars[0], vars[1]);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want: Vec<Vec<isize>> = expected(&domains, &[(&[0, 1, 2], &[2], 1, 1)])
            .into_iter()
            .filter(|t| t[0] != t[1])
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
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
    fn the_initial_relaxation_never_loses_a_solution() {
        // Width 1 keeps the first relaxation, with no splitting and no merging.
        let mut rng = Lcg(77);
        for _ in 0..200 {
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
            let lb = rng.below(n as u64 + 1) as usize;
            let ub = lb + rng.below((n - lb) as u64 + 1) as usize;
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &values, lb, ub)]);
            let (problem, _) = full(&domains, &values, lb, ub);
            let got = accepted(&compile(problem, (0..n).collect(), 1));
            for solution in &want {
                assert!(
                    got.contains(solution),
                    "lost {solution:?}: domains {domains:?} values {values:?} [{lb}, {ub}]"
                );
            }
        }
    }

    #[test]
    fn merging_nodes_keeps_the_edges_of_the_merged_node() {
        // Smallest instance found by search. `merge_nodes_with_flag` used to take the removed
        // edges of the target node for live ones, which lost the solution [0, 0, 1] at width 3.
        let domains = [dom(3), dom(3), dom(2)];
        let (problem, _) = full(&domains, &[1], 0, 1);
        let got = accepted(&compile(problem, vec![0, 1, 2], 3));
        assert!(got.contains(&vec![0, 0, 1]));
    }

    #[test]
    fn relaxed_mdds_never_lose_a_solution() {
        // For random instances and several width budgets, every solution of the oracle must be
        // accepted: a relaxation may accept more, never less.
        let mut rng = Lcg(77);
        for _ in 0..200 {
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
            let lb = rng.below(n as u64 + 1) as usize;
            let ub = lb + rng.below((n - lb) as u64 + 1) as usize;
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &values, lb, ub)]);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = full(&domains, &values, lb, ub);
                let mdd = compile(problem, order.clone(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {domains:?} values {values:?} [{lb}, {ub}] order {order:?} width {width}"
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
        let (mut problem, vars) = full(&[dom(3), dom(3)], &[1], 1, 1);
        problem[vars[0]].set_domain(vec![1]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![1, 0], vec![1, 2]]);
    }

    #[test]
    fn a_domain_that_grows_after_construction_is_fine() {
        // Only the set of counted values is stored, so a new value is simply counted.
        let (mut problem, vars) = full(&[dom(2), dom(2)], &[9], 1, 1);
        problem[vars[0]].set_domain(vec![0, 9]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![9, 0], vec![9, 1]]);
    }
}
