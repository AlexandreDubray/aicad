//! `Sum`: the variables of the scope must add up to a target value.
//!
//! # Meaning
//!
//! Constrains $x_1, \dots, x_n$ so that $\sum_i x_i = t$ for a given target $t$. Values may be
//! negative. With a single variable it fixes that variable to $t$.
//!
//! A variable may appear only once in the scope: the compiled property cannot know how many times
//! a layer is counted, so [`Sum::new`] panics on a repeated variable instead of compiling
//! something different from [`is_satisfied`](Constraint::is_satisfied).
//!
//! An empty scope sums to 0. [`is_satisfied`](Constraint::is_satisfied) reflects it, but the MDD
//! has no layer at which to filter anything, so an empty `Sum` with a non-zero target is not seen
//! by the compilation.
//!
//! # How it compiles
//!
//! A node carries the interval $[m, M]$ of the sums of the scope values on the paths that reach
//! it (top-down) or leave it (bottom-up): $m$ is the smallest such sum and $M$ the largest.
//! A node with no path at all holds the sentinel $[+\infty, -\infty]$.
//!
//! ## Property update
//!
//! On an edge of the scope with value $v$, the interval of the parent (or child) is shifted by
//! $v$; out of scope layers copy it. When a node has several parents, the results are combined by
//! taking the smallest $m$ and the largest $M$.
//!
//! ## Node merging
//!
//! Merging two nodes takes the smallest $m$ and the largest $M$. The merged interval contains the
//! sums of both nodes, so the merged node over-approximates them (it also covers the values in
//! between, which may not be reachable).
//!
//! ## Edge filtering
//!
//! An edge with value $v$ between a parent $[m_p, M_p]$ and a child $[m_c, M_c]$ lies on paths
//! whose total is somewhere in $[m_p + v + m_c, M_p + v + M_c]$. It is removed when that interval
//! does not contain the target $t$. An edge next to a node with no path at all is removed too.
//! This is bounds reasoning: a sum that is inside the interval
//! but not reachable (a hole) is not detected, so a compiled MDD may keep paths that do not sum
//! to $t$ until nodes are split enough for the intervals to become single values.
//!
//! # Design notes
//!
//! - **No domain is read.** The property only adds the assigned values, so domains may change
//!   freely after construction. [`Sum::new`] takes the problem only to match the other
//!   constructors.
//! - **No overflow handling.** Sums are plain `isize` additions, which assumes that the sum of
//!   the largest values does not overflow. The sentinels $\pm\infty$ are `isize::MAX` and
//!   `isize::MIN` and are never added to.
//! - **The domain is not part of the structural key**, only the arity and the target; the arena
//!   supplies the domain, see the key's documentation.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, Sum};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2, 3], None);
//! let sum = Sum::new(vars, 6, &problem);
//!
//! assert!(sum.is_satisfied(&[1, 2, 3]));
//! assert!(!sum.is_satisfied(&[1, 2, 2]));
//! ```
use super::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;

/// Per-node state of [`Sum`]: the smallest and the largest sum of the scope values along the
/// paths that reach the node (top-down) or leave it (bottom-up).
///
/// `(isize::MAX, isize::MIN)` means that no path has been folded in yet, which is the identity of
/// `merge`. `merge` takes the smallest minimum and the largest maximum, so a merged node
/// over-approximates both parents. See the [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct SumProperty {
    min: isize,
    max: isize,
}

/// The constraint $\sum_i x_i = t$. See the [module documentation](self) for the semantics and
/// for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Sum {
    /// Scope of the constraint, without repetition.
    variables: Vec<VariableIndex>,
    /// Target value the sum of the scope's variables must equal.
    target: isize,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called.
    layer_in_scope: Vec<u64>,
}

impl Sum {
    /// Builds the constraint that `variables` sum to `target`.
    ///
    /// `problem` is not read; it is kept so that all constraints are built the same way.
    ///
    /// # Panics
    ///
    /// If a variable appears more than once in `variables`.
    pub fn new(variables: Vec<VariableIndex>, target: isize, _problem: &Problem) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        if distinct.len() != variables.len() {
            panic!("Sum does not support a variable repeated in its scope");
        }
        Self {
            variables,
            target,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Sum {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        ConstraintShapeKey::Sum {
            arity: self.variables.len(),
            target: self.target,
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
        let parent = parent.as_any().downcast_ref::<SumProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of SumProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<SumProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of SumProperty",
                    child.name()
                );
        });
        // A node without any path (the sentinels) cannot be crossed by an edge. This also keeps
        // the additions below away from the sentinels, which would overflow.
        if parent.min > parent.max || child.min > child.max {
            return true;
        }

        let local_min = parent.min + child.min + assignment;
        let local_max = parent.max + child.max + assignment;
        local_min > self.target || local_max < self.target
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover the scope, otherwise this
        // panics on the out-of-range index.
        let mut total: isize = 0;
        for variable in self.variables.iter().copied() {
            total += assignment[variable.0];
        }
        total == self.target
    }

    fn name(&self) -> &'static str {
        "Sum"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(SumProperty {
            min: isize::MAX,
            max: isize::MIN,
        })
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(SumProperty { min: 0, max: 0 })
    }
}

impl ConstraintProperty for SumProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<SumProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        let delta = if in_scope { assignment } else { 0 };
        // The sentinels (no path yet) must stay sentinels: adding to them would overflow.
        let other_min = if other.min == isize::MAX {
            isize::MAX
        } else {
            other.min + delta
        };
        let other_max = if other.max == isize::MIN {
            isize::MIN
        } else {
            other.max + delta
        };
        self.min = self.min.min(other_min);
        self.max = self.max.max(other_max);
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<SumProperty>()
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
        hasher.write_isize(self.min);
        hasher.write_isize(self.max);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<SumProperty>()
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
        "SumProperty"
    }
}

#[cfg(test)]
mod test_sum {
    use super::SumProperty;
    use crate::constraints::{
        AllDifferent, Constraint, ConstraintProperty, ConstraintShapeKey, Sum,
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

    /// One variable per domain, and a Sum over the variables listed in `scope`.
    fn scoped(
        domains: &[Vec<isize>],
        scope: &[usize],
        target: isize,
    ) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        sum(
            &mut problem,
            scope.iter().map(|&i| vars[i]).collect(),
            target,
        );
        (problem, vars)
    }

    /// One variable per domain, all in the scope.
    fn full(domains: &[Vec<isize>], target: isize) -> (Problem, Vec<VariableIndex>) {
        scoped(domains, &(0..domains.len()).collect::<Vec<_>>(), target)
    }

    /// The constraint over variables `0..domains.len()` with the identity ordering already set.
    fn sum_of(domains: &[Vec<isize>], target: isize) -> Sum {
        let (problem, vars) = full(domains, target);
        let mut sum = Sum::new(vars.clone(), target, &problem);
        sum.update_variable_ordering(&vars);
        sum
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
    /// a second pass would remove (see the task "MDD propagation is not iterated to a
    /// fixpoint"). The exactness tests need the fixpoint, because they check
    /// what the constraint rules out, not how many passes the engine runs.
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

    /// The expected solutions, written independently of `Sum`: the cartesian product of the
    /// domains, keeping the tuples whose entries at the `scopes` positions add up to the
    /// matching target.
    fn expected(domains: &[Vec<isize>], scopes: &[(&[usize], isize)]) -> Vec<Vec<isize>> {
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
            scopes
                .iter()
                .all(|(scope, target)| scope.iter().map(|&i| t[i]).sum::<isize>() == *target)
        });
        tuples.sort();
        tuples
    }

    /// The property of a single path that assigns `values`, built the way the compiler builds
    /// it: one in-scope `update` per edge.
    fn path_property(sum: &Sum, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = sum.empty_property();
        for &value in values {
            let mut next = sum.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        property
    }

    /// A property whose interval is exactly `[lo, hi]`.
    fn interval(sum: &Sum, lo: isize, hi: isize) -> Box<dyn ConstraintProperty> {
        let mut property = path_property(sum, &[lo]);
        property.merge(&*path_property(sum, &[hi]));
        property
    }

    fn clone_of(property: &dyn ConstraintProperty) -> Box<dyn ConstraintProperty> {
        Box::new(
            property
                .as_any()
                .downcast_ref::<SumProperty>()
                .unwrap()
                .clone(),
        )
    }

    /// The `(min, max)` of a property.
    fn bounds(property: &dyn ConstraintProperty) -> (isize, isize) {
        let property = property.as_any().downcast_ref::<SumProperty>().unwrap();
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

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_agrees_with_the_definition() {
        // Every assignment over a domain with negative values, for several targets.
        let values = [-2, 0, 3];
        for target in -4..=6 {
            let sum = sum_of(&[values.to_vec(), values.to_vec(), values.to_vec()], target);
            for a in values {
                for b in values {
                    for c in values {
                        assert_eq!(
                            sum.is_satisfied(&[a, b, c]),
                            a + b + c == target,
                            "{:?} target {target}",
                            [a, b, c]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        let sum = sum_of(&[dom(4), dom(4), dom(4)], 6);
        assert!(sum.is_satisfied(&[1, 2, 3]));
        assert!(sum.is_satisfied(&[3, 3, 0]));
        assert!(!sum.is_satisfied(&[1, 2, 2]));
        assert!(!sum.is_satisfied(&[3, 3, 1]));
        let negative = sum_of(&[vec![-5, 2], vec![-5, 2]], -3);
        assert!(negative.is_satisfied(&[-5, 2]));
        assert!(!negative.is_satisfied(&[2, 2]));
    }

    #[test]
    fn is_satisfied_ignores_out_of_scope_variables() {
        let (problem, vars) = scoped(&[dom(5), dom(5), dom(5)], &[0, 2], 4);
        let sum = Sum::new(vec![vars[0], vars[2]], 4, &problem);
        assert!(sum.is_satisfied(&[1, 4, 3]));
        assert!(sum.is_satisfied(&[1, 0, 3]));
        assert!(!sum.is_satisfied(&[1, 3, 4]));
    }

    #[test]
    fn empty_scope_sums_to_zero() {
        let problem = Problem::default();
        assert!(Sum::new(vec![], 0, &problem).is_satisfied(&[]));
        assert!(!Sum::new(vec![], 1, &problem).is_satisfied(&[]));
    }

    #[test]
    fn a_single_variable_must_equal_the_target() {
        let sum = sum_of(&[dom(5)], 3);
        assert!(sum.is_satisfied(&[3]));
        assert!(!sum.is_satisfied(&[2]));
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        sum_of(&[dom(2), dom(2), dom(2)], 1).is_satisfied(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        Sum::new(vec![x, y, x], 3, &problem);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_name_and_structural_key() {
        let (problem, vars) = full(&[dom(2), dom(2), dom(2)], 2);
        let sum = Sum::new(vars.clone(), -7, &problem);
        assert_eq!(sum.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(sum.name(), "Sum");
        assert_eq!(
            sum.structural_key(&problem),
            ConstraintShapeKey::Sum {
                arity: 3,
                target: -7
            }
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2), dom(2)], &[0, 2, 3], 1);
        let mut sum = Sum::new(vec![vars[0], vars[2], vars[3]], 1, &problem);
        // Layer order: v3, v1, v0, v2.
        sum.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[2]]);
        assert!(sum.is_layer_in_scope(0));
        assert!(!sum.is_layer_in_scope(1));
        assert!(sum.is_layer_in_scope(2));
        assert!(sum.is_layer_in_scope(3));
    }

    #[test]
    fn layers_beyond_the_first_word_are_tracked() {
        // 130 variables, the scope is at layers 0, 63, 64, 65 and 129.
        let mut problem = Problem::default();
        let vars = problem.add_variables(130, dom(2), None);
        let scope_layers = [0usize, 63, 64, 65, 129];
        let scope: Vec<VariableIndex> = scope_layers.iter().map(|&l| vars[l]).collect();
        let mut sum = Sum::new(scope, 2, &problem);
        sum.update_variable_ordering(&vars);
        for layer in 0..130 {
            assert_eq!(
                sum.is_layer_in_scope(layer),
                scope_layers.contains(&layer),
                "layer {layer}"
            );
        }
    }

    #[test]
    fn a_new_ordering_replaces_the_previous_one() {
        let (problem, vars) = scoped(&[dom(2), dom(2), dom(2)], &[0, 1], 1);
        let mut sum = Sum::new(vec![vars[0], vars[1]], 1, &problem);
        sum.update_variable_ordering(&[vars[0], vars[1], vars[2]]);
        assert!(sum.is_layer_in_scope(0) && sum.is_layer_in_scope(1) && !sum.is_layer_in_scope(2));
        sum.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        assert!(!sum.is_layer_in_scope(0) && sum.is_layer_in_scope(1) && sum.is_layer_in_scope(2));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        Sum::new(vars, 1, &problem).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        let mut sum = Sum::new(vars.clone(), 1, &problem);
        // `vars[1]` does not appear in the order.
        sum.update_variable_ordering(&[vars[0]]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn empty_property_is_the_zero_interval() {
        let sum = sum_of(&[dom(3), dom(3)], 2);
        assert_eq!(bounds(&*sum.empty_property()), (0, 0));
        assert_eq!(sum.empty_property().order_key(), vec![0.0, 0.0]);
    }

    #[test]
    fn identity_property_holds_the_no_path_sentinels() {
        let sum = sum_of(&[dom(3), dom(3)], 2);
        assert_eq!(bounds(&*sum.identity_property()), (isize::MAX, isize::MIN));
        assert!(!same(&*sum.identity_property(), &*sum.empty_property()));
    }

    #[test]
    fn a_path_adds_up_its_values() {
        let sum = sum_of(&[dom(5), dom(5), dom(5)], 4);
        assert_eq!(bounds(&*path_property(&sum, &[1])), (1, 1));
        assert_eq!(bounds(&*path_property(&sum, &[1, 4, 2])), (7, 7));
        assert_eq!(bounds(&*path_property(&sum, &[-3, 1])), (-2, -2));
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let parent = interval(&sum, 2, 6);
        let mut child = sum.identity_property();
        // The assignment must not be added when the layer is out of scope.
        child.update(&*parent, 99, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn folding_two_parents_takes_the_extreme_sums() {
        let sum = sum_of(&[dom(5), dom(5), dom(5)], 4);
        let left = interval(&sum, 1, 3);
        let right = interval(&sum, 2, 8);
        let mut node = sum.identity_property();
        node.update(&*left, 10, true);
        node.update(&*right, 10, true);
        assert_eq!(bounds(&*node), (11, 18));
    }

    #[test]
    fn folding_does_not_depend_on_the_order_of_the_parents() {
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let left = interval(&sum, 1, 3);
        let right = interval(&sum, -2, 8);
        let mut lr = sum.identity_property();
        lr.update(&*left, 2, true);
        lr.update(&*right, 2, true);
        let mut rl = sum.identity_property();
        rl.update(&*right, 2, true);
        rl.update(&*left, 2, true);
        assert!(same(&*lr, &*rl));
    }

    #[test]
    fn a_parent_without_paths_stays_without_paths() {
        // Updating from the identity must neither overflow nor invent a path.
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let mut node = sum.identity_property();
        node.update(&*sum.identity_property(), 3, true);
        assert!(same(&*node, &*sum.identity_property()));
        // And it does not disturb a real parent folded in afterwards.
        node.update(&*path_property(&sum, &[1]), 3, true);
        assert_eq!(bounds(&*node), (4, 4));
    }

    #[test]
    fn merge_takes_the_smallest_min_and_the_largest_max() {
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let mut a = interval(&sum, 1, 3);
        a.merge(&*interval(&sum, 2, 8));
        assert_eq!(bounds(&*a), (1, 8));
        assert_eq!(a.order_key(), vec![1.0, 8.0]);
        let mut disjoint = interval(&sum, -4, -2);
        disjoint.merge(&*interval(&sum, 5, 6));
        assert_eq!(bounds(&*disjoint), (-4, 6));
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let a = interval(&sum, 1, 3);
        let b = interval(&sum, -2, 8);

        let mut twice = clone_of(&*a);
        twice.merge(&*a);
        assert!(same(&*twice, &*a));

        let mut ab = clone_of(&*a);
        ab.merge(&*b);
        let mut ba = clone_of(&*b);
        ba.merge(&*a);
        assert!(same(&*ab, &*ba));

        let mut with_identity = clone_of(&*a);
        with_identity.merge(&*sum.identity_property());
        assert!(same(&*with_identity, &*a));
    }

    #[test]
    fn eq_and_hash_compare_both_bounds() {
        let sum = sum_of(&[dom(5), dom(5)], 4);
        let a = interval(&sum, 1, 3);
        assert!(same(&*a, &*interval(&sum, 1, 3)));
        assert_eq!(hash_of(&*a), hash_of(&*interval(&sum, 1, 3)));
        // Same min, different max; same max, different min.
        assert!(!same(&*a, &*interval(&sum, 1, 4)));
        assert!(!same(&*a, &*interval(&sum, 0, 3)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&sum, 1, 4)));
        assert_ne!(hash_of(&*a), hash_of(&*interval(&sum, 0, 3)));
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        let sum = sum_of(&[dom(2), dom(2)], 1);
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        let foreign = AllDifferent::new(vars, &problem).identity_property();
        sum.identity_property().update(&*foreign, 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        let sum = sum_of(&[dom(2), dom(2)], 1);
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        let foreign = AllDifferent::new(vars, &problem).identity_property();
        sum.identity_property().merge(&*foreign);
    }

    #[test]
    #[should_panic(expected = "Calling eq on property")]
    fn eq_with_a_property_of_another_constraint_is_rejected() {
        let sum = sum_of(&[dom(2), dom(2)], 1);
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        let foreign = AllDifferent::new(vars, &problem).identity_property();
        same(&*sum.identity_property(), &*foreign);
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn an_edge_is_kept_when_the_target_is_inside_the_interval() {
        // Target 10. The edge value 4 gives the totals [3 + 4 + 1, 5 + 4 + 2] = [8, 11].
        let sum = sum_of(&[dom(9), dom(9), dom(9)], 10);
        let parent = interval(&sum, 3, 5);
        let child = interval(&sum, 1, 2);
        assert!(!sum.is_assignment_invalid(&*parent, &*child, 1, 4));
    }

    #[test]
    fn the_bounds_of_the_interval_are_inclusive() {
        let parent_lo = 3;
        let child_lo = 1;
        let parent_hi = 5;
        let child_hi = 2;
        let (parent, child) = {
            let helper = sum_of(&[dom(9), dom(9), dom(9)], 0);
            (
                interval(&helper, parent_lo, parent_hi),
                interval(&helper, child_lo, child_hi),
            )
        };
        // With edge value 4 the totals are [8, 11]: 8 and 11 are reachable, 7 and 12 are not.
        for (target, invalid) in [(7, true), (8, false), (11, false), (12, true)] {
            let sum = sum_of(&[dom(9), dom(9), dom(9)], target);
            assert_eq!(
                sum.is_assignment_invalid(&*parent, &*child, 1, 4),
                invalid,
                "target {target}"
            );
        }
    }

    #[test]
    fn an_edge_is_kept_inside_a_hole_of_the_interval() {
        // Reachable totals are only 0 and 10 but the interval is [0, 10]: bounds reasoning keeps
        // a target of 5. This is the documented relaxation.
        let sum = sum_of(&[dom(11), dom(11)], 5);
        let parent = interval(&sum, 0, 10);
        let child = sum.empty_property();
        assert!(!sum.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn a_single_point_interval_is_an_exact_test() {
        let parent = path_property(&sum_of(&[dom(9), dom(9)], 0), &[2]);
        let child = sum_of(&[dom(9), dom(9)], 0).empty_property();
        for assignment in 0..9 {
            let sum = sum_of(&[dom(9), dom(9)], 7);
            assert_eq!(
                sum.is_assignment_invalid(&*parent, &*child, 1, assignment),
                2 + assignment != 7,
                "assignment {assignment}"
            );
        }
    }

    #[test]
    fn negative_values_and_targets_are_handled() {
        // Totals are [-6 + -2, -4 + -2] = [-8, -6] for the edge value -2.
        let helper = sum_of(&[dom(2), dom(2)], 0);
        let parent = interval(&helper, -6, -4);
        let child = helper.empty_property();
        for (target, invalid) in [(-9, true), (-8, false), (-6, false), (-5, true), (0, true)] {
            let sum = sum_of(&[dom(2), dom(2)], target);
            assert_eq!(
                sum.is_assignment_invalid(&*parent, &*child, 1, -2),
                invalid,
                "target {target}"
            );
        }
    }

    #[test]
    fn the_parent_and_the_child_both_count() {
        let helper = sum_of(&[dom(2), dom(2)], 0);
        let sum = sum_of(&[dom(9), dom(9), dom(9)], 10);
        // Only the parent: [2, 2] + 3 + [0, 0] = 5; only the child: 3 + [7, 7] = 10.
        let parent = path_property(&helper, &[2]);
        let child = path_property(&helper, &[7]);
        assert!(sum.is_assignment_invalid(&*parent, &*helper.empty_property(), 1, 3));
        assert!(!sum.is_assignment_invalid(&*helper.empty_property(), &*child, 1, 3));
        // Both: 2 + 3 + 7 = 12 > 10.
        assert!(sum.is_assignment_invalid(&*parent, &*child, 1, 3));
    }

    #[test]
    #[should_panic(expected = "instead of SumProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let sum = sum_of(&[dom(2), dom(2)], 1);
        let (problem, vars) = full(&[dom(2), dom(2)], 1);
        let foreign = AllDifferent::new(vars, &problem).identity_property();
        let own = sum.empty_property();
        sum.is_assignment_invalid(&*foreign, &*own, 0, 0);
    }

    #[test]
    fn an_edge_next_to_a_node_without_paths_is_invalid() {
        // Such a node is about to be removed; the sentinels must not be added to anything.
        let sum = sum_of(&[dom(5), dom(5)], 3);
        let identity = sum.identity_property();
        let real = path_property(&sum, &[1]);
        assert!(sum.is_assignment_invalid(&*identity, &*identity, 0, 0));
        assert!(sum.is_assignment_invalid(&*identity, &*real, 0, 2));
        assert!(sum.is_assignment_invalid(&*real, &*identity, 0, 2));
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
        let cases: Vec<(Vec<Vec<isize>>, isize)> = vec![
            (vec![dom(3), dom(3)], 3),
            (vec![dom(4), dom(4), dom(4)], 5),
            (vec![vec![-1, 0, 1], vec![-1, 0, 1]], 0),
            (vec![vec![-5, -2, 2, 5], vec![-5, -2, 2, 5]], -3),
            (vec![vec![0, 10], vec![0, 5], vec![0, 1]], 11),
            (vec![dom(3), vec![2, 7], dom(4)], 9),
            (vec![dom(2), dom(2), dom(2), dom(2)], 2),
        ];
        for (domains, target) in cases {
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            for order in permutations(n) {
                let (problem, _) = full(&domains, target);
                let mdd = settled(problem, order.clone());
                assert_eq!(
                    accepted(&mdd),
                    expected(&domains, &[(&scope, target)]),
                    "domains {domains:?} target {target} order {order:?}"
                );
            }
        }
    }

    #[test]
    fn a_single_variable_is_fixed_to_the_target() {
        let (problem, _) = full(&[dom(5)], 3);
        let mdd = settled(problem, vec![0]);
        assert_eq!(accepted(&mdd), vec![vec![3]]);
    }

    #[test]
    fn unreachable_targets_are_unsat() {
        for (domains, target) in [
            (vec![vec![0], vec![0]], 5),
            (vec![dom(2), dom(2)], 10),
            (vec![dom(2), dom(2)], -10),
        ] {
            let (problem, _) = full(&domains, target);
            let mdd = settled(problem, vec![0, 1]);
            assert!(mdd.is_unsat(), "{domains:?} target {target}");
            assert_eq!(mdd.get_solution(), None);
        }
    }

    #[test]
    fn a_target_inside_a_hole_has_no_solution() {
        // Only the totals 0 and 10 are reachable; the target 5 is inside the interval.
        let domains = [vec![0, 5], vec![0, 5]];
        let (problem, _) = full(&domains, 5);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), expected(&domains, &[(&[0, 1], 5)]));
        let domains = [vec![0, 10], vec![0, 10]];
        let (problem, _) = full(&domains, 5);
        let mdd = settled(problem, vec![0, 1]);
        assert!(accepted(&mdd).is_empty());
    }

    /// Two Sum constraints, over `first` and `second`, on one variable per domain.
    fn two_scopes(
        domains: &[Vec<isize>],
        first: (&[usize], isize),
        second: (&[usize], isize),
    ) -> Problem {
        let (mut problem, vars) = scoped(domains, first.0, first.1);
        sum(
            &mut problem,
            second.0.iter().map(|&i| vars[i]).collect(),
            second.1,
        );
        problem
    }

    #[test]
    fn a_constraint_ignores_the_layers_of_the_other_one() {
        let domains = [dom(3), dom(3), dom(3), dom(3)];
        for order in [vec![0, 1, 2, 3], vec![3, 1, 2, 0], vec![1, 3, 0, 2]] {
            let problem = two_scopes(&domains, (&[0, 2], 3), (&[1, 3], 1));
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &[(&[0, 2], 3), (&[1, 3], 1)]),
                "order {order:?}"
            );
        }
    }

    #[test]
    fn overlapping_sums_are_exact() {
        let domains = [dom(4), dom(4), dom(4)];
        let problem = two_scopes(&domains, (&[0, 1], 4), (&[1, 2], 3));
        let mdd = settled(problem, vec![0, 1, 2]);
        assert_eq!(
            accepted(&mdd),
            expected(&domains, &[(&[0, 1], 4), (&[1, 2], 3)])
        );
    }

    #[test]
    fn combined_with_all_different_is_exact() {
        let domains = [dom(4), dom(4), dom(4)];
        let (mut problem, vars) = full(&domains, 6);
        all_different(&mut problem, vars);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want: Vec<Vec<isize>> = expected(&domains, &[(&[0, 1, 2], 6)])
            .into_iter()
            .filter(|t| t[0] != t[1] && t[0] != t[2] && t[1] != t[2])
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert_eq!(want.len(), 6);
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
        let mut rng = Lcg(2024);
        for _ in 0..200 {
            let n = 2 + rng.below(4) as usize;
            let domains: Vec<Vec<isize>> = (0..n)
                .map(|_| {
                    let mut d: Vec<isize> = (-3..4).filter(|_| rng.below(2) == 1).collect();
                    if d.is_empty() {
                        d.push(rng.below(7) as isize - 3);
                    }
                    d
                })
                .collect();
            let target = rng.below(11) as isize - 5;
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, target)]);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = full(&domains, target);
                let mdd = compile(problem, order.clone(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {domains:?} target {target} order {order:?} width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_target_between_the_sums_of_the_edges_is_unsat() {
        // x, y in {0, 10}, x + y = 5: no pair sums to 5, so the MDD must be empty.
        let domains = [vec![0, 10], vec![0, 10]];
        let (problem, _) = full(&domains, 5);
        let mdd = settled(problem, vec![0, 1]);
        assert!(accepted(&mdd).is_empty());
    }

    // ----------------------------------------------------------------------------------------
    // Domains that change after construction
    // ----------------------------------------------------------------------------------------

    #[test]
    fn a_domain_that_shrinks_after_construction_is_fine() {
        let (mut problem, vars) = full(&[dom(4), dom(4)], 4);
        problem[vars[0]].set_domain(vec![1, 3]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![1, 3], vec![3, 1]]);
    }

    #[test]
    fn a_domain_that_grows_after_construction_is_fine() {
        // No domain is read at construction time, so a new value is simply summed.
        let (mut problem, vars) = full(&[dom(2), dom(2)], 10);
        problem[vars[0]].set_domain(vec![0, 1, 9]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![9, 1]]);
    }
}
