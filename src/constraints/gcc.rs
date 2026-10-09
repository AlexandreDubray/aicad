//! `Gcc`: global cardinality, each value of a given set occurs a bounded number of times.
//!
//! # Meaning
//!
//! Given a list of values $v$ with bounds $lo_v \leq hi_v$, constrains $x_1, \dots, x_n$ so that
//!
//! $$\forall v: \quad lo_v \leq |\{ i \mid x_i = v \}| \leq hi_v.$$
//!
//! Values that are not in the list are free: they may occur any number of times. A value that is
//! in no domain occurs 0 times, so a bound with $lo_v > 0$ on it has no solution. With
//! $lo_v > hi_v$, or $lo_v > n$, the constraint has no solution either; the MDD detects it only
//! once its intervals are single values, as for [`Among`]. An empty scope counts 0 for every
//! value, so it is satisfied exactly when no $lo_v$ is positive; as for [`Sum`], the MDD has no
//! layer at which to filter anything, so an empty `Gcc` is not seen by the compilation.
//!
//! This generalises [`Among`], which bounds the *total* count of a set of values: a `Gcc` with
//! the single bound $(v, lo, hi)$ is the same constraint as `Among` on $\{v\}$. With $hi_v = 1$
//! for every value of the domains it is an [`AllDifferent`].
//!
//! A variable may appear only once in the scope, and a value may have only one bound. The
//! compiled property cannot know how many times a layer is counted, and two bounds on one value
//! would have to be intersected, so [`Gcc::new`] panics on either instead of compiling something
//! different from [`is_satisfied`](Constraint::is_satisfied).
//!
//! # How it compiles
//!
//! A node carries, for every bounded value $v$, the interval $[m_v, M_v]$ of the number of scope
//! variables that take the value $v$ on the paths that reach it (top-down) or leave it
//! (bottom-up). $m_v$ is the smallest such count and $M_v$ the largest. Both are replaced by
//! $lo_v$ when they exceed it if the bound $hi_v$ can never be reached, that is if
//! $hi_v \geq n$ (see below). A node with no path at all holds the sentinel $[+\infty, 0]$ for
//! every value, recognised by $m_v > M_v$.
//!
//! ## Property update
//!
//! On an edge of the scope with a bounded value $v$, the interval of $v$ of the parent (or
//! child) is shifted by 1; the intervals of the other values, and all of them on an edge with an
//! unbounded value or on an out of scope layer, are copied. When a node has several parents, the
//! results are combined value by value, taking the smallest $m_v$ and the largest $M_v$. A
//! parent with no path is ignored.
//!
//! When $hi_v \geq n$ the upper bound can never bind, and once $lo_v$ variables are counted
//! nothing that happens later can matter. The counts of $v$ are then capped at $lo_v$. This is an
//! exact reduction of the state space, not a relaxation, and it is the same idea as for
//! [`AtLeast`].
//!
//! ## Node merging
//!
//! Merging two nodes takes, for every value, the smallest $m_v$ and the largest $M_v$. The merged
//! intervals contain the counts of both nodes, so the merged node over-approximates them.
//!
//! ## Edge filtering
//!
//! An edge with value $w$ between a parent and a child lies, for each bounded value $v$, on paths
//! whose count of $v$ is somewhere in $[m_p + m_c + \delta, M_p + M_c + \delta]$, with
//! $\delta = 1$ if $v = w$ and 0 otherwise. The edge is removed as soon as one value $v$ has
//! $m_p + m_c + \delta > hi_v$ or $M_p + M_c + \delta < lo_v$. This includes an edge with an
//! unbounded value: it adds nothing to any count, but it can still be the one that leaves too few
//! variables to reach some $lo_v$. An edge next to a node with no path at all is removed too.
//!
//! This is bounds reasoning on each value on its own, as for `Among`: a count that is inside an
//! interval but not reachable is not detected, and neither is a combination of values that no
//! path realises (two values that must each occur twice among three variables, for example). A
//! compiled MDD may keep paths that violate the constraint until nodes are split enough for the
//! intervals to become single values.
//!
//! # Design notes
//!
//! - **The state grows with the number of bounded values.** A node holds one interval per bounded
//!   value, so the number of distinct states can grow exponentially with that number. This is why
//!   a `Gcc` can compile to a much larger diagram than an `Among` or an `AtLeast`.
//! - **Values and bounds are copied into [`Arc`]s** shared by the constraint and all its
//!   properties. No domain is read, so domains may change freely after construction.
//! - **The structural key is self-contained**: it holds the arity and the bounds sorted by value.
//!   The compiled structure does not depend on the order in which the bounds were declared (the
//!   order only permutes the slots of the property), so two `Gcc` declared in a different order
//!   get the same key.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, Gcc};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(4, vec![0, 1, 2], None);
//! // The value 1 occurs exactly once, the value 2 occurs at most once; the value 0 is free.
//! let gcc = Gcc::new(vars, vec![(1, 1, 1), (2, 0, 1)]);
//!
//! assert!(gcc.is_satisfied(&[0, 1, 2, 0]));
//! assert!(gcc.is_satisfied(&[0, 0, 1, 0]));
//! assert!(!gcc.is_satisfied(&[1, 1, 0, 0]));
//! assert!(!gcc.is_satisfied(&[2, 2, 1, 0]));
//! ```
use super::*;
use crate::modelling::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`Gcc`]: for every bounded value, the smallest and the largest number of
/// scope variables that take it along the paths that reach the node (top-down) or leave it
/// (bottom-up).
///
/// The vectors have one slot per bounded value. `(usize::MAX, 0)` in every slot means that no
/// path has been folded in yet, which is the identity of `merge`; it is recognised by
/// `min > max`. `merge` takes the smallest minimum and the largest maximum of each slot, so a
/// merged node over-approximates both parents. See the [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
struct GccProperty {
    /// Maps each bounded value to its slot
    map: Arc<FxHashMap<isize, usize>>,
    /// The lower bound of each value, by slot
    lo: Arc<Vec<usize>>,
    /// Whether the upper bound of each value, by slot, can never be reached (`hi >= n`). Its
    /// counts are then capped at `lo`.
    saturates: Arc<Vec<bool>>,
    /// Smallest count of each value, by slot
    min: Vec<usize>,
    /// Largest count of each value, by slot
    max: Vec<usize>,
}

impl GccProperty {
    fn new(
        n: usize,
        map: Arc<FxHashMap<isize, usize>>,
        lo: Arc<Vec<usize>>,
        saturates: Arc<Vec<bool>>,
        min_seed: usize,
    ) -> Self {
        Self {
            map,
            lo,
            saturates,
            min: vec![min_seed; n],
            max: vec![0; n],
        }
    }

    /// True if no path has been folded into this property. A property without any slot (a `Gcc`
    /// with no bound) never has a path to tell apart, and filters nothing.
    fn has_no_path(&self) -> bool {
        self.min.iter().zip(&self.max).any(|(min, max)| min > max)
    }
}

/// The constraint that each of a list of values occurs a bounded number of times. See the
/// [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Gcc {
    /// Scope of the constraint, without repetition.
    variables: Vec<VariableIndex>,
    /// The bounds `(value, lo, hi)`, one per value, in the order given to [`Gcc::new`]. The slot of
    /// a value in the properties is its position in this list.
    bounds: Vec<(isize, usize, usize)>,
    /// Maps each bounded value to its slot in the properties' vectors
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    /// The required lower occurrence bound of each value, by slot
    lo: Arc<Vec<usize>>,
    /// Whether the upper occurrence bound of each value, by slot, can never be reached
    /// (`hi >= |variables|`)
    saturates: Arc<Vec<bool>>,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
}

impl Gcc {
    /// Builds the constraint that every value of `bounds`, a list of `(value, lo, hi)`, occurs
    /// between `lo` and `hi` times among `variables`. A value that is not in `bounds` is
    /// unconstrained.
    ///
    /// # Panics
    ///
    /// If a variable appears more than once in `variables`, or a value more than once in
    /// `bounds`.
    pub fn new(variables: Vec<VariableIndex>, bounds: Vec<(isize, usize, usize)>) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        if distinct.len() != variables.len() {
            panic!("Gcc does not support a variable repeated in its scope");
        }
        let values: FxHashSet<isize> = bounds.iter().map(|&(value, _, _)| value).collect();
        if values.len() != bounds.len() {
            panic!("Gcc does not support two bounds for the same value");
        }

        let val_to_bit = Arc::new(
            bounds
                .iter()
                .copied()
                .enumerate()
                .map(|(bit, (value, _, _))| (value, bit))
                .collect(),
        );
        let lo: Vec<usize> = bounds.iter().copied().map(|(_, lo, _)| lo).collect();
        let saturates: Vec<bool> = bounds
            .iter()
            .map(|&(_, _, hi)| hi >= variables.len())
            .collect();
        Self {
            variables,
            bounds,
            val_to_bit,
            lo: Arc::new(lo),
            saturates: Arc::new(saturates),
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Gcc {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let mut bounds = self.bounds.clone();
        bounds.sort_unstable_by_key(|&(value, _, _)| value);
        ConstraintShapeKey::Gcc {
            arity: self.variables.len(),
            bounds,
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
        let parent = parent.as_any().downcast_ref::<GccProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of GccProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<GccProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of GccProperty",
                    child.name()
                );
        });
        // A node without any path cannot be crossed by an edge. This also keeps the additions
        // below away from the sentinel `usize::MAX`, which would overflow.
        if parent.has_no_path() || child.has_no_path() {
            return true;
        }

        // `bit` is `None` when `assignment` isn't itself one of the bounded values - in that
        // case this edge contributes `delta = 0` to every bounded value's count, but the bound
        // check below must still run: an edge to an *unbounded* value can still be the one that
        // makes some other bounded value's count infeasible to complete (not enough variables
        // left to reach its lower bound, or already past its upper bound).
        let bit = self.val_to_bit.get(&assignment).copied();
        for (slot, (_, lb, ub)) in self.bounds.iter().copied().enumerate() {
            let delta = usize::from(bit == Some(slot));
            let min = parent.min[slot] + child.min[slot] + delta;
            if min > ub {
                return true;
            }
            let max = parent.max[slot] + child.max[slot] + delta;
            if max < lb {
                return true;
            }
        }
        false
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover the scope, otherwise this
        // panics on the out-of-range index.
        self.bounds.iter().all(|&(value, lo, hi)| {
            let count = self
                .variables
                .iter()
                .filter(|variable| assignment[variable.0] == value)
                .count();
            lo <= count && count <= hi
        })
    }

    fn name(&self) -> &'static str {
        "GCC"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(GccProperty::new(
            self.bounds.len(),
            self.val_to_bit.clone(),
            self.lo.clone(),
            self.saturates.clone(),
            usize::MAX,
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(GccProperty::new(
            self.bounds.len(),
            self.val_to_bit.clone(),
            self.lo.clone(),
            self.saturates.clone(),
            0,
        ))
    }
}

impl ConstraintProperty for GccProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<GccProperty>()
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

        // The slot that this edge counts, if any: no slot is counted out of scope or for an
        // unbounded value, and `self.min.len()` stands for it.
        let target_bit = if in_scope {
            match self.map.get(&assignment) {
                None => self.min.len(),
                Some(&bit) => bit,
            }
        } else {
            self.min.len()
        };

        // Then, we integrate the min-max values for each bounded value from the other property.
        // For a value whose upper bound can never bind (`saturates[bit]`), cap the tracked
        // count at its lower bound once reached: every count >= lo is then behaviorally
        // identical for every future decision, so collapsing them is exact, not a relaxation.
        for bit in 0..self.min.len() {
            if bit == target_bit {
                let mut new_min = other.min[bit] + 1;
                let mut new_max = other.max[bit] + 1;
                if self.saturates[bit] {
                    new_min = new_min.min(self.lo[bit]);
                    new_max = new_max.min(self.lo[bit]);
                }
                self.min[bit] = self.min[bit].min(new_min);
                self.max[bit] = self.max[bit].max(new_max);
            } else {
                self.min[bit] = self.min[bit].min(other.min[bit]);
                self.max[bit] = self.max[bit].max(other.max[bit]);
            }
        }
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<GccProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        for bit in 0..self.min.len() {
            self.min[bit] = self.min[bit].min(other.min[bit]);
            self.max[bit] = self.max[bit].max(other.max[bit]);
        }
    }

    fn order_key(&self) -> Vec<f64> {
        // One axis per bounded value's min, then one axis per bounded value's max: Gcc's bounds
        // are genuinely separate dimensions of state, so they aren't collapsed to one number.
        self.min
            .iter()
            .chain(self.max.iter())
            .map(|&x| x as f64)
            .collect()
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for &bound in self.min.iter() {
            hasher.write_usize(bound);
        }
        for &bound in self.max.iter() {
            hasher.write_usize(bound);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<GccProperty>()
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
        "GccProperty"
    }
}

#[cfg(test)]
mod test_gcc {
    use super::GccProperty;
    use crate::constraints::{
        AllDifferent, Constraint, ConstraintProperty, ConstraintShapeKey, Gcc,
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

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    /// One variable per domain, and a Gcc over the variables listed in `scope`.
    fn scoped(
        domains: &[Vec<isize>],
        scope: &[usize],
        bounds: &[(isize, usize, usize)],
    ) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        gcc(
            &mut problem,
            scope.iter().map(|&i| vars[i]).collect(),
            bounds.to_vec(),
        );
        (problem, vars)
    }

    /// One variable per domain, all in the scope.
    fn full(
        domains: &[Vec<isize>],
        bounds: &[(isize, usize, usize)],
    ) -> (Problem, Vec<VariableIndex>) {
        scoped(domains, &(0..domains.len()).collect::<Vec<_>>(), bounds)
    }

    /// The constraint over variables `0..n` with the identity ordering already set.
    fn gcc_of(n: usize, bounds: &[(isize, usize, usize)]) -> Gcc {
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(3), None);
        let mut constraint = Gcc::new(vars.clone(), bounds.to_vec());
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
    /// a second pass would remove (see the task "MDD propagation is not iterated to a
    /// fixpoint"). The exactness tests need the fixpoint, because they check what the constraint
    /// rules out, not how many passes the engine runs.
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

    /// One test case: domains and bounds.
    type Case = (Vec<Vec<isize>>, Vec<(isize, usize, usize)>);

    /// One Gcc of the oracle: scope positions and bounds.
    type Spec<'a> = (&'a [usize], &'a [(isize, usize, usize)]);

    /// The expected solutions, written independently of `Gcc`: the cartesian product of the
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
            specs.iter().all(|(scope, bounds)| {
                bounds.iter().all(|&(value, lo, hi)| {
                    let mut count = 0;
                    for &position in scope.iter() {
                        if t[position] == value {
                            count += 1;
                        }
                    }
                    lo <= count && count <= hi
                })
            })
        });
        tuples.sort();
        tuples
    }

    /// The property of a single path that assigns `values`, built the way the compiler builds
    /// it: one in-scope `update` per edge.
    fn path_property(constraint: &Gcc, values: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = constraint.empty_property();
        for &value in values {
            let mut next = constraint.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        property
    }

    /// A property whose interval of `value` is exactly `[lo, hi]` and whose other intervals are
    /// `[0, 0]`. The constraint must not saturate `value`, otherwise the counts are capped.
    fn interval(
        constraint: &Gcc,
        value: isize,
        lo: usize,
        hi: usize,
    ) -> Box<dyn ConstraintProperty> {
        let mut property = path_property(constraint, &vec![value; lo]);
        property.merge(&*path_property(constraint, &vec![value; hi]));
        property
    }

    /// A constraint of ten variables with the bounds `(1, 0, 3)` and `(2, 0, 3)`, which saturate
    /// nothing: the properties it builds are the ones used to probe other constraints that have
    /// the same two values in the same order.
    fn helper() -> Gcc {
        gcc_of(10, &[(1, 0, 3), (2, 0, 3)])
    }

    fn clone_of(property: &dyn ConstraintProperty) -> Box<dyn ConstraintProperty> {
        Box::new(
            property
                .as_any()
                .downcast_ref::<GccProperty>()
                .unwrap()
                .clone(),
        )
    }

    /// The `(min, max)` of the interval of `value` in a property.
    fn bounds_of(property: &dyn ConstraintProperty, value: isize) -> (usize, usize) {
        let property = property.as_any().downcast_ref::<GccProperty>().unwrap();
        let slot = property.map[&value];
        (property.min[slot], property.max[slot])
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
        // Every assignment over {0, 1, 2}, for several lists of bounds.
        let lists: Vec<Vec<(isize, usize, usize)>> = vec![
            vec![(1, 1, 2)],
            vec![(0, 0, 1), (1, 1, 1)],
            vec![(2, 2, 3), (0, 0, 0)],
            vec![(1, 3, 4)],
            vec![(0, 1, 2), (1, 1, 2), (2, 1, 2)],
        ];
        for bounds in lists {
            let constraint = gcc_of(3, &bounds);
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let tuple = [a, b, c];
                        let holds = bounds.iter().all(|&(value, lo, hi)| {
                            let count = tuple.iter().filter(|&&v| v == value).count();
                            lo <= count && count <= hi
                        });
                        assert_eq!(
                            constraint.is_satisfied(&tuple),
                            holds,
                            "{tuple:?} with {bounds:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        // The value 1 occurs exactly once, the value 2 at most once, the value 0 is free.
        let constraint = gcc_of(4, &[(1, 1, 1), (2, 0, 1)]);
        assert!(constraint.is_satisfied(&[0, 1, 2, 0]));
        assert!(constraint.is_satisfied(&[0, 0, 1, 0]));
        assert!(!constraint.is_satisfied(&[1, 1, 0, 0]));
        assert!(!constraint.is_satisfied(&[2, 2, 1, 0]));
        assert!(!constraint.is_satisfied(&[0, 0, 2, 0]));
    }

    #[test]
    fn an_unbounded_value_is_free() {
        // Only the value 0 is constrained; the value 3 may occur any number of times.
        let constraint = gcc_of(2, &[(0, 1, 1)]);
        assert!(constraint.is_satisfied(&[0, 3]));
        assert!(constraint.is_satisfied(&[3, 0]));
        assert!(!constraint.is_satisfied(&[3, 3]));
        assert!(!constraint.is_satisfied(&[0, 0]));
    }

    #[test]
    fn a_value_that_never_occurs_counts_zero() {
        let must_occur = gcc_of(2, &[(9, 1, 2)]);
        let may_not_occur = gcc_of(2, &[(9, 0, 2)]);
        for a in 0..2 {
            for b in 0..2 {
                assert!(!must_occur.is_satisfied(&[a, b]));
                assert!(may_not_occur.is_satisfied(&[a, b]));
            }
        }
    }

    #[test]
    fn no_bound_accepts_everything() {
        let constraint = gcc_of(2, &[]);
        assert!(constraint.is_satisfied(&[0, 0]));
        assert!(constraint.is_satisfied(&[1, 2]));
    }

    #[test]
    fn empty_scope_counts_zero() {
        assert!(Gcc::new(vec![], vec![]).is_satisfied(&[]));
        assert!(Gcc::new(vec![], vec![(1, 0, 2)]).is_satisfied(&[]));
        assert!(!Gcc::new(vec![], vec![(1, 1, 2)]).is_satisfied(&[]));
    }

    #[test]
    fn bounds_that_cannot_be_met_are_never_satisfied() {
        // lo > hi, and lo above the number of variables.
        let crossed = gcc_of(3, &[(1, 2, 1)]);
        let too_many = gcc_of(3, &[(1, 4, 5)]);
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
    fn is_satisfied_ignores_out_of_scope_variables() {
        // Only variables 0 and 2 are in scope.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        let constraint = Gcc::new(vec![vars[0], vars[2]], vec![(1, 2, 2)]);
        assert!(constraint.is_satisfied(&[1, 0, 1]));
        assert!(!constraint.is_satisfied(&[1, 1, 0]));
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        gcc_of(3, &[(1, 0, 3)]).is_satisfied(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(3), None);
        let y = problem.add_variable(dom(3), None);
        Gcc::new(vec![x, y, x], vec![(1, 0, 2)]);
    }

    #[test]
    #[should_panic(expected = "two bounds")]
    fn two_bounds_for_one_value_are_rejected() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(3), None);
        Gcc::new(vars, vec![(1, 0, 2), (2, 0, 1), (1, 1, 3)]);
    }

    // ----------------------------------------------------------------------------------------
    // Scope and ordering
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_name_and_structural_key() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(5), None);
        // Declared out of order: the key sorts them by value.
        let constraint = Gcc::new(vars.clone(), vec![(2, 0, 1), (0, 1, 1)]);
        assert_eq!(constraint.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(constraint.name(), "GCC");
        assert_eq!(
            constraint.structural_key(&problem),
            ConstraintShapeKey::Gcc {
                arity: 3,
                bounds: vec![(0, 1, 1), (2, 0, 1)]
            }
        );
    }

    #[test]
    fn the_structural_key_does_not_depend_on_the_order_of_the_bounds() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(5), None);
        let a = Gcc::new(vars.clone(), vec![(0, 1, 1), (2, 0, 1), (4, 0, 2)]);
        let b = Gcc::new(vars.clone(), vec![(4, 0, 2), (0, 1, 1), (2, 0, 1)]);
        let different_bound = Gcc::new(vars.clone(), vec![(0, 1, 1), (2, 0, 1), (4, 0, 3)]);
        let different_value = Gcc::new(vars, vec![(0, 1, 1), (2, 0, 1), (3, 0, 2)]);
        assert_eq!(a.structural_key(&problem), b.structural_key(&problem));
        assert_ne!(
            a.structural_key(&problem),
            different_bound.structural_key(&problem)
        );
        assert_ne!(
            a.structural_key(&problem),
            different_value.structural_key(&problem)
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, dom(2), None);
        let mut constraint = Gcc::new(vec![vars[0], vars[2], vars[3]], vec![(1, 0, 1)]);
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
        let mut constraint = Gcc::new(scope, vec![(1, 0, 2)]);
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
        let mut constraint = Gcc::new(vec![vars[0], vars[1]], vec![(1, 0, 1)]);
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
        Gcc::new(vars, vec![(1, 0, 1)]).is_layer_in_scope(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn update_variable_ordering_panics_when_a_scope_variable_is_missing() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        let mut constraint = Gcc::new(vars.clone(), vec![(1, 0, 1)]);
        // `vars[1]` does not appear in the order.
        constraint.update_variable_ordering(&[vars[0]]);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn empty_property_is_the_zero_interval_of_every_value() {
        let constraint = gcc_of(4, &[(1, 0, 2), (2, 1, 3)]);
        let empty = constraint.empty_property();
        assert_eq!(bounds_of(&*empty, 1), (0, 0));
        assert_eq!(bounds_of(&*empty, 2), (0, 0));
        // The key lists the minima of all the values, then their maxima.
        assert_eq!(empty.order_key(), vec![0.0; 4]);
    }

    #[test]
    fn identity_property_holds_the_no_path_sentinels() {
        let constraint = gcc_of(4, &[(1, 0, 2), (2, 1, 3)]);
        let identity = constraint.identity_property();
        assert_eq!(bounds_of(&*identity, 1), (usize::MAX, 0));
        assert_eq!(bounds_of(&*identity, 2), (usize::MAX, 0));
        assert!(!same(&*identity, &*constraint.empty_property()));
    }

    #[test]
    fn update_counts_only_the_slot_of_the_value() {
        let constraint = gcc_of(6, &[(1, 0, 3), (2, 0, 3)]);
        let one = path_property(&constraint, &[1]);
        assert_eq!(bounds_of(&*one, 1), (1, 1));
        assert_eq!(bounds_of(&*one, 2), (0, 0));
        let mixed = path_property(&constraint, &[1, 2, 2]);
        assert_eq!(bounds_of(&*mixed, 1), (1, 1));
        assert_eq!(bounds_of(&*mixed, 2), (2, 2));
    }

    #[test]
    fn update_with_an_unbounded_value_copies_the_parent() {
        let constraint = gcc_of(6, &[(1, 0, 3), (2, 0, 3)]);
        let parent = path_property(&constraint, &[1, 2]);
        let mut child = constraint.identity_property();
        child.update(&*parent, 0, true);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let constraint = gcc_of(6, &[(1, 0, 3), (2, 0, 3)]);
        let parent = path_property(&constraint, &[1, 2]);
        let mut child = constraint.identity_property();
        // The value 1 is bounded but the layer is out of scope: nothing is counted.
        child.update(&*parent, 1, false);
        assert!(same(&*child, &*parent));
    }

    #[test]
    fn a_value_whose_upper_bound_can_be_reached_keeps_exact_counts() {
        // Four variables, `hi` 3 < 4: every count up to 3 is kept.
        let constraint = gcc_of(4, &[(1, 2, 3)]);
        assert_eq!(bounds_of(&*path_property(&constraint, &[1; 3]), 1), (3, 3));
    }

    #[test]
    fn a_value_whose_upper_bound_can_never_be_reached_is_capped_at_its_lower_bound() {
        // Three variables, `hi` 3 >= 3: the upper bound can never bind, counts stop at `lo` = 2.
        let constraint = gcc_of(3, &[(1, 2, 3)]);
        assert_eq!(bounds_of(&*path_property(&constraint, &[1]), 1), (1, 1));
        assert_eq!(bounds_of(&*path_property(&constraint, &[1, 1]), 1), (2, 2));
        assert_eq!(
            bounds_of(&*path_property(&constraint, &[1, 1, 1]), 1),
            (2, 2)
        );
    }

    #[test]
    fn the_cap_applies_to_each_value_on_its_own() {
        // Value 1 saturates (hi 3 >= 3), value 2 does not (hi 1 < 3).
        let constraint = gcc_of(3, &[(1, 1, 3), (2, 0, 1)]);
        let property = path_property(&constraint, &[1, 1, 2, 2]);
        assert_eq!(bounds_of(&*property, 1), (1, 1));
        assert_eq!(bounds_of(&*property, 2), (2, 2));
    }

    #[test]
    fn folding_two_parents_takes_the_extreme_counts_of_each_value() {
        let constraint = helper();
        // Parent A has seen one 1, parent B one 2; both reach the node with an edge of value 1.
        let mut node = constraint.identity_property();
        node.update(&*path_property(&constraint, &[1]), 1, true);
        node.update(&*path_property(&constraint, &[2]), 1, true);
        assert_eq!(bounds_of(&*node, 1), (1, 2));
        assert_eq!(bounds_of(&*node, 2), (0, 1));
    }

    #[test]
    fn folding_a_wide_interval_shifts_both_ends() {
        let constraint = helper();
        let mut node = constraint.identity_property();
        node.update(&*interval(&constraint, 1, 0, 1), 1, true);
        node.update(&*interval(&constraint, 1, 2, 3), 1, true);
        assert_eq!(bounds_of(&*node, 1), (1, 4));
    }

    #[test]
    fn folding_does_not_depend_on_the_order_of_the_parents() {
        let constraint = helper();
        let left = path_property(&constraint, &[1, 1]);
        let right = path_property(&constraint, &[2]);
        let mut lr = constraint.identity_property();
        lr.update(&*left, 2, true);
        lr.update(&*right, 2, true);
        let mut rl = constraint.identity_property();
        rl.update(&*right, 2, true);
        rl.update(&*left, 2, true);
        assert!(same(&*lr, &*rl));
    }

    #[test]
    fn a_parent_without_paths_stays_without_paths() {
        // Folding the identity must neither overflow nor turn the sentinel maximum 0 into a count.
        let constraint = helper();
        let mut node = constraint.identity_property();
        node.update(&*constraint.identity_property(), 1, true);
        assert!(same(&*node, &*constraint.identity_property()));
        // And it does not disturb a real parent folded in afterwards.
        node.update(&*path_property(&constraint, &[2]), 1, true);
        assert_eq!(bounds_of(&*node, 1), (1, 1));
        assert_eq!(bounds_of(&*node, 2), (1, 1));
    }

    #[test]
    fn merge_takes_the_smallest_min_and_the_largest_max_of_each_value() {
        let constraint = helper();
        let mut a = path_property(&constraint, &[1, 1, 2]);
        a.merge(&*path_property(&constraint, &[1, 2, 2]));
        assert_eq!(bounds_of(&*a, 1), (1, 2));
        assert_eq!(bounds_of(&*a, 2), (1, 2));
        assert_eq!(a.order_key(), vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let constraint = helper();
        let a = path_property(&constraint, &[1, 2]);
        let b = interval(&constraint, 2, 0, 3);

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
    fn eq_and_hash_compare_every_value() {
        let constraint = helper();
        let a = path_property(&constraint, &[1, 2]);
        assert!(same(&*a, &*path_property(&constraint, &[2, 1])));
        assert_eq!(hash_of(&*a), hash_of(&*path_property(&constraint, &[2, 1])));
        // One interval differs, in either value, at either end.
        for other in [
            path_property(&constraint, &[1]),
            path_property(&constraint, &[2]),
            interval(&constraint, 1, 1, 2),
            {
                let mut p = path_property(&constraint, &[1, 2]);
                p.merge(&*path_property(&constraint, &[1, 2, 2]));
                p
            },
        ] {
            assert!(!same(&*a, &*other));
            assert_ne!(hash_of(&*a), hash_of(&*other));
        }
    }

    #[test]
    fn a_gcc_without_bounds_has_a_single_state() {
        let constraint = gcc_of(3, &[]);
        let identity = constraint.identity_property();
        let empty = constraint.empty_property();
        assert!(same(&*identity, &*empty));
        let mut node = constraint.identity_property();
        node.update(&*empty, 1, true);
        assert!(same(&*node, &*empty));
        for value in 0..3 {
            assert!(!constraint.is_assignment_invalid(&*empty, &*empty, 1, value));
        }
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        helper().identity_property().update(&*foreign(), 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        helper().identity_property().merge(&*foreign());
    }

    #[test]
    #[should_panic(expected = "Calling eq on property")]
    fn eq_with_a_property_of_another_constraint_is_rejected() {
        same(&*helper().identity_property(), &*foreign());
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_bounds_of_a_value_are_inclusive() {
        // Value 1 on a parent [1, 2] and a child [0, 1]: an edge of value 1 gives the counts
        // [2, 4], an edge of another value gives [1, 3]. The second bounded value, 2, stays at 0.
        let helper = helper();
        let parent = interval(&helper, 1, 1, 2);
        let child = interval(&helper, 1, 0, 1);
        // (lo, hi, edge value, invalid?)
        let cases = [
            (5, 5, 1, true),  // max 4 < lo 5
            (4, 4, 1, false), // max 4 == lo 4
            (0, 2, 1, false), // min 2 == hi 2
            (0, 1, 1, true),  // min 2 > hi 1
            (4, 4, 0, true),  // max 3 < lo 4
            (3, 3, 0, false), // max 3 == lo 3
            (0, 1, 0, false), // min 1 == hi 1
            (0, 0, 0, true),  // min 1 > hi 0
        ];
        for (lo, hi, value, invalid) in cases {
            let constraint = gcc_of(10, &[(1, lo, hi), (2, 0, 3)]);
            assert_eq!(
                constraint.is_assignment_invalid(&*parent, &*child, 1, value),
                invalid,
                "bounds [{lo}, {hi}] edge value {value}"
            );
        }
    }

    #[test]
    fn every_bounded_value_is_checked() {
        let helper = helper();
        let nothing = helper.empty_property();
        // Value 1 is fine (lo 0), value 2 needs 3 occurrences that no path brings.
        let needs_two = gcc_of(10, &[(1, 0, 3), (2, 3, 3)]);
        assert!(needs_two.is_assignment_invalid(&*nothing, &*nothing, 1, 1));
        // Swapped roles.
        let needs_one = gcc_of(10, &[(1, 3, 3), (2, 0, 3)]);
        assert!(needs_one.is_assignment_invalid(&*nothing, &*nothing, 1, 2));
        // Both are met.
        let fine = gcc_of(10, &[(1, 0, 3), (2, 0, 3)]);
        assert!(!fine.is_assignment_invalid(&*nothing, &*nothing, 1, 1));
    }

    #[test]
    fn an_edge_with_an_unbounded_value_can_be_invalid() {
        // The edge adds nothing to the count of 1, which is still short of its lower bound.
        let constraint = gcc_of(10, &[(1, 2, 3), (2, 0, 3)]);
        let nothing = constraint.empty_property();
        assert!(constraint.is_assignment_invalid(&*nothing, &*nothing, 1, 0));
        // With one 1 on each side the same edge is fine.
        let one = path_property(&constraint, &[1]);
        assert!(!constraint.is_assignment_invalid(&*one, &*one, 1, 0));
    }

    #[test]
    fn only_the_value_of_the_edge_is_shifted() {
        // Each of 1 and 2 occurs at most once; the parent has already seen one 1.
        let constraint = gcc_of(10, &[(1, 0, 1), (2, 0, 1)]);
        let parent = path_property(&constraint, &[1]);
        let child = constraint.empty_property();
        // A second 1 is too many, a 2 is fine, an unbounded value is fine.
        assert!(constraint.is_assignment_invalid(&*parent, &*child, 1, 1));
        assert!(!constraint.is_assignment_invalid(&*parent, &*child, 1, 2));
        assert!(!constraint.is_assignment_invalid(&*parent, &*child, 1, 0));
    }

    #[test]
    fn the_parent_and_the_child_both_count() {
        let constraint = gcc_of(10, &[(1, 2, 2), (2, 0, 3)]);
        let one = path_property(&constraint, &[1]);
        let none = constraint.empty_property();
        // Parent only: 1 + 1 = 2 with an edge of value 1. Child only: the same.
        assert!(!constraint.is_assignment_invalid(&*one, &*none, 1, 1));
        assert!(!constraint.is_assignment_invalid(&*none, &*one, 1, 1));
        // Both: 1 + 1 + 1 = 3 > hi.
        assert!(constraint.is_assignment_invalid(&*one, &*one, 1, 1));
    }

    #[test]
    fn an_edge_next_to_a_node_without_paths_is_invalid() {
        // Such a node is about to be removed; the sentinels must not be added to anything.
        let constraint = helper();
        let identity = constraint.identity_property();
        let real = path_property(&constraint, &[1]);
        assert!(constraint.is_assignment_invalid(&*identity, &*identity, 0, 0));
        assert!(constraint.is_assignment_invalid(&*identity, &*real, 0, 1));
        assert!(constraint.is_assignment_invalid(&*real, &*identity, 0, 1));
    }

    #[test]
    #[should_panic(expected = "instead of GccProperty")]
    fn a_property_of_another_constraint_is_rejected() {
        let constraint = helper();
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
        let cases: Vec<Case> = vec![
            // All different, with three values.
            (
                vec![dom(3), dom(3), dom(3)],
                vec![(0, 0, 1), (1, 0, 1), (2, 0, 1)],
            ),
            // Exactly twice.
            (vec![dom(2); 4], vec![(1, 2, 2)]),
            // A range and an upper bound.
            (vec![dom(3), dom(3), dom(3)], vec![(0, 1, 2), (2, 0, 1)]),
            // Nothing binds.
            (vec![dom(2), dom(2), dom(2)], vec![(1, 0, 3)]),
            // Two values exactly once, the others free.
            (
                vec![dom(3), dom(3), dom(3), dom(3)],
                vec![(0, 1, 1), (1, 1, 1)],
            ),
            // Negative values and domains that differ.
            (
                vec![vec![-1, 4], vec![4, 7], vec![-1, 7]],
                vec![(4, 1, 1), (-1, 0, 1)],
            ),
            (vec![dom(3), vec![5, 6], dom(3)], vec![(5, 1, 2), (0, 1, 1)]),
            // Lower bounds only, so that every count saturates.
            (vec![dom(3); 4], vec![(0, 1, 4), (1, 1, 4)]),
        ];
        for (domains, bounds) in cases {
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            for order in permutations(n) {
                let (problem, _) = full(&domains, &bounds);
                let mdd = settled(problem, order.clone());
                assert_eq!(
                    accepted(&mdd),
                    expected(&domains, &[(&scope, &bounds)]),
                    "domains {domains:?} bounds {bounds:?} order {order:?}"
                );
            }
        }
    }

    #[test]
    fn unreachable_bounds_are_unsat() {
        let cases: Vec<Case> = vec![
            // A value that no domain contains must occur.
            (vec![vec![0], vec![0]], vec![(1, 1, 2)]),
            // Both variables are forced to 1, which may occur once.
            (vec![vec![1], vec![1]], vec![(1, 0, 1)]),
            // Two values must each occur twice among three variables: this one is only seen
            // once the MDD is refined, because the values are bounded on their own.
            (vec![dom(2); 3], vec![(0, 2, 3), (1, 2, 3)]),
            // lo above hi.
            (vec![dom(2); 2], vec![(1, 2, 1)]),
        ];
        for (domains, bounds) in cases {
            let (problem, _) = full(&domains, &bounds);
            let mdd = settled(problem, (0..domains.len()).collect());
            assert!(accepted(&mdd).is_empty(), "{domains:?} {bounds:?}");
        }
    }

    #[test]
    fn a_value_that_cannot_occur_may_still_be_capped_at_zero() {
        let domains = [dom(2), dom(2)];
        let (problem, _) = full(&domains, &[(9, 0, 0)]);
        assert_eq!(accepted(&settled(problem, vec![0, 1])).len(), 4);
    }

    #[test]
    fn no_bound_accepts_every_tuple() {
        let domains = [dom(2), dom(3)];
        let (problem, _) = full(&domains, &[]);
        assert_eq!(accepted(&settled(problem, vec![0, 1])).len(), 6);
    }

    #[test]
    fn a_single_bound_is_an_among_on_that_value() {
        let cases: Vec<(Vec<Vec<isize>>, isize, usize, usize)> = vec![
            (vec![dom(3); 3], 1, 1, 2),
            (vec![dom(2); 4], 1, 2, 2),
            (vec![dom(3); 3], 2, 0, 1),
            (vec![dom(3); 3], 0, 2, 3),
        ];
        for (domains, value, lo, hi) in cases {
            let (problem, _) = full(&domains, &[(value, lo, hi)]);
            let from_gcc = accepted(&settled(problem, (0..domains.len()).collect()));

            let mut problem = Problem::default();
            let vars: Vec<VariableIndex> = domains
                .iter()
                .map(|d| problem.add_variable(d.clone(), None))
                .collect();
            among(&mut problem, vars, vec![value], lo, hi);
            let from_among = accepted(&settled(problem, (0..domains.len()).collect()));

            assert_eq!(from_gcc, from_among, "value {value} in [{lo}, {hi}]");
        }
    }

    #[test]
    fn at_most_one_of_each_value_is_all_different() {
        let domains = [dom(4), dom(4), dom(4)];
        let bounds: Vec<(isize, usize, usize)> = (0..4).map(|v| (v, 0, 1)).collect();
        let (problem, _) = full(&domains, &bounds);
        let from_gcc = accepted(&settled(problem, vec![0, 1, 2]));

        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        all_different(&mut problem, vars);
        let from_all_different = accepted(&settled(problem, vec![0, 1, 2]));

        assert_eq!(from_gcc.len(), 24);
        assert_eq!(from_gcc, from_all_different);
    }

    /// Two Gcc constraints on one variable per domain.
    fn two_scopes(domains: &[Vec<isize>], first: Spec, second: Spec) -> Problem {
        let (mut problem, vars) = scoped(domains, first.0, first.1);
        gcc(
            &mut problem,
            second.0.iter().map(|&i| vars[i]).collect(),
            second.1.to_vec(),
        );
        problem
    }

    #[test]
    fn a_constraint_ignores_the_layers_of_the_other_one() {
        let domains = [dom(3), dom(3), dom(3), dom(3)];
        let first: Spec = (&[0, 2], &[(1, 1, 1)]);
        let second: Spec = (&[1, 3], &[(2, 1, 2), (0, 0, 1)]);
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
        let first: Spec = (&[0, 1], &[(1, 1, 1), (2, 0, 1)]);
        let second: Spec = (&[1, 2], &[(0, 1, 2)]);
        let problem = two_scopes(&domains, first, second);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want = expected(&domains, &[first, second]);
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn combined_with_not_equals_is_exact() {
        let domains = [dom(3), dom(3), dom(3)];
        let bounds = [(2, 1, 1)];
        let (mut problem, vars) = full(&domains, &bounds);
        not_equals(&mut problem, vars[0], vars[1]);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want: Vec<Vec<isize>> = expected(&domains, &[(&[0, 1, 2], &bounds)])
            .into_iter()
            .filter(|t| t[0] != t[1])
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn a_shift_cover_like_the_one_of_the_nurse_models_is_exact() {
        // Five days, the shifts 1 to 3 and 0 for a day off: each shift is bounded, the day off is
        // free.
        let domains = vec![dom(4); 5];
        let bounds = [(1, 1, 2), (2, 0, 2), (3, 1, 3)];
        let (problem, _) = full(&domains, &bounds);
        let mdd = settled(problem, (0..5).collect());
        let scope: Vec<usize> = (0..5).collect();
        assert_eq!(accepted(&mdd), expected(&domains, &[(&scope, &bounds)]));
    }

    #[test]
    fn a_saturated_value_needs_at_most_lo_plus_one_nodes_per_layer() {
        // Six binary variables, the value 1 occurs at least twice: its upper bound is 6, which can
        // never bind, so a layer only needs the counts 0, 1 and "2 or more".
        let domains = vec![dom(2); 6];
        let (problem, _) = full(&domains, &[(1, 2, 6)]);
        let mdd = settled(problem, (0..6).collect());
        let sizes: Vec<usize> = (0..=6).map(|l| mdd.number_nodes_in_layer(l)).collect();
        assert_eq!(sizes, vec![1, 2, 3, 3, 3, 2, 1]);
    }

    #[test]
    fn a_value_whose_upper_bound_can_bind_keeps_every_count() {
        // The same six variables with the value 1 at most three times: the counts 0 to 3 all
        // matter.
        let domains = vec![dom(2); 6];
        let (problem, _) = full(&domains, &[(1, 0, 3)]);
        let mdd = settled(problem, (0..6).collect());
        let widest = (0..=6).map(|l| mdd.number_nodes_in_layer(l)).max().unwrap();
        assert_eq!(widest, 4);
    }

    #[test]
    fn a_saturated_lower_bound_compiles_smaller_than_the_same_among() {
        let domains = vec![dom(2); 6];
        let (problem, _) = full(&domains, &[(1, 2, 6)]);
        let compact = settled(problem, (0..6).collect());

        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        // An Among with a lower bound only is a Gcc with that bound, uncapped.
        among(&mut problem, vars, vec![1], 2, 7);
        let verbose = settled(problem, (0..6).collect());

        assert_eq!(accepted(&compact), accepted(&verbose));
        let nodes = |mdd: &Mdd| -> usize { (0..=6).map(|l| mdd.number_nodes_in_layer(l)).sum() };
        assert!(nodes(&compact) < nodes(&verbose));
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

    /// A random instance: domains over {0, ..., 3} and, for about half of the four values, a
    /// bound whose upper end may exceed the number of variables (so that saturating and
    /// binding bounds both show up) and whose lower end may be unreachable.
    fn random_instance(rng: &mut Lcg) -> Case {
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
        let mut bounds = vec![];
        for value in 0..4isize {
            if rng.below(2) == 1 {
                let lo = rng.below(n as u64 + 1) as usize;
                let hi = lo + rng.below((n + 2 - lo) as u64) as usize;
                bounds.push((value, lo, hi));
            }
        }
        // Declaration order does not matter, so shuffle it.
        for i in (1..bounds.len()).rev() {
            bounds.swap(i, rng.below(i as u64 + 1) as usize);
        }
        (domains, bounds)
    }

    #[test]
    fn random_instances_are_exact_once_settled() {
        let mut rng = Lcg(2024);
        for _ in 0..200 {
            let (domains, bounds) = random_instance(&mut rng);
            let n = domains.len();
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let (problem, _) = full(&domains, &bounds);
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &[(&scope, &bounds)]),
                "domains {domains:?} bounds {bounds:?} order {order:?}"
            );
        }
    }

    #[test]
    fn the_initial_relaxation_never_loses_a_solution() {
        // Width 1 keeps the first relaxation, with no splitting and no merging.
        let mut rng = Lcg(77);
        for _ in 0..200 {
            let (domains, bounds) = random_instance(&mut rng);
            let n = domains.len();
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &bounds)]);
            let (problem, _) = full(&domains, &bounds);
            let got = accepted(&compile(problem, (0..n).collect(), 1));
            for solution in &want {
                assert!(
                    got.contains(solution),
                    "lost {solution:?}: domains {domains:?} bounds {bounds:?}"
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
            let (domains, bounds) = random_instance(&mut rng);
            let n = domains.len();
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let scope: Vec<usize> = (0..n).collect();
            let want = expected(&domains, &[(&scope, &bounds)]);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = full(&domains, &bounds);
                let mdd = compile(problem, order.clone(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {domains:?} bounds {bounds:?} order {order:?} width {width}"
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
        let (mut problem, vars) = full(&[dom(3), dom(3)], &[(1, 1, 1)]);
        problem[vars[0]].set_domain(vec![1]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![1, 0], vec![1, 2]]);
    }

    #[test]
    fn a_domain_that_grows_after_construction_is_fine() {
        // Only the bounds are stored, so a new value is simply counted.
        let (mut problem, vars) = full(&[dom(2), dom(2)], &[(9, 1, 1)]);
        problem[vars[0]].set_domain(vec![0, 9]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![9, 0], vec![9, 1]]);
    }
}
