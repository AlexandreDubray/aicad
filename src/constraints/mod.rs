//! Constraints, and the contract that lets the MDD compiler use them.
//!
//! This page is the reference for anyone who uses a constraint through its traits or adds a new
//! one. The meaning of each constraint and the way it is compiled are in its own module:
//! [`AllDifferent`], [`NotEquals`], [`Sum`], [`Among`], [`AtLeast`], [`Gcc`] and [`Regular`].
//!
//! # Meaning
//!
//! A constraint restricts the values taken by the variables of its *scope*. It is made of two
//! parts, one trait each:
//!
//! - [`Constraint`] is the constraint itself: its scope, the test
//!   [`is_satisfied`](Constraint::is_satisfied) that defines what it means, its structural key,
//!   and the filtering of the edges of an MDD.
//! - [`ConstraintProperty`] is the state that the compiler attaches to every node of the MDD for
//!   this constraint. It summarises the partial assignments (restricted to the scope) of the
//!   paths that reach the node, or that leave it. What is summarised, and how coarsely, is the
//!   whole design of a constraint: see the table [below](#the-constraints).
//!
//! [`is_satisfied`](Constraint::is_satisfied) is the specification. Everything else is an
//! implementation of it that can be wrong, and the tests of every constraint compare the compiled
//! MDD with it on small instances.
//!
//! # How a constraint is compiled
//!
//! ## Setup
//!
//! [`Mdd::new`](crate::mdd::Mdd::new) receives the constraints to compile.
//!
//! 1. The variables of the MDD are the union of the scopes. A variable that is in no scope has no
//!    layer, and a constraint with an empty scope is never seen by the compilation.
//! 2. The ordering heuristic chooses the order of the layers. It is then repaired so that every
//!    pair of [`precedence_edges`](Constraint::precedence_edges) holds, which is how
//!    [`Regular`] gets its variables in the order of its automaton.
//! 3. [`update_variable_ordering`](Constraint::update_variable_ordering) is called once on every
//!    constraint with the final order. This is when a constraint learns which layers are its own.
//!    [`is_layer_in_scope`](Constraint::is_layer_in_scope) is meaningless before.
//!
//! ## Properties
//!
//! Every node has, for every constraint, a *top-down* property that summarises the paths from the
//! root to the node and a *bottom-up* one for the paths from the node to the sink. Only the
//! layers in the scope of the constraint count; the others copy the property.
//!
//! - The top-down property of the root is [`empty_property`](Constraint::empty_property), and the
//!   bottom-up property of the sink is
//!   [`empty_property_backward`](Constraint::empty_property_backward): the summary of the empty
//!   path.
//! - Every other property is computed as a **fold**. It starts from
//!   [`identity_property`](Constraint::identity_property), the summary of no path at all, and
//!   absorbs one incident edge at a time: [`update`](ConstraintProperty::update) with the parent's
//!   property for the edges above a node, and
//!   [`update_backward`](ConstraintProperty::update_backward) with the child's for the edges
//!   below. `update` accumulates into `self`; it never overwrites. The edge value is part of the
//!   summary only if `in_scope` is true.
//!
//! ## Merging and collapsing
//!
//! To keep a layer under the width limit, the compiler merges two nodes: their properties are
//! combined with [`merge`](ConstraintProperty::merge), and the merged node keeps the edges of
//! both. The merged property must describe at least the paths of both, so a merged node accepts
//! more than the nodes it replaces: this is a relaxation.
//!
//! Two nodes whose top-down and bottom-up properties are equal for every constraint (compared with
//! [`eq`](ConstraintProperty::eq), after [`hash`](ConstraintProperty::hash)) are collapsed into
//! one without any loss. [`order_key`](ConstraintProperty::order_key) gives the coordinates used
//! by the similarity-based merge heuristic to decide which nodes to merge.
//!
//! ## Edge filtering
//!
//! Once the properties of a layer are up to date, every edge of a layer in the scope of the
//! constraint is given to
//! [`is_assignment_invalid`](Constraint::is_assignment_invalid), with the top-down property of its
//! parent, the bottom-up property of its child and its value. If the answer is true the edge is
//! removed, and so are the nodes that this leaves without parent or child.
//!
//! # Laws
//!
//! A constraint is correct if its properties obey the following. They are checked on every
//! constraint by the tests of this module, with no MDD involved.
//!
//! 1. **Merge is a join.** `merge` is idempotent, commutative and associative, and
//!    [`identity_property`](Constraint::identity_property) is neutral for it.
//! 2. **A fold is a merge.** Absorbing two edges with `update` (or `update_backward`) gives the
//!    same property, whatever the order, as merging the properties obtained from each edge alone.
//! 3. **Equality is faithful.** Equal properties have equal hashes, and filter the same edges, so
//!    collapsing them loses nothing. `eq` may be finer than the filtering needs, never coarser.
//! 4. **Filtering is sound.** For an assignment that satisfies the constraint,
//!    [`is_assignment_invalid`](Constraint::is_assignment_invalid) is never true on its edges,
//!    whether the properties come from that single path or from nodes obtained by merging any
//!    number of paths. It may be false on an edge that no solution uses: that is incompleteness,
//!    and the price of a relaxation.
//! 5. **Order keys have a fixed length** and hold finite numbers.
//!
//! A merged node can remove an edge that one of the nodes it replaces keeps, when that node holds
//! no solution at all: soundness is about solutions, not about every path. The tests also check
//! that, on the properties of a single path, every constraint removes an assignment that violates
//! it.
//!
//! # The constraints
//!
//! | Constraint | State of a node | Merge | Edge removed when |
//! |---|---|---|---|
//! | [`AllDifferent`] | two bitsets of values: taken on all paths, taken on some path | intersection, union | the value is taken on all paths; a Hall set is full |
//! | [`NotEquals`] | bitset of the values of `x` and `y` seen on the paths | union | the other side holds only that value |
//! | [`Sum`] | interval of the sums | smallest min, largest max | the target is out of the interval (bounds only) |
//! | [`Among`] | interval of the counts of values of a set | smallest min, largest max | the bounds are out of the interval (bounds only) |
//! | [`AtLeast`] | the same, capped at the lower bound | smallest min, largest max | the largest count cannot reach the lower bound |
//! | [`Gcc`] | one interval per bounded value, capped at its lower bound when its upper bound cannot bind | per value | a bound is out of its interval (bounds only) |
//! | [`Regular`] | bitset of automaton states | union | no state of the parent leads into the child |
//!
//! "Bounds only" means that a count that is inside the interval but not reachable is not
//! detected: a compiled MDD can keep violating paths until nodes are split enough for the
//! intervals to shrink to single values.
//!
//! # Design notes
//!
//! - **Properties are self-contained.** A property keeps an [`Arc`](std::sync::Arc) to the data it
//!   needs (a value map, a transition table), so it can be folded and merged without access to
//!   its constraint, and clones are cheap.
//! - **Domains are read at construction, or not at all.** [`AllDifferent`], [`NotEquals`] and
//!   [`Regular`] number the values of the domains of their scope when they are built; shrinking a
//!   domain afterwards is fine, growing one is not (see each module). The other constraints do not
//!   read domains.
//! - **Constructors panic on a malformed constraint** (a repeated variable, an automaton that
//!   leaves its own states), except [`AllDifferent`] and [`NotEquals`] on a repeated variable,
//!   which are accepted with a warning and have no solution: this lets a learned constraint on an
//!   UNSAT problem be added on the fly and make the compilation conclude.
//! - **[`name`](Constraint::name) is a label** of the logs and metrics: renaming a constraint
//!   changes them. [`as_any`](Constraint::as_any) exists so that code specialised to a kind of
//!   constraint (the ConsFormer loss) can downcast.
//! - **The set of constraints is closed.** [`ConstraintShapeKey`] is an enum that the arena
//!   matches exhaustively to rebuild a constraint from its key, so a constraint cannot be added
//!   from outside the crate: adding one means editing it, which the checklist below describes.
//!
//! # Adding a constraint
//!
//! 1. **Write the module** `src/constraints/<name>.rs`, with the `//!` sections of the existing
//!    ones: *Meaning*, *How it compiles* (*Property update*, *Node merging*, *Edge filtering*),
//!    *Design notes*, *Example*. [`among`](mod@among) is the shortest model to follow. Say what an empty
//!    scope, a repeated variable and a changed domain do.
//! 2. **Implement [`Constraint`]** and a private property type that implements
//!    [`ConstraintProperty`]. Let the constructor panic on input that cannot mean anything.
//! 3. **Declare it** here (`pub mod` and `pub use`).
//! 4. **Add a variant to [`ConstraintShapeKey`]** and to [`arity`](ConstraintShapeKey::arity), and
//!    return it from `structural_key`. The key must hold everything the compiled structure
//!    depends on and nothing else, in a canonical form (sort the sets).
//! 5. **Rebuild it in the arena**: add the case to `instantiate_constraint` in `mdd/arena.rs`;
//!    the compiler points at it, since the match is exhaustive.
//! 6. **Expose it**: a function in `modelling`, and the Python binding in `pyaicad` if it is
//!    part of the Python model.
//! 7. **Test it** against `is_satisfied` and a brute-force oracle: the compiled MDD must contain
//!    every solution at every width, and be exact once it is fully refined and propagated. Add
//!    an instance to `test_contract` at the end of this file, which checks the laws above.
//!
//! # Example
//!
//! The compiler's walk can be done by hand. Between two and three of three binary variables must
//! take the value 1; after `x1 = 1` and before `x3 = 0`, the middle variable must be 1.
//!
//! ```
//! use aicad::constraints::{Among, Constraint, ConstraintProperty};
//! use aicad::modelling::*;
//! use rustc_hash::FxHashSet;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1], None);
//! let mut among = Among::new(vars.clone(), FxHashSet::from_iter([1]), 2, 3);
//! // The layers are in the order of the variables.
//! among.update_variable_ordering(&vars);
//!
//! // The node reached by x1 = 1, from the root.
//! let mut above = among.identity_property();
//! above.update(&*among.empty_property(), 1, among.is_layer_in_scope(0));
//!
//! // The node left by x3 = 0, towards the sink.
//! let mut below = among.identity_property();
//! below.update_backward(&*among.empty_property_backward(), 0, among.is_layer_in_scope(2));
//!
//! // At layer 1, x2 = 0 would leave a single 1, x2 = 1 makes two.
//! assert!(among.is_assignment_invalid(&*above, &*below, 1, 0));
//! assert!(!among.is_assignment_invalid(&*above, &*below, 1, 1));
//! ```
pub mod all_different;
pub mod among;
pub mod at_least;
pub mod gcc;
pub mod not_equals;
pub mod regular;
pub mod sum;

use deepsize::DeepSizeOf;
use dyn_clone::DynClone;
use std::any::Any;
use std::hash::Hasher;

use crate::modelling::*;

pub use all_different::AllDifferent;
pub use among::Among;
pub use at_least::AtLeast;
pub use gcc::Gcc;
pub use not_equals::NotEquals;
pub use regular::Regular;
pub use sum::Sum;

/// A constraint on the variables of its scope, and the filtering it does on an MDD.
///
/// See the [module documentation](self) for the protocol that the compiler follows and the laws
/// that an implementation must obey.
pub trait Constraint: DeepSizeOf + DynClone + Send + Sync {
    /// Tells the constraint the order of the layers: `order[layer]` is the variable branched at
    /// that layer. Every variable of the scope appears in `order`. Called once, by `Mdd::new`,
    /// before any other method that depends on layers.
    fn update_variable_ordering(&mut self, order: &[VariableIndex]);
    /// Exact, hashable description of the structure that this constraint compiles to, used as
    /// the key of the arena that shares compiled structures.
    ///
    /// Two constraints with equal keys must compile to the same structure, so the key holds
    /// everything the structure depends on, in a canonical form (sorted sets, for instance), and
    /// nothing that is specific to the variables of the scope. Domains are not part of it: the
    /// arena supplies a nominal domain shared by all the variables.
    fn structural_key(&self, problem: &Problem) -> ConstraintShapeKey;
    /// Pairs `(before, after)` that the variable order must respect.
    ///
    /// Most constraints do not care about the relative order of their scope, and keep the default
    /// empty list. A constraint that reads its scope as a sequence (e.g., `Regular`) needs its variables
    /// to be branched in that sequence. `Mdd::new` collects the pairs of every constraint and
    /// repairs the order of the heuristic to satisfy them, before any
    /// [`update_variable_ordering`](Constraint::update_variable_ordering).
    fn precedence_edges(&self) -> Vec<(VariableIndex, VariableIndex)> {
        vec![]
    }
    /// True if the variable branched at `layer` is in the scope of the constraint.
    fn is_layer_in_scope(&self, layer: usize) -> bool;
    /// The variables of the scope, in the order in which the constraint reads them.
    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_>;
    /// True if the constraint holds on `assignment`, which gives a value for every variable of
    /// the problem, indexed by variable. This is the specification of the constraint.
    fn is_satisfied(&self, assignment: &[isize]) -> bool;
    /// Name of the kind of constraint, used as a label in logs and metrics.
    fn name(&self) -> &'static str;
    /// The constraint as [`Any`], to downcast to its concrete type.
    fn as_any(&self) -> &dyn Any;
    /// The summary of no path at all: the start of a fold, and the neutral element of
    /// [`merge`](ConstraintProperty::merge).
    fn identity_property(&self) -> Box<dyn ConstraintProperty>;
    /// The summary of the empty path, which is the top-down property of the root. Defaults to the
    /// identity, which is right when the empty path summarises to nothing.
    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        self.identity_property()
    }
    /// The summary of the empty path read from the sink, which is the bottom-up property of the
    /// sink. Defaults to [`empty_property`](Constraint::empty_property), which is right when the
    /// two directions are symmetric.
    fn empty_property_backward(&self) -> Box<dyn ConstraintProperty> {
        self.empty_property()
    }
    /// True if the edge of value `assignment` at `layer` can be removed, given the top-down
    /// property of its `parent` and the bottom-up property of its `child`. Only called for the
    /// layers in the scope. Must be sound; see the laws in the [module documentation](self).
    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        layer: usize,
        assignment: isize,
    ) -> bool;
}

// Needed so that a `Box<dyn Constraint>` can be cloned.
dyn_clone::clone_trait_object!(Constraint);

/// Exact, hashable description of a constraint's compiled structure: the key of the arena that
/// shares structures between constraints of the same shape in the learning based module.
///
/// A key holds what the structure depends on, apart from the domain (the arena supplies one
/// nominal domain) and the width limit (the arena adds it). Sets are sorted so that constraints
/// declared in a different order share a key. Every variant is rebuilt into a constraint by the
/// arena, which is why the set of constraints is closed; see *Adding a constraint* in the
/// [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintShapeKey {
    /// [`AllDifferent`] is defined by the size of its scope.
    AllDifferent {
        /// Number of variables in the scope.
        arity: usize,
    },
    /// [`NotEquals`] always has two variables.
    NotEquals,
    /// [`Among`].
    Among {
        /// Number of variables in the scope.
        arity: usize,
        /// The counted values, sorted.
        values: Vec<isize>,
        /// Lower bound of the count.
        lb: usize,
        /// Upper bound of the count.
        ub: usize,
    },
    /// [`AtLeast`].
    AtLeast {
        /// Number of variables in the scope.
        arity: usize,
        /// The counted values, sorted.
        values: Vec<isize>,
        /// Lower bound of the count.
        lb: usize,
    },
    /// [`Gcc`].
    Gcc {
        /// Number of variables in the scope.
        arity: usize,
        /// The `(value, lower bound, upper bound)` of every bounded value, sorted by value. The
        /// compiled structure does not depend on the order in which the bounds were declared
        /// (it only permutes the slots of the property), so sorting lets two declarations in a
        /// different order share a structure.
        bounds: Vec<(isize, usize, usize)>,
    },
    /// [`Regular`].
    Regular {
        /// Number of variables in the scope.
        arity: usize,
        /// The transition table of the automaton, `[state][symbol]`.
        transitions: Vec<Vec<Option<usize>>>,
        /// The initial state.
        initial_state: usize,
        /// The accepting states, sorted.
        accepting_states: Vec<usize>,
    },
    /// [`Sum`].
    Sum {
        /// Number of variables in the scope.
        arity: usize,
        /// The value of the sum.
        target: isize,
    },
}

impl ConstraintShapeKey {
    /// Returns the arity of the constraint represented by this shape
    pub fn arity(&self) -> usize {
        match self {
            Self::NotEquals => 2,
            Self::AllDifferent { arity }
            | Self::Among { arity, .. }
            | Self::AtLeast { arity, .. }
            | Self::Gcc { arity, .. }
            | Self::Regular { arity, .. }
            | Self::Sum { arity, .. } => *arity,
        }
    }
}

/// The state that the compiler keeps on a node of the MDD for one constraint: a summary of the
/// partial assignments of the paths that reach the node (top-down) or leave it (bottom-up),
/// restricted to the scope of the constraint.
///
/// See the [module documentation](self) for how properties are folded, merged and compared, and
/// for the laws that they must obey.
pub trait ConstraintProperty: DeepSizeOf + DynClone + Send + Sync {
    /// Absorbs into `self` one edge above the node: `other` is the top-down property of the
    /// parent and `assignment` the value of the edge. The value is part of the summary only if
    /// `in_scope`, which tells whether the layer of the parent is in the scope; otherwise the
    /// edge adds nothing and `other` is absorbed as it is. Accumulates, never overwrites.
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool);
    /// Absorbs into `self` one edge below the node: `other` is the bottom-up property of the
    /// child. Defaults to [`update`](ConstraintProperty::update), which is right when the summary
    /// does not depend on the direction.
    fn update_backward(
        &mut self,
        other: &dyn ConstraintProperty,
        assignment: isize,
        in_scope: bool,
    ) {
        self.update(other, assignment, in_scope);
    }
    /// Replaces `self` by a property that describes at least the paths of both. A join:
    /// idempotent, commutative, associative, and neutral on the identity.
    fn merge(&mut self, other: &dyn ConstraintProperty);
    /// Feeds the hasher with what [`eq`](ConstraintProperty::eq) compares.
    fn hash(&self, hasher: &mut dyn Hasher);
    /// True if the two properties are the same state: they filter the same edges, and nodes that
    /// hold them can be collapsed. Panics if `other` is the property of another constraint.
    fn eq(&self, other: &dyn ConstraintProperty) -> bool;
    /// Coordinates that place this property in the state space of its constraint, used by
    /// `MergeHeuristic::StateSimilarity` to merge the nodes that are closest. Finite numbers, and
    /// always the same count for a given constraint. Properties that are close must be similar;
    /// equal ones do not reach the heuristic, since `collapse` merges them first.
    fn order_key(&self) -> Vec<f64>;
    /// The property as [`Any`], to downcast to its concrete type.
    fn as_any(&self) -> &dyn Any;
    /// Name of the kind of property, used in panic messages.
    fn name(&self) -> &'static str;
}

dyn_clone::clone_trait_object!(ConstraintProperty);

impl std::hash::Hash for dyn ConstraintProperty {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ConstraintProperty::hash(self, state)
    }
}

impl PartialEq for dyn ConstraintProperty {
    fn eq(&self, other: &Self) -> bool {
        ConstraintProperty::eq(self, other)
    }
}

impl Eq for dyn ConstraintProperty {}

#[cfg(test)]
mod test_contract {
    //! Tests of the contract itself, run on every constraint through the traits only. No MDD is
    //! built: the properties are walked by hand, the way the compiler walks them, on a few tiny
    //! instances. What is specific to one constraint is tested in its own module.

    use super::*;
    use deepsize::DeepSizeOf;
    use rustc_hash::{FxHashSet, FxHasher};
    use std::any::TypeId;

    // ----------------------------------------------------------------------------------------
    // Instances
    // ----------------------------------------------------------------------------------------

    /// A constraint on variables `0..domains.len()`, branched in this order (the variable `i` is
    /// at layer `i`), with its ordering already set.
    struct Instance {
        name: &'static str,
        domains: Vec<Vec<isize>>,
        constraint: Box<dyn Constraint>,
        /// `TypeId` of the concrete constraint, which `as_any` must give back.
        type_id: TypeId,
        /// Number of variables in the scope.
        scope_len: usize,
        /// Whether an assignment that violates the constraint is always removed somewhere along
        /// its path when the properties are exact (those of a single path).
        complete: bool,
    }

    impl Instance {
        fn new<C: Constraint + 'static>(
            name: &'static str,
            domains: Vec<Vec<isize>>,
            scope: &[usize],
            make: impl FnOnce(Vec<VariableIndex>, &Problem) -> C,
            complete: bool,
        ) -> Self {
            let mut problem = Problem::default();
            let vars: Vec<VariableIndex> = domains
                .iter()
                .map(|d| problem.add_variable(d.clone(), None))
                .collect();
            let mut constraint = make(scope.iter().map(|&i| vars[i]).collect(), &problem);
            constraint.update_variable_ordering(&vars);
            Self {
                name,
                domains,
                constraint: Box::new(constraint),
                type_id: TypeId::of::<C>(),
                scope_len: scope.len(),
                complete,
            }
        }

        fn layers(&self) -> usize {
            self.domains.len()
        }
    }

    fn doms(n: usize, d: isize) -> Vec<Vec<isize>> {
        vec![(0..d).collect(); n]
    }

    fn set(values: &[isize]) -> FxHashSet<isize> {
        values.iter().copied().collect()
    }

    /// The automaton "a 1 is never followed by a 0" over {0, 1, 2}.
    fn no_one_then_zero() -> Vec<Vec<Option<usize>>> {
        vec![
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
            vec![None, Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
        ]
    }

    /// One or two instances of each constraint, some with layers out of their scope.
    fn instances() -> Vec<Instance> {
        vec![
            Instance::new(
                "AllDifferent",
                doms(4, 4),
                &[0, 1, 2, 3],
                AllDifferent::new,
                true,
            ),
            Instance::new(
                "AllDifferent, partial scope",
                doms(5, 3),
                &[0, 1, 3],
                AllDifferent::new,
                true,
            ),
            Instance::new(
                "NotEquals",
                doms(4, 3),
                &[1, 3],
                |v, p| NotEquals::new(v[0], v[1], p),
                true,
            ),
            Instance::new(
                "Among",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, _| Among::new(v, set(&[1, 2]), 1, 2),
                true,
            ),
            Instance::new(
                "Among, partial scope",
                doms(5, 3),
                &[0, 2, 3, 4],
                |v, _| Among::new(v, set(&[1, 2]), 1, 2),
                true,
            ),
            Instance::new(
                "AtLeast",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, _| AtLeast::new(v, set(&[1]), 2),
                true,
            ),
            Instance::new(
                "AtLeast, partial scope",
                doms(5, 3),
                &[1, 2, 4],
                |v, _| AtLeast::new(v, set(&[0, 2]), 2),
                true,
            ),
            Instance::new(
                "Gcc",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, _| Gcc::new(v, vec![(0, 1, 2), (1, 0, 1), (2, 1, 3)]),
                true,
            ),
            Instance::new(
                "Gcc, saturating bounds",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, _| Gcc::new(v, vec![(0, 1, 4), (1, 0, 4)]),
                true,
            ),
            Instance::new(
                "Gcc, partial scope",
                doms(5, 3),
                &[0, 1, 3, 4],
                |v, _| Gcc::new(v, vec![(0, 1, 2), (1, 1, 1)]),
                true,
            ),
            Instance::new(
                "Sum",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, p| Sum::new(v, 4, p),
                true,
            ),
            Instance::new(
                "Sum, partial scope",
                doms(5, 3),
                &[0, 1, 3],
                |v, p| Sum::new(v, 3, p),
                true,
            ),
            Instance::new(
                "Regular",
                doms(4, 3),
                &[0, 1, 2, 3],
                |v, p| Regular::new(v, no_one_then_zero(), 0, FxHashSet::from_iter([0, 1, 2]), p),
                true,
            ),
            Instance::new(
                "Regular, partial scope",
                doms(5, 3),
                &[0, 2, 3],
                |v, p| Regular::new(v, no_one_then_zero(), 0, FxHashSet::from_iter([1, 2, 3]), p),
                true,
            ),
        ]
    }

    // ----------------------------------------------------------------------------------------
    // Walking the properties by hand
    // ----------------------------------------------------------------------------------------

    /// All the assignments of the instance.
    fn assignments(instance: &Instance) -> Vec<Vec<isize>> {
        let mut all: Vec<Vec<isize>> = vec![vec![]];
        for domain in &instance.domains {
            all = all
                .into_iter()
                .flat_map(|prefix| {
                    domain.iter().map(move |&v| {
                        let mut t = prefix.clone();
                        t.push(v);
                        t
                    })
                })
                .collect();
        }
        all
    }

    /// The top-down property of a node reached by the path `prefix`, from the root.
    fn top_down(constraint: &dyn Constraint, prefix: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = constraint.empty_property();
        for (layer, &value) in prefix.iter().enumerate() {
            let mut next = constraint.identity_property();
            next.update(&*property, value, constraint.is_layer_in_scope(layer));
            property = next;
        }
        property
    }

    /// The bottom-up property of the node at layer `from`, left by the path `t[from..]`, down to
    /// the sink of an MDD of `t.len()` layers.
    fn bottom_up(
        constraint: &dyn Constraint,
        from: usize,
        t: &[isize],
    ) -> Box<dyn ConstraintProperty> {
        let mut property = constraint.empty_property_backward();
        for layer in (from..t.len()).rev() {
            let mut next = constraint.identity_property();
            next.update_backward(&*property, t[layer], constraint.is_layer_in_scope(layer));
            property = next;
        }
        property
    }

    /// Whether the edge of value `t[layer]` is removed on the path `t`, with exact properties.
    fn removed_at(constraint: &dyn Constraint, t: &[isize], layer: usize) -> bool {
        let parent = top_down(constraint, &t[..layer]);
        let child = bottom_up(constraint, layer + 1, t);
        constraint.is_assignment_invalid(&*parent, &*child, layer, t[layer])
    }

    /// A small deterministic generator, to sample the pairs and triples of properties.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    /// For each layer `l`, the top-down properties of every prefix of length `l`, plus a few
    /// merges of two of them (relaxed nodes).
    fn top_down_pool(instance: &Instance, rng: &mut Lcg) -> Vec<Vec<Box<dyn ConstraintProperty>>> {
        let c = instance.constraint.as_ref();
        let all = assignments(instance);
        (0..instance.layers())
            .map(|l| {
                let mut pool: Vec<Box<dyn ConstraintProperty>> =
                    all.iter().map(|t| top_down(c, &t[..l])).collect();
                for _ in 0..20 {
                    let mut merged = pool[rng.below(pool.len())].clone();
                    merged.merge(&*pool[rng.below(pool.len())]);
                    pool.push(merged);
                }
                pool
            })
            .collect()
    }

    /// For each layer `l`, the bottom-up properties of every suffix starting at `l`, plus merges.
    fn bottom_up_pool(instance: &Instance, rng: &mut Lcg) -> Vec<Vec<Box<dyn ConstraintProperty>>> {
        let c = instance.constraint.as_ref();
        let all = assignments(instance);
        (0..=instance.layers())
            .map(|l| {
                let mut pool: Vec<Box<dyn ConstraintProperty>> =
                    all.iter().map(|t| bottom_up(c, l, t)).collect();
                for _ in 0..20 {
                    let mut merged = pool[rng.below(pool.len())].clone();
                    merged.merge(&*pool[rng.below(pool.len())]);
                    pool.push(merged);
                }
                pool
            })
            .collect()
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        use std::hash::Hasher;
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    fn merged(
        a: &(dyn ConstraintProperty + 'static),
        b: &dyn ConstraintProperty,
    ) -> Box<dyn ConstraintProperty> {
        let mut m = dyn_clone::clone_box(a);
        m.merge(b);
        m
    }

    // ----------------------------------------------------------------------------------------
    // Filtering is sound, and complete on exact properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn an_assignment_that_satisfies_the_constraint_is_never_removed() {
        for instance in instances() {
            let c = instance.constraint.as_ref();
            for t in assignments(&instance) {
                if !c.is_satisfied(&t) {
                    continue;
                }
                for layer in (0..instance.layers()).filter(|&l| c.is_layer_in_scope(l)) {
                    assert!(
                        !removed_at(c, &t, layer),
                        "{}: {t:?} satisfies the constraint but its edge at layer {layer} is removed",
                        instance.name
                    );
                }
            }
        }
    }

    #[test]
    fn an_assignment_that_violates_the_constraint_is_removed_somewhere() {
        for instance in instances().into_iter().filter(|i| i.complete) {
            let c = instance.constraint.as_ref();
            let mut violating = 0;
            for t in assignments(&instance) {
                if c.is_satisfied(&t) {
                    continue;
                }
                violating += 1;
                let removed = (0..instance.layers())
                    .filter(|&l| c.is_layer_in_scope(l))
                    .any(|l| removed_at(c, &t, l));
                assert!(
                    removed,
                    "{}: {t:?} violates the constraint and no edge of its path is removed",
                    instance.name
                );
            }
            assert!(violating > 0, "{}: nothing is violated", instance.name);
        }
    }

    #[test]
    fn the_instances_have_both_solutions_and_violations() {
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let all = assignments(&instance);
            let sat = all.iter().filter(|t| c.is_satisfied(t)).count();
            assert!(
                sat > 0 && sat < all.len(),
                "{}: {sat}/{}",
                instance.name,
                all.len()
            );
        }
    }

    // ----------------------------------------------------------------------------------------
    // Merging
    // ----------------------------------------------------------------------------------------

    #[test]
    fn merge_is_a_join() {
        let mut rng = Lcg(1);
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let identity = c.identity_property();
            for pool in top_down_pool(&instance, &mut rng)
                .into_iter()
                .chain(bottom_up_pool(&instance, &mut rng))
            {
                for _ in 0..200 {
                    let a = &pool[rng.below(pool.len())];
                    let b = &pool[rng.below(pool.len())];
                    let d = &pool[rng.below(pool.len())];
                    let name = instance.name;
                    assert!(same(&*merged(&**a, &**a), &**a), "{name}: not idempotent");
                    assert!(
                        same(&*merged(&**a, &**b), &*merged(&**b, &**a)),
                        "{name}: not commutative"
                    );
                    assert!(
                        same(
                            &*merged(&*merged(&**a, &**b), &**d),
                            &*merged(&**a, &*merged(&**b, &**d))
                        ),
                        "{name}: not associative"
                    );
                    assert!(
                        same(&*merged(&**a, &*identity), &**a),
                        "{name}: the identity is not neutral on the right"
                    );
                    assert!(
                        same(&*merged(&*identity, &**a), &**a),
                        "{name}: the identity is not neutral on the left"
                    );
                }
            }
        }
    }

    #[test]
    fn folding_two_parents_is_merging_the_two_folds() {
        let mut rng = Lcg(2);
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let td = top_down_pool(&instance, &mut rng);
            let bu = bottom_up_pool(&instance, &mut rng);
            for layer in 0..instance.layers() {
                for &value in &instance.domains[layer] {
                    for in_scope in [true, false] {
                        for (pool, backward) in [(&td[layer], false), (&bu[layer + 1], true)] {
                            for _ in 0..100 {
                                let a = &pool[rng.below(pool.len())];
                                let b = &pool[rng.below(pool.len())];
                                let fold = |others: &[&dyn ConstraintProperty]| {
                                    let mut p = c.identity_property();
                                    for other in others {
                                        if backward {
                                            p.update_backward(*other, value, in_scope);
                                        } else {
                                            p.update(*other, value, in_scope);
                                        }
                                    }
                                    p
                                };
                                let both = fold(&[&**a, &**b]);
                                let reversed = fold(&[&**b, &**a]);
                                let apart = merged(&*fold(&[&**a]), &*fold(&[&**b]));
                                let name = instance.name;
                                assert!(
                                    same(&*both, &*apart),
                                    "{name}: folding two parents differs from merging the folds (layer {layer}, value {value}, in scope {in_scope}, backward {backward})"
                                );
                                assert!(
                                    same(&*both, &*reversed),
                                    "{name}: the fold depends on the order of the parents"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_merged_node_never_removes_an_edge_of_a_satisfying_assignment() {
        // A merged node stands for several prefixes (top-down) and several suffixes (bottom-up).
        // Every satisfying assignment made of one of the prefixes, an edge and one of the
        // suffixes must keep that edge, whatever the number of nodes merged.
        let mut rng = Lcg(3);
        let mut checked = 0;
        let mut removed = 0;
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let all = assignments(&instance);
            for layer in (0..instance.layers()).filter(|&l| c.is_layer_in_scope(l)) {
                for _ in 0..400 {
                    let prefixes: Vec<&Vec<isize>> = (0..1 + rng.below(4))
                        .map(|_| &all[rng.below(all.len())])
                        .collect();
                    let suffixes: Vec<&Vec<isize>> = (0..1 + rng.below(4))
                        .map(|_| &all[rng.below(all.len())])
                        .collect();
                    let mut parent = top_down(c, &prefixes[0][..layer]);
                    for p in &prefixes[1..] {
                        parent.merge(&*top_down(c, &p[..layer]));
                    }
                    let mut child = bottom_up(c, layer + 1, suffixes[0]);
                    for s in &suffixes[1..] {
                        child.merge(&*bottom_up(c, layer + 1, s));
                    }
                    for &value in &instance.domains[layer] {
                        let invalid = c.is_assignment_invalid(&*parent, &*child, layer, value);
                        removed += usize::from(invalid);
                        for p in &prefixes {
                            for s in &suffixes {
                                let mut t = p[..layer].to_vec();
                                t.push(value);
                                t.extend_from_slice(&s[layer + 1..]);
                                if c.is_satisfied(&t) {
                                    checked += 1;
                                    assert!(
                                        !invalid,
                                        "{}: merging {} prefixes and {} suffixes removes the edge {value} of {t:?} at layer {layer}",
                                        instance.name,
                                        prefixes.len(),
                                        suffixes.len()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0 && removed > 0, "the law was never exercised");
    }

    // ----------------------------------------------------------------------------------------
    // Equality and hashing
    // ----------------------------------------------------------------------------------------

    #[test]
    fn equality_is_an_equivalence_and_agrees_with_the_hash() {
        let mut rng = Lcg(4);
        for instance in instances() {
            let pools = top_down_pool(&instance, &mut rng);
            for pool in pools {
                for a in &pool {
                    assert!(same(&**a, &**a), "{}: eq is not reflexive", instance.name);
                    let copy = a.clone();
                    assert!(same(&*copy, &**a) && hash_of(&*copy) == hash_of(&**a));
                }
                for _ in 0..300 {
                    let a = &pool[rng.below(pool.len())];
                    let b = &pool[rng.below(pool.len())];
                    assert_eq!(
                        same(&**a, &**b),
                        same(&**b, &**a),
                        "{}: eq is not symmetric",
                        instance.name
                    );
                    if same(&**a, &**b) {
                        assert_eq!(
                            hash_of(&**a),
                            hash_of(&**b),
                            "{}: equal properties hash differently",
                            instance.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dyn_properties_can_be_deduplicated_in_a_hash_set() {
        let mut rng = Lcg(5);
        for instance in instances() {
            let pool = &top_down_pool(&instance, &mut rng)[instance.layers() - 1];
            let mut classes: Vec<&Box<dyn ConstraintProperty>> = vec![];
            for p in pool {
                if !classes.iter().any(|q| same(&***q, &**p)) {
                    classes.push(p);
                }
            }
            let hashed: std::collections::HashSet<Box<dyn ConstraintProperty>> =
                pool.iter().cloned().collect();
            assert_eq!(hashed.len(), classes.len(), "{}", instance.name);
        }
    }

    #[test]
    fn equal_properties_filter_alike() {
        // `collapse` replaces nodes with equal properties by one, which is only sound if the
        // properties tell everything the filtering reads.
        let mut rng = Lcg(6);
        let mut pairs = 0;
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let td = top_down_pool(&instance, &mut rng);
            let bu = bottom_up_pool(&instance, &mut rng);
            for layer in (0..instance.layers()).filter(|&l| c.is_layer_in_scope(l)) {
                for (side, pool, other) in [
                    ("parent", &td[layer], &bu[layer + 1]),
                    ("child", &bu[layer + 1], &td[layer]),
                ] {
                    for (i, a) in pool.iter().enumerate() {
                        let Some(b) = pool[..i].iter().find(|b| same(&***b, &**a)) else {
                            continue;
                        };
                        pairs += 1;
                        for o in other.iter() {
                            for &value in &instance.domains[layer] {
                                let (ra, rb) = if side == "parent" {
                                    (
                                        c.is_assignment_invalid(&**a, &**o, layer, value),
                                        c.is_assignment_invalid(&**b, &**o, layer, value),
                                    )
                                } else {
                                    (
                                        c.is_assignment_invalid(&**o, &**a, layer, value),
                                        c.is_assignment_invalid(&**o, &**b, layer, value),
                                    )
                                };
                                assert_eq!(
                                    ra, rb,
                                    "{}: equal {side} properties filter differently (layer {layer}, value {value})",
                                    instance.name
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(pairs > 0);
    }

    #[test]
    fn a_clone_is_independent_of_the_original() {
        let mut rng = Lcg(7);
        for instance in instances() {
            let pool = &top_down_pool(&instance, &mut rng)[instance.layers() / 2];
            for _ in 0..100 {
                let a = &pool[rng.below(pool.len())];
                let b = &pool[rng.below(pool.len())];
                let snapshot = a.clone();
                let mut copy = a.clone();
                copy.merge(&**b);
                copy.update(&**b, instance.domains[0][0], true);
                assert!(
                    same(&**a, &*snapshot),
                    "{}: changing a clone changed the original",
                    instance.name
                );
            }
        }
    }

    #[test]
    fn order_keys_are_finite_and_have_a_fixed_length() {
        let mut rng = Lcg(8);
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let length = c.empty_property().order_key().len();
            for pool in top_down_pool(&instance, &mut rng)
                .into_iter()
                .chain(bottom_up_pool(&instance, &mut rng))
            {
                for p in pool {
                    let key = p.order_key();
                    assert_eq!(key.len(), length, "{}", instance.name);
                    assert!(
                        key.iter().all(|x| x.is_finite()),
                        "{}: {key:?}",
                        instance.name
                    );
                }
            }
            assert_eq!(
                c.identity_property().order_key().len(),
                length,
                "{}",
                instance.name
            );
        }
    }

    // ----------------------------------------------------------------------------------------
    // The constraint itself
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_key_has_the_arity_of_the_scope() {
        let mut problem = Problem::default();
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let key = c.structural_key(&problem);
            assert_eq!(key.arity(), instance.scope_len, "{}", instance.name);
            assert_eq!(key.arity(), c.iter_scope().count(), "{}", instance.name);
            // The key of a clone is the same key.
            assert_eq!(dyn_clone::clone_box(c).structural_key(&problem), key);
        }
        // `structural_key` receives a problem but these constraints do not need it to be filled.
        problem.add_variable(vec![0], None);
    }

    #[test]
    fn as_any_gives_back_the_concrete_constraint() {
        for instance in instances() {
            assert_eq!(
                instance.constraint.as_any().type_id(),
                instance.type_id,
                "{}",
                instance.name
            );
        }
    }

    #[test]
    fn every_kind_of_constraint_has_its_own_name() {
        let mut names: Vec<&'static str> =
            instances().iter().map(|i| i.constraint.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names,
            vec![
                "AllDifferent",
                "Among",
                "AtLeast",
                "GCC",
                "Not Equals",
                "Regular",
                "Sum"
            ]
        );
    }

    #[test]
    fn the_scope_is_what_is_satisfied_reads() {
        // Changing a variable outside the scope never changes the answer.
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let scope: Vec<usize> = c.iter_scope().map(|v| v.0).collect();
            for t in assignments(&instance) {
                for outside in (0..instance.layers()).filter(|i| !scope.contains(i)) {
                    for &other in &instance.domains[outside] {
                        let mut u = t.clone();
                        u[outside] = other;
                        assert_eq!(c.is_satisfied(&t), c.is_satisfied(&u), "{}", instance.name);
                    }
                }
            }
        }
    }

    #[test]
    fn layers_in_scope_are_the_layers_of_the_scope_variables() {
        for instance in instances() {
            let c = instance.constraint.as_ref();
            let scope: Vec<usize> = c.iter_scope().map(|v| v.0).collect();
            for layer in 0..instance.layers() {
                assert_eq!(
                    c.is_layer_in_scope(layer),
                    scope.contains(&layer),
                    "{}",
                    instance.name
                );
            }
        }
    }

    #[test]
    fn the_size_of_a_property_is_reported() {
        for instance in instances() {
            assert!(
                instance.constraint.identity_property().deep_size_of() > 0,
                "{}",
                instance.name
            );
        }
    }

    // ----------------------------------------------------------------------------------------
    // Defaults of the traits
    // ----------------------------------------------------------------------------------------

    /// A property that counts the calls to `update`, to see which method a default calls.
    #[derive(Clone, deepsize::DeepSizeOf)]
    struct Counter(usize);

    impl ConstraintProperty for Counter {
        fn update(&mut self, _: &dyn ConstraintProperty, _: isize, _: bool) {
            self.0 += 1;
        }
        fn merge(&mut self, _: &dyn ConstraintProperty) {}
        fn hash(&self, hasher: &mut dyn Hasher) {
            hasher.write_usize(self.0);
        }
        fn eq(&self, other: &dyn ConstraintProperty) -> bool {
            other
                .as_any()
                .downcast_ref::<Counter>()
                .is_some_and(|o| o.0 == self.0)
        }
        fn order_key(&self) -> Vec<f64> {
            vec![self.0 as f64]
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn name(&self) -> &'static str {
            "Counter"
        }
    }

    macro_rules! constraint_with_defaults {
        ($name:ident, { $($extra:item)* }) => {
            #[derive(Clone, deepsize::DeepSizeOf)]
            struct $name;

            impl Constraint for $name {
                fn update_variable_ordering(&mut self, _: &[VariableIndex]) {}
                fn structural_key(&self, _: &Problem) -> ConstraintShapeKey {
                    ConstraintShapeKey::NotEquals
                }
                fn is_layer_in_scope(&self, _: usize) -> bool {
                    true
                }
                fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
                    Box::new(std::iter::empty())
                }
                fn is_satisfied(&self, _: &[isize]) -> bool {
                    true
                }
                fn name(&self) -> &'static str {
                    stringify!($name)
                }
                fn as_any(&self) -> &dyn Any {
                    self
                }
                fn identity_property(&self) -> Box<dyn ConstraintProperty> {
                    Box::new(Counter(7))
                }
                fn is_assignment_invalid(&self, _: &dyn ConstraintProperty, _: &dyn ConstraintProperty, _: usize, _: isize) -> bool {
                    false
                }
                $($extra)*
            }
        };
    }

    constraint_with_defaults!(OnlyRequired, {});
    constraint_with_defaults!(WithEmpty, {
        fn empty_property(&self) -> Box<dyn ConstraintProperty> {
            Box::new(Counter(9))
        }
    });

    fn counter_of(property: &dyn ConstraintProperty) -> usize {
        property.as_any().downcast_ref::<Counter>().unwrap().0
    }

    #[test]
    fn by_default_there_is_no_precedence_requirement() {
        assert!(OnlyRequired.precedence_edges().is_empty());
    }

    #[test]
    fn by_default_the_root_and_the_sink_hold_the_identity() {
        assert_eq!(counter_of(&*OnlyRequired.empty_property()), 7);
        assert_eq!(counter_of(&*OnlyRequired.empty_property_backward()), 7);
    }

    #[test]
    fn by_default_the_sink_holds_the_property_of_the_root() {
        assert_eq!(counter_of(&*WithEmpty.empty_property()), 9);
        assert_eq!(counter_of(&*WithEmpty.empty_property_backward()), 9);
    }

    #[test]
    fn by_default_the_backward_update_is_the_update() {
        let mut property = Counter(0);
        property.update_backward(&Counter(0), 3, true);
        assert_eq!(property.0, 1);
    }

    // ----------------------------------------------------------------------------------------
    // The shape keys
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_arity_of_every_kind_of_key() {
        let keys = [
            (ConstraintShapeKey::AllDifferent { arity: 5 }, 5),
            (ConstraintShapeKey::NotEquals, 2),
            (
                ConstraintShapeKey::Among {
                    arity: 4,
                    values: vec![1],
                    lb: 0,
                    ub: 2,
                },
                4,
            ),
            (
                ConstraintShapeKey::AtLeast {
                    arity: 3,
                    values: vec![1],
                    lb: 1,
                },
                3,
            ),
            (
                ConstraintShapeKey::Gcc {
                    arity: 6,
                    bounds: vec![(1, 0, 2)],
                },
                6,
            ),
            (
                ConstraintShapeKey::Regular {
                    arity: 7,
                    transitions: vec![vec![Some(0)]],
                    initial_state: 0,
                    accepting_states: vec![0],
                },
                7,
            ),
            (
                ConstraintShapeKey::Sum {
                    arity: 8,
                    target: 3,
                },
                8,
            ),
        ];
        for (key, arity) in keys {
            assert_eq!(key.arity(), arity, "{key:?}");
        }
    }

    #[test]
    fn keys_of_different_kinds_are_never_equal() {
        let a = ConstraintShapeKey::AllDifferent { arity: 2 };
        let b = ConstraintShapeKey::Sum {
            arity: 2,
            target: 0,
        };
        assert_ne!(a, b);
        assert_ne!(a, ConstraintShapeKey::NotEquals);
    }
}
