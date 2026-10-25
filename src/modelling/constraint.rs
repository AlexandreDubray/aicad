//! The traits that a constraint implements, and the key that identifies the structure it compiles to.
//!
//! They live in `modelling` because a [`Problem`] stores constraints as `dyn Constraint`; the
//! concrete constraints and the contract that they obey are documented in
//! [`constraints`](crate::constraints), which re-exports everything here.

use super::{Problem, VariableIndex};
use deepsize::DeepSizeOf;
use dyn_clone::DynClone;
use std::any::Any;
use std::hash::Hasher;

/// A constraint on the variables of its scope, and the filtering it does on an MDD.
///
/// See the [module documentation](crate::constraints) for the protocol that the compiler follows and the laws
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
    /// layers in the scope. Must be sound; see the laws in the [module documentation](crate::constraints).
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
/// [module documentation](crate::constraints).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintShapeKey {
    /// [`AllDifferent`](crate::constraints::AllDifferent) is defined by the size of its scope.
    AllDifferent {
        /// Number of variables in the scope.
        arity: usize,
    },
    /// [`NotEquals`](crate::constraints::NotEquals) always has two variables.
    NotEquals,
    /// [`Among`](crate::constraints::Among).
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
    /// [`AtLeast`](crate::constraints::AtLeast).
    AtLeast {
        /// Number of variables in the scope.
        arity: usize,
        /// The counted values, sorted.
        values: Vec<isize>,
        /// Lower bound of the count.
        lb: usize,
    },
    /// [`Gcc`](crate::constraints::Gcc).
    Gcc {
        /// Number of variables in the scope.
        arity: usize,
        /// The `(value, lower bound, upper bound)` of every bounded value, sorted by value. The
        /// compiled structure does not depend on the order in which the bounds were declared
        /// (it only permutes the slots of the property), so sorting lets two declarations in a
        /// different order share a structure.
        bounds: Vec<(isize, usize, usize)>,
    },
    /// [`Regular`](crate::constraints::Regular).
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
    /// [`Sum`](crate::constraints::Sum).
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
/// See the [module documentation](crate::constraints) for how properties are folded, merged and compared, and
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
    /// True if the property describes a single state of the constraint: all the paths it
    /// summarises lead to the same state. A property in a node resulting from non-interchangeable
    /// paths must return false.
    fn is_single_state(&self) -> bool;
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
