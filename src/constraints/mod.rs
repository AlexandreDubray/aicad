pub mod all_different;
pub mod among;
pub mod gcc;
pub mod not_equals;
pub mod sum;

use deepsize::DeepSizeOf;
use dyn_clone::DynClone;
use std::any::Any;
use std::hash::Hasher;

use crate::mdd::*;
use crate::modelling::*;

pub use all_different::AllDifferent;
pub use among::Among;
pub use gcc::Gcc;
pub use not_equals::NotEquals;
pub use sum::Sum;

pub trait Constraint: DeepSizeOf + DynClone + Send + Sync {
    /// Update the variable ordering. `order[layer]` gives the variable branched at that layer;
    /// every variable in the constraint's own scope is guaranteed to appear in `order`.
    fn update_variable_ordering(&mut self, order: &[VariableIndex]);
    /// Returns true if the layer is in the scope of the constraint
    fn is_layer_in_scope(&self, layer: usize) -> bool;
    /// Returns an iterator on the constraint's scope
    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_>;
    /// Returns true if the constraint is satisfied by the assignment
    fn is_satisfied(&self, assignment: &[isize]) -> bool;
    fn name(&self) -> &'static str;
    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64>;
    fn as_any(&self) -> &dyn Any;
    fn identity_property(&self) -> Box<dyn ConstraintProperty>;
    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        self.identity_property()
    }
    /// Returns true if the assignment is invalid and the edge can be removed
    fn is_assignment_invalid(
        &self,
        node: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        layer: usize,
        assignment: isize,
    ) -> bool;
}

dyn_clone::clone_trait_object!(Constraint);

pub trait ConstraintProperty: DeepSizeOf + DynClone + Send + Sync {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool);
    fn hash(&self, hasher: &mut dyn Hasher);
    fn eq(&self, other: &dyn ConstraintProperty) -> bool;
    fn as_any(&self) -> &dyn Any;
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
