pub mod all_different;
pub mod among;
pub mod gcc;
pub mod not_equals;
pub mod sum;

use deepsize::DeepSizeOf;
use std::any::Any;
use std::hash::Hasher;

use crate::mdd::*;
use crate::modelling::variable::Variable;
use crate::modelling::*;

pub use all_different::AllDifferent;
pub use among::Among;
pub use gcc::Gcc;
pub use not_equals::NotEquals;
pub use sum::Sum;

pub trait Constraint: DeepSizeOf {
    /// Initialise the data structures for constraint propagation (e.g., properties)
    fn init(&mut self, vars: &[Variable]);
    /// Update the variable ordering. Update the (optional) information for the constraint's
    /// propagator and store which layers are in the constraint scope.
    fn update_variable_ordering(&mut self, ordering: &[usize]);
    fn reset_property_top_down(&mut self, node: NodeIndex);
    /// Updates the top-down local property of the mdd
    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize);
    fn reset_property_bottom_up(&mut self, node: NodeIndex);
    /// Updates the bottom-up local property of the mdd
    fn update_property_bottom_up(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        assignment: isize,
    );
    /// Returns true if the layer is in the scope of the constraint
    fn is_layer_in_scope(&self, layer: usize) -> bool;
    /// Returns true if the assignment is invalid and the edge can be removed
    fn is_assignment_invalid(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        decision: VariableIndex,
        assignment: isize,
    ) -> bool;
    /// Adds a node in the given layer. Updates the properties of the constraints
    fn add_node_in_layer(&mut self, layer: usize);
    /// Returns an iterator on the constraint's scope
    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_>;
    /// Returns true if the constraint is satisfied by the assignment
    fn is_satisfied(&self, assignment: &[isize]) -> bool;
    fn hash_node_state(&self, node: NodeIndex, hasher: &mut dyn Hasher);
    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool;
    fn name(&self) -> &'static str;
    fn shrink_layers(&mut self, layers_size: &[usize]);
    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64>;
    fn as_any(&self) -> &dyn Any;
}
