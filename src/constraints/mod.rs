pub mod all_different;
pub mod not_equals;

use crate::mdd::*;
use crate::modelling::VariableIndex;

pub use all_different::AllDifferent;
pub use not_equals::NotEquals;

pub trait Constraint {
    /// Update the variable ordering. Update the (optional) information for the constraint's
    /// propagator and store which layers are in the constraint scope.
    fn update_variable_ordering(&mut self, ordering: &[usize]);
    /// Updates the top-down local property of the mdd 
    fn update_property_top_down(&mut self, layers: &Vec<Layer>, nodes: &Vec<Node>, edges: &Vec<Edge>);
    /// Updates the bottom-up local property of the mdd 
    fn update_property_bottom_up(&mut self, layers: &Vec<Layer>, nodes: &Vec<Node>, edges: &Vec<Edge>);
    /// Returns true if the layer is in the scope of the constraint
    fn is_layer_in_scope(&self, layer: LayerIndex) -> bool;
    /// Returns true if the assignment is invalid and the edge can be removed
    fn is_assignment_invalid(&self, mdd: &Mdd, edge: EdgeIndex) -> bool;
    /// Adds a node in the given layer. Updates the properties of the constraints
    fn add_node_in_layer(&mut self, layer: LayerIndex);
    /// Returns an iterator on the constraint's scope
    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_>;
}
