pub mod all_different;

use crate::core::VariableIndex;
use crate::mdd::mdd::*;

pub trait Constraint {
    /// Returns the initial local property for the nodes of the MDD.
    fn get_properties(&self) -> Vec<u64>;
    /// Updates the top-down local property of the node
    fn update_property_top_down(&self, mdd: &mut Mdd, layer: LayerIndex, node: NodeIndex, constraint_index: usize);
    /// Updates the bottom-up local property of the node
    fn update_property_bottom_up(&self, mdd: &mut Mdd, layer: LayerIndex, node: NodeIndex, constraint_index: usize);
    /// Iterates over the scope of the variable
    //TODO: change to an iterator
    fn iter_scope(&self) -> Vec<VariableIndex>;
}
