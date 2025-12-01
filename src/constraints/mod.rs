pub mod all_different;
pub mod equals;

use crate::modelling::VariableIndex;
use crate::mdd::*;

pub use all_different::AllDifferent;

pub trait Constraint {
    /// Adds informtion about the variable ordering in the constraint
    fn update_variable_ordering(&mut self, ordering: &Vec<usize>);
    /// Updates the top-down local property of the node
    fn update_property_top_down(&mut self, mdd: &Mdd);
    /// Updates the bottom-up local property of the node
    fn update_property_bottom_up(&mut self, mdd: &Mdd);
    /// Returns true if the assignment is valid
    fn is_assignment_valid(&self, mdd: &Mdd, edge: EdgeIndex) -> bool;
    /// Iterates over the scope of the variable
    //TODO: change to an iterator
    fn iter_scope(&self) -> Vec<VariableIndex>;
}
