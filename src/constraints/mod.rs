pub mod all_different;

pub trait Constraint {
    /// Returns the initial local property for the nodes of the MDD.
    fn get_properties(&self) -> Vec<u64>;
    /// Returns the initial local property for the root of the MDD.
    fn get_root_property(&self) -> Vec<u64>;
    /// Returns the initial local property for the sink of the MDD.
    fn get_sink_property(&self) -> Vec<u64>;
    /// Integrate an edge to a property
    fn integrate_edge(&self, property: &mut Vec<u64>, value: isize);
    /// Aggregate multiple properties into a target one
    fn aggregate_properties(&self, target: &mut Vec<u64>, properties: &[Vec<u64>]);
}
