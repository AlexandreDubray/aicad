use super::*;
use crate::modelling::{VariableIndex, Problem};
use crate::mdd::*;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::utils::SparseBitset;

// Structures for the allDifferent constraint.
//
// References:
//    - Hoda, S., Van Hoeve, W. J., & Hooker, J. N. (2010, September). A systematic approach to MDD-based constraint programming. CP2010

/// Structure that is used to compute the local properties of the MDD's nodes. The same property is
/// used for top-down and bottom-up computation. The property is divided into two sets (A, S)
/// where:
///     - $A(n)$ represents the values appearing on all path to/from node $n$
///     - $S(n)$ represents the value appearing on some path to/from node $n$
/// We use bitset to represent the sets.
///
/// The property operators are defined as follows.
///     1. The integration of a value $v$ into the property $(A, S)$ is defined by $$(A, S) \otimes
///        v = (A \cup \{v\}, S \cup \{v\}$$.
///        implemented using the | operator
///     2. The aggregation of two properties $(A, S)$ and $(A^\prime, S^\prime)$ is computed as $$(A, S) \oplus
///        (A^\prime, S^\prime) = (A \cap A^\prime, S \cup S^\prime)$$
struct AllDifferentProperty {
    /// Values that appear on all source-n (top-down property) or n-sink (bottom-up
    /// property) path.
    value_all_path: SparseBitset<isize>,
    /// Values that appear on some source-n (top-down property) or n-sink (bottom-up
    /// property) path.
    value_some_path: SparseBitset<isize>,
}

impl AllDifferentProperty {

    /// Creates a new property with bitsiets of nb_words 64-bit unsigned integers
    pub fn new(domain: &FxHashSet<isize>) -> Self {
        let value_all_path = SparseBitset::new(domain.iter().copied());
        let value_some_path = SparseBitset::new(domain.iter().copied());
        Self {
            value_all_path,
            value_some_path,
        }
    }
}

impl AllDifferentProperty {

    /// Returns true if the shift-th bit in the word-th word is set to true in the all-path
    /// bitvector.
    fn is_value_on_all_path(&self, value: isize) -> bool {
        self.value_all_path.contains(value)
    }

    /// Returns true if the shift-th bit in the word-th word is set to true in the some-path
    /// bitvector.
    fn is_value_on_some_path(&self, value: isize) -> bool {
        self.value_some_path.contains(value)
    }

    /// Returns the number of bits set to 1 in the some-path bitvector.
    fn number_possible_values(&self) -> usize {
        self.value_some_path.size()
    }

    /// Returns the number of bits set to 1 in the some-path bitvector when unioned with the
    /// some-path bitvector of another property.
    fn number_possible_values_union(&self, other: &AllDifferentProperty) -> usize {
        self.value_some_path.size_union(&other.value_some_path)
    }
}

pub struct AllDifferent {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Top-down properties for each node in the MDD
    top_down_properties: Vec<Vec<AllDifferentProperty>>,
    /// Bottom-up properties for each node in the MDD
    bottom_up_properties: Vec<Vec<AllDifferentProperty>>,
    /// For each variable in the scope, indicates how many variables are above and below it in the
    /// MDD.
    map_hall_set: FxHashMap<VariableIndex, (usize, usize)>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl AllDifferent {

    /// Creates a new AllDifferent constraint over variables
    pub fn new(problem: &Problem, variables: Vec<VariableIndex>) -> Self {
        let mut domains = FxHashSet::<isize>::default();
        for variable in variables.iter().copied() {
            for value in problem[variable].iter_domain() {
                domains.insert(value);
            }
        }
        let top_down_properties = (0..problem.number_variables() + 1).map(|_| vec![AllDifferentProperty::new(&domains)]).collect::<Vec<Vec<AllDifferentProperty>>>();
        let bottom_up_properties = (0..problem.number_variables() + 1).map(|_| vec![AllDifferentProperty::new(&domains)]).collect::<Vec<Vec<AllDifferentProperty>>>();

        // Map each variable in the scope to the number of variable above and below it in the MDD.
        // Used to compute hall set propagation rules.
        let map_hall_set = FxHashMap::<VariableIndex, (usize, usize)>::default();
        let layer_in_scope = (0..(problem.number_variables() / 64).max(1)).map(|_| 0).collect::<Vec<u64>>();
        Self {
            variables,
            top_down_properties,
            bottom_up_properties,
            map_hall_set,
            layer_in_scope,
        }
    }

}

impl Constraint for AllDifferent {

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        // The layers in the scope of the variable are indicated using a bitvector of 64-bit words.
        // For each layer l its word index is given by l / 64 and the bit index by l % 64
        for variable in self.variables.iter() {
            let layer = ordering[variable.0];
            // Sets the bit of the layer to 1
            self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
        }
        // Compute the hall set sizes up and down the mdd. For a given layer l in the scope of the
        // constraint its hall set size up (resp. down) is the number of layer k such that k < l (k
        // > l) and k is in the constraint's scope

        // We sort each variable in the constraint's scope by its position in the ordering
        let mut scope_variable_order = self.variables.iter().copied().map(|v| (ordering[v.0], v)).collect::<Vec<(usize, VariableIndex)>>();
        scope_variable_order.sort_unstable();
        // The hall set sizes are stored as a tuple (size up, size down) and is given, for node i, by (i, n-i)
        let n = self.variables.len();
        for (pos, (_, variable)) in scope_variable_order.iter().copied().enumerate() {
            self.map_hall_set.insert(variable, (pos, n - 1 - pos));
        }
    }

    fn update_property_top_down(&mut self, mdd: &Mdd) {
        // We skip the first layer as it has no predecessors
        for target_layer in mdd.iter_layers().skip(1) {
            // We update the top-down properties for each node. Since the properties for the
            // allDifferent can be computed incrementally, we do this edge by edge
            for i in 0..mdd[target_layer].number_nodes() {
                // Node for which we update the property (i.e, the target of the edge coming from
                // layer - 1 into layer)
                let target_node = mdd[target_layer].node_at(i);
                let mut edge_ptr = mdd[target_node].first_parent();
                while let Some(edge) = edge_ptr {
                    // Gets the (word, shift) for the assignment
                    let assignment = mdd[edge].assignment();

                    // Parent of this edge
                    let source_node = mdd[edge].from();
                    let source_layer = mdd[source_node].layer();
                    debug_assert!(source_layer.0 < target_layer.0);
                    let source_index = mdd[source_node].index_in_layer();
                    let layer_in_scope = self.is_layer_in_scope(source_layer);

                    // For the set A we need to do $A \cap (A^\prime \cup \{assignment\})$. Hence,
                    // we can not directly integrate the assignment into A (as is done for the S
                    // set, since this is a union of union.
                    // Hence, we integrate the assignment into $S^\prime$ and then reverse it.
                    let is_in_set = self.top_down_properties[source_layer.0][source_index].value_all_path.contains(assignment);
                    // Only integrate the edge if the layer is in the scope of the constraint.
                    if layer_in_scope {
                        self.top_down_properties[target_layer.0][i].value_some_path.insert(assignment);
                        self.top_down_properties[source_layer.0][source_index].value_all_path.insert(assignment);
                    }

                    // Aggregate the source properties into the target properties.
                    // Since we need a mutable reference to the properties of layer and a
                    // non-mutable references to the source layer we can not directly update the
                    // properties. We use the `split_at_mut` method to get two mutable references
                    // to non-overlapping slice of the top_down_properties vector. Then, we can use
                    // these references to update the properties.
                    let (td_properties_above, td_properties_below) = self.top_down_properties.split_at_mut(target_layer.0);
                    td_properties_below[0][i].value_all_path.interesect(&td_properties_above[source_layer.0][source_index].value_all_path);
                    td_properties_below[0][i].value_some_path.union(&td_properties_above[source_layer.0][source_index].value_some_path);

                    // Reverse the integration of the edge into the $A^\prime$ set.
                    if layer_in_scope && !is_in_set{
                        self.top_down_properties[source_layer.0][source_index].value_all_path.remove(assignment);
                    }

                    edge_ptr = mdd[edge].next_parent();
                }
            }
        }
    }

    fn update_property_bottom_up(&mut self, mdd: &Mdd) {
        // Same procedure as the top-down, but in the other direction
        for source_layer in mdd.iter_layers().skip(1) {
            let layer_in_scope = self.is_layer_in_scope(source_layer);
            for i in 0..mdd[source_layer].number_nodes() {
                let source_node = mdd[source_layer].node_at(i);
                let mut edge_ptr = mdd[source_node].first_child();
                while let Some(edge) = edge_ptr {
                    let assignment = mdd[edge].assignment();

                    let target_node = mdd[edge].to();
                    let target_layer = mdd[target_node].layer();
                    let target_index = mdd[target_node].index_in_layer();

                    let is_in_set = self.bottom_up_properties[target_layer.0][target_index].value_all_path.contains(assignment);
                    if layer_in_scope {
                        self.bottom_up_properties[source_layer.0][i].value_some_path.insert(assignment);
                        self.bottom_up_properties[target_layer.0][target_index].value_all_path.insert(assignment);
                    }

                    let (bu_properties_above, bu_properties_below) = self.bottom_up_properties.split_at_mut(target_layer.0);
                    bu_properties_above[source_layer.0][i].value_all_path.interesect(&bu_properties_below[0][target_index].value_all_path);
                    bu_properties_above[source_layer.0][i].value_some_path.union(&bu_properties_below[0][target_index].value_some_path);

                    if layer_in_scope && !is_in_set {
                        self.bottom_up_properties[target_layer.0][target_index].value_all_path.remove(assignment);
                    }

                    edge_ptr = mdd[edge].next_child();
                }
            }
        }
    }

    /// Returns true if the layer is constrained by self
    fn is_layer_in_scope(&self, layer: LayerIndex) -> bool {
        self.layer_in_scope[layer.0 / 64] & (1 << (layer.0 % 64)) != 0
    }

    fn is_assignment_invalid(&self, mdd: &Mdd, edge: EdgeIndex) -> bool {
        let assignment = mdd[edge].assignment();

        let source = mdd[edge].from();
        let source_layer = mdd[source].layer();
        let source_index = mdd[source].index_in_layer();

        let target = mdd[edge].to();
        let target_layer = mdd[target].layer();
        let target_index = mdd[target].index_in_layer();


        // If the value appears on all path from the source or to the sink, then it will be taken
        // by another variable and can not be assigned to this one.
        if self.top_down_properties[source_layer.0][source_index].value_all_path.contains(assignment) ||
           self.bottom_up_properties[target_layer.0][target_index].value_all_path.contains(assignment) {
               println!("On all path from TD or BU");
                return true;
        }
        // If not, we check for Hall-set conditions
        let decision = mdd[source_layer].decision();
        let (hall_set_size_up, hall_set_size_down) = *self.map_hall_set.get(&decision).unwrap();
        let is_on_td_path = self.top_down_properties[source_layer.0][source_index].value_some_path.contains(assignment);
        let is_on_bu_path = self.bottom_up_properties[target_layer.0][target_index].value_some_path.contains(assignment);
        if is_on_td_path && hall_set_size_up == self.top_down_properties[source_layer.0][source_index].number_possible_values() {
            // First, the variables above are a Hall set: they can take as much values as the union of
            // their domain and this union includes the current assignment.
               println!("Hall set up");
            return true;
        } else if is_on_bu_path && hall_set_size_down == self.bottom_up_properties[target_layer.0][target_index].value_some_path.size() {
            // Same but for the variables in later layers.
               println!("Hall set down");
               println!("{}", self.bottom_up_properties[target_layer.0][target_index].value_some_path);
            return true;
        } else if is_on_bu_path && is_on_td_path && hall_set_size_up + hall_set_size_down == self.top_down_properties[source_layer.0][source_index].value_some_path.size_union(&self.bottom_up_properties[target_layer.0][target_index].value_some_path) {
            // Same but for all other variables in the constraint.
               println!("Hall set both");
            return true;
        }
        false
    }
}

impl std::fmt::Display for AllDifferentProperty {

    fn fmt(&self, f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A: {} - S: {}", self.value_all_path, self.value_some_path)
    }
}

#[cfg(test)]
mod test_all_diff {

    use crate::modelling::*;
    use crate::mdd::*;
    use crate::mdd::mdd::test_mdd::*;

    #[test]
    pub fn test_basic_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0]);
        let y = problem.add_variable(vec![0, 1]);

        all_different(&mut problem, vec![x, y]);
        problem.set_variable_ordering(vec![0, 1]);

        let mut mdd = Mdd::new(&problem);
        mdd.propagate_constraints(&mut problem);
        assert!(mdd.number_nodes() == 3);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![1]);
    }

    #[test]
    pub fn test_no_propagation() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1]);

        all_different(&mut problem, vec![x, y]);
        problem.set_variable_ordering(vec![0, 1]);

        let mut mdd = Mdd::new(&problem);
        mdd.propagate_constraints(&mut problem);
        assert!(mdd.number_nodes() == 3);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![0, 1]);
    }

    #[test]
    pub fn test_basic_hall_set_up() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1]);
        let z = problem.add_variable(vec![0, 1, 2]);
        all_different(&mut problem, vec![x, y, z]);
        problem.set_variable_ordering(vec![0, 1, 2]);

        let mut mdd = Mdd::new(&problem);
        mdd.propagate_constraints(&mut problem);
        assert!(mdd.number_nodes() == 4);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(2)) == vec![2]);
    }

    #[test]
    pub fn test_basic_hall_set_down() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2]);
        let y = problem.add_variable(vec![0, 1]);
        let z = problem.add_variable(vec![0, 1]);
        all_different(&mut problem, vec![x, y, z]);
        problem.set_variable_ordering(vec![0, 1, 2]);

        let mut mdd = Mdd::new(&problem);
        mdd.propagate_constraints(&mut problem);
        assert!(mdd.number_nodes() == 4);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![2]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(2)) == vec![0, 1]);
    }

    #[test]
    pub fn test_hall_set_around() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1, 2]);
        let z = problem.add_variable(vec![0, 1]);
        all_different(&mut problem, vec![x, y, z]);
        problem.set_variable_ordering(vec![0, 1, 2]);

        let mut mdd = Mdd::new(&problem);
        mdd.propagate_constraints(&mut problem);
        assert!(mdd.number_nodes() == 4);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![2]);
        assert!(node_possible_values(&mdd, NodeIndex(2)) == vec![0, 1]);
    }

}
