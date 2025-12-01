use super::*;
use crate::modelling::{VariableIndex, Problem};
use crate::mdd::*;
use rustc_hash::FxHashMap;

/// Structure for the allDifferent constraint.
///
/// References:
///     - Hoda, S., Van Hoeve, W. J., & Hooker, J. N. (2010, September). A systematic approach to MDD-based constraint programming. CP2010

struct AllDifferentProperty {
    value_all_path: Vec<u64>,
    value_some_path: Vec<u64>,
}

impl AllDifferentProperty {

    pub fn new(nb_words: usize) -> Self {
        Self {
            value_all_path: vec![0; nb_words],
            value_some_path: vec![0; nb_words],
        }
    }
}

impl AllDifferentProperty {

    fn is_value_on_all_path(&self, word: usize, shift: usize) -> bool {
        self.value_all_path[word] & (1 << shift) != 0
    }

    fn is_value_on_some_path(&self, word: usize, shift: usize) -> bool {
        self.value_some_path[word] & (1 << shift) != 0
    }

    fn number_possible_values(&self) -> usize {
        self.value_some_path.iter().map(|word| word.count_ones()).sum::<u32>() as usize
    }
}

pub struct AllDifferent {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Map each variable in the scope to a bit in the bitsets
    map_bit: FxHashMap<isize, (usize, usize)>,
    /// Number of words in each bitset. The properties for this constraint contains 2*nb_words (2
    /// bitvectors)
    nb_words: usize,
    /// Top-down properties for each node in the MDD
    top_down_properties: Vec<Vec<AllDifferentProperty>>,
    /// Bottom-up properties for each node in the MDD
    bottom_up_properties: Vec<Vec<AllDifferentProperty>>,
    /// For each variable in the scope, indicates how many variables are above and below it in the
    /// MDD.
    map_hall_set: FxHashMap<VariableIndex, (usize, usize)>,
    layer_in_scope: Vec<u64>,
}

impl AllDifferent {

    pub fn new(problem: &Problem, variables: Vec<VariableIndex>) -> Self {
        let mut map_bit = FxHashMap::<isize, (usize, usize)>::default();
        let mut bit = 0;
        for variable in variables.iter().copied() {
            for value in problem[variable].iter_domain() {
                if !map_bit.contains_key(&value) {
                    map_bit.insert(value, (bit / 64, bit % 64));
                    bit += 1;
                }
            }
        }
        let nb_words = (bit / 64).max(1);

        let top_down_properties = (0..problem.number_variables() + 1).map(|_| vec![AllDifferentProperty::new(nb_words)]).collect::<Vec<Vec<AllDifferentProperty>>>();
        let bottom_up_properties = (0..problem.number_variables() + 1).map(|_| vec![AllDifferentProperty::new(nb_words)]).collect::<Vec<Vec<AllDifferentProperty>>>();

        let map_hall_set = FxHashMap::<VariableIndex, (usize, usize)>::default();
        let layer_in_scope = (0..(problem.number_variables() / 64).max(1)).map(|_| 0).collect::<Vec<u64>>();
        Self {
            variables,
            map_bit,
            nb_words,
            top_down_properties,
            bottom_up_properties,
            map_hall_set,
            layer_in_scope,
        }
    }

    fn is_layer_in_constraint(&self, layer: LayerIndex) -> bool {
        self.layer_in_scope[layer.0 / 64] & (1 << (layer.0 % 64)) != 0
    }
}

impl Constraint for AllDifferent {

    fn update_variable_ordering(&mut self, ordering: &Vec<usize>) {
        for variable in self.variables.iter() {
            let layer = ordering[variable.0];
            self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
        }
        let mut scope_variable_order = self.variables.iter().copied().map(|v| (ordering[v.0], v)).collect::<Vec<(usize, VariableIndex)>>();
        scope_variable_order.sort();
        let mut pos = 0;
        let n = self.variables.len();
        for (_, variable) in scope_variable_order.iter().copied() {
            self.map_hall_set.insert(variable, (pos, n - 1 - pos));
            pos += 1;
        }
    }

    /// Update the top-down property of a given node. This implies two operations:
    ///     1. Aggregating the local property of all parents. This is done using intersection for
    ///        the A bitset and union for the S bitset
    ///     2. Incorporating the edges' values to the local properties of the parents before the
    ///        integrations. We do this as union of {value} with the A and S bitsets.
    ///
    /// Since the aggregation operations are comutative, we can do this update edge by edge.
    fn update_property_top_down(&mut self, mdd: &Mdd) {
        for layer in mdd.iter_layers().skip(1) {
            let layer_in_scope = self.is_layer_in_constraint(layer - 1);
            for i in 0..mdd[layer].number_nodes() {
                let target_node = mdd[layer].node_at(i);
                let mut edge_ptr = mdd[target_node].first_parent();
                while let Some(edge) = edge_ptr {
                    let assignment = mdd[edge].assignment();
                    let (assignment_word, assignment_shift) = *self.map_bit.get(&assignment).unwrap();

                    let source_node = mdd[edge].from();
                    let source_layer = mdd[source_node].layer().0;
                    let source_index = mdd[source_node].index_in_layer();

                    let to_rev = self.top_down_properties[source_layer][source_index].value_some_path[assignment_word];
                    if layer_in_scope {
                        self.top_down_properties[layer.0][i].value_all_path[assignment_word] |= 1 << assignment_shift;
                        self.top_down_properties[source_layer][source_index].value_some_path[assignment_word] |= 1 << assignment_shift;
                    }

                    for word in 0..self.nb_words {
                        self.top_down_properties[layer.0][i].value_all_path[word] |= self.top_down_properties[source_layer][source_index].value_all_path[word];
                        self.top_down_properties[layer.0][i].value_some_path[word] &= self.top_down_properties[source_layer][source_index].value_some_path[word];
                    }

                    self.top_down_properties[source_layer][source_index].value_some_path[assignment_word] = to_rev;

                    edge_ptr = mdd[edge].next_parent();
                }
            }
        }
    }

    fn update_property_bottom_up(&mut self, mdd: &Mdd) {
        for layer in mdd.iter_layers().rev().skip(1) {
            let layer_in_scope = self.is_layer_in_constraint(layer);
            for i in 0..mdd[layer].number_nodes() {
                let target_node = mdd[layer].node_at(i);
                let mut edge_ptr = mdd[target_node].first_child();
                while let Some(edge) = edge_ptr {
                    let assignment = mdd[edge].assignment();
                    let (assignment_word, assignment_shift) = *self.map_bit.get(&assignment).unwrap();

                    let target_node = mdd[edge].to();
                    let target_layer = mdd[target_node].layer().0;
                    let target_index = mdd[target_node].index_in_layer();

                    let to_rev = self.bottom_up_properties[target_layer][target_index].value_all_path[assignment_word];
                    if layer_in_scope {
                        self.bottom_up_properties[layer.0][i].value_all_path[assignment_word] |= 1 << assignment_shift;
                        self.bottom_up_properties[target_layer][target_index].value_some_path[assignment_word] |= 1 << assignment_shift;
                    }

                    for word in 0..self.nb_words {
                        self.bottom_up_properties[layer.0][i].value_all_path[word] &= self.bottom_up_properties[target_layer][target_index].value_all_path[word];
                        self.bottom_up_properties[layer.0][i].value_some_path[word] |= self.bottom_up_properties[target_layer][target_index].value_some_path[word];
                    }
                    self.bottom_up_properties[target_layer][target_index].value_all_path[assignment_word] = to_rev;

                    edge_ptr = mdd[edge].next_child();
                }
            }
        }
    }

    fn is_assignment_valid(&self, mdd: &Mdd, edge: EdgeIndex) -> bool {
        let assignment = mdd[edge].assignment();
        let (word, shift) = *self.map_bit.get(&assignment).unwrap();

        let source = mdd[edge].from();
        let source_layer = mdd[source].layer();
        let source_index = mdd[source].index_in_layer();

        let target = mdd[edge].to();
        let target_layer = mdd[target].layer();
        let target_index = mdd[target].index_in_layer();


        if self.top_down_properties[source_layer.0][source_index].is_value_on_all_path(word, shift) ||
           self.bottom_up_properties[target_layer.0][target_index].is_value_on_all_path(word, shift) {
                return false;
        }
        let decision = mdd[source_layer].decision();
        let (hall_set_size_up, hall_set_size_down) = *self.map_hall_set.get(&decision).unwrap();
        !((self.top_down_properties[source_layer.0][source_index].is_value_on_some_path(word, shift) && hall_set_size_up == self.top_down_properties[source_layer.0][source_index].number_possible_values()) ||
           (self.bottom_up_properties[target_layer.0][target_index].is_value_on_some_path(word, shift) && hall_set_size_down == self.bottom_up_properties[target_layer.0][target_index].number_possible_values()))
    }

    fn iter_scope(&self) -> Vec<VariableIndex> {
        self.variables.clone()
    }
}

#[cfg(test)]
mod test_all_diff {

    use crate::modelling::*;
    use crate::mdd::*;

    #[test]
    pub fn test_basic() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0]);
        let y = problem.add_variable(vec![0, 1]);

        all_different(&mut problem, vec![x, y]);
        problem.set_variable_ordering(vec![0, 1]);

        let mut mdd = Mdd::new(&problem);
        mdd.refine(&mut problem);
        assert!(mdd.number_nodes() == 3);
        assert!(mdd[LayerIndex(0)].number_edges() == 1);
        assert!(mdd[LayerIndex(1)].number_edges() == 1);
    }

}
