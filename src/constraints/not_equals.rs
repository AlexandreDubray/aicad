use super::*;
use crate::utils::SparseBitset;
use crate::modelling::*;
use crate::mdd::*;
use rustc_hash::FxHashSet;

pub struct NotEquals {
    x: VariableIndex,
    y: VariableIndex,
    domains: FxHashSet<isize>,
    top_down_properties: Vec<Vec<SparseBitset<isize>>>,
    bottom_up_properties: Vec<Vec<SparseBitset<isize>>>,
    layer_x: usize,
    layer_y: usize,
}

impl NotEquals {

    pub fn new(problem: &Problem, x: VariableIndex, y: VariableIndex) -> Self {
        let mut domains = FxHashSet::<isize>::default();
        for value in problem[x].iter_domain() {
            domains.insert(value);
        }
        for value in problem[y].iter_domain() {
            domains.insert(value);
        }
        let top_down_properties = (0..problem.number_variables() + 1).map(|_| {
            vec![SparseBitset::new(domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
        let bottom_up_properties = (0..problem.number_variables() + 1).map(|_| {
            vec![SparseBitset::new(domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
        Self {
            x,
            y,
            domains,
            top_down_properties,
            bottom_up_properties,
            layer_x: 0,
            layer_y: 0,
        }
    }

}

impl Constraint for NotEquals {

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        self.layer_x = ordering[self.x.0];
        self.layer_y = ordering[self.y.0];
    }

    fn update_property_top_down(&mut self, mdd: &Mdd)  {
        // First layer has no predecessor
        for target_layer in mdd.iter_layers().skip(1) {
            for i in 0..mdd[target_layer].number_nodes() {
                self.top_down_properties[target_layer.0][i].reset(0);
                let target_node = mdd[target_layer].node_at(i);
                for j in 0..mdd[target_node].number_parents() {
                    let edge = mdd[target_node].parent_edge_at(j);
                    let assignment = mdd[edge].assignment();

                    // Parent of this edge
                    let source_node = mdd[edge].from();
                    let source_layer = mdd[source_node].layer();
                    debug_assert!(source_layer.0 < target_layer.0);
                    let source_index = mdd[source_node].index_in_layer();

                    if self.is_layer_in_scope(source_layer) {
                        self.top_down_properties[target_layer.0][i].insert(assignment);
                    }

                    // Aggregate the source properties into the target properties.
                    // Since we need a mutable reference to the properties of layer and a
                    // non-mutable references to the source layer we can not directly update the
                    // properties. We use the `split_at_mut` method to get two mutable references
                    // to non-overlapping slice of the top_down_properties vector. Then, we can use
                    // these references to update the properties.
                    let (td_properties_above, td_properties_below) = self.top_down_properties.split_at_mut(target_layer.0);
                    td_properties_below[0][i].union(&td_properties_above[source_layer.0][source_index]);
                }
            }
        }
    }

    fn update_property_bottom_up(&mut self, mdd: &Mdd) {
        // Same procedure as the top-down, but in the other direction
        for source_layer in mdd.iter_layers().rev().skip(1) {
            let layer_in_scope = self.is_layer_in_scope(source_layer);
            for i in 0..mdd[source_layer].number_nodes() {
                self.bottom_up_properties[source_layer.0][i].reset(0);
                let source_node = mdd[source_layer].node_at(i);
                for j in 0..mdd[source_node].number_children() {
                    let edge = mdd[source_node].child_edge_at(j);
                    let assignment = mdd[edge].assignment();

                    let target_node = mdd[edge].to();
                    let target_layer = mdd[target_node].layer();
                    let target_index = mdd[target_node].index_in_layer();

                    if layer_in_scope {
                        self.bottom_up_properties[source_layer.0][i].insert(assignment);
                    }

                    let (bu_properties_above, bu_properties_below) = self.bottom_up_properties.split_at_mut(target_layer.0);
                    bu_properties_above[source_layer.0][i].union(&bu_properties_below[0][target_index]);
                }
            }
        }
    }

    fn is_layer_in_scope(&self, layer: LayerIndex) -> bool {
        layer.0 == self.layer_x || layer.0 == self.layer_y
    }

    fn is_assignment_invalid(&self, mdd: &Mdd, edge: EdgeIndex) -> bool {
        let assignment = mdd[edge].assignment();
        let source = mdd[edge].from();
        let source_layer = mdd[source].layer();
        let source_index = mdd[source].index_in_layer();

        let decision = mdd[source_layer].decision();
        if decision == self.x {
            if self.layer_x < self.layer_y {
                self.bottom_up_properties[source_layer.0][source_index].contains(assignment) && self.bottom_up_properties[source_layer.0][source_index].size() == 1
            } else {
                self.top_down_properties[source_layer.0][source_index].contains(assignment) && self.top_down_properties[source_layer.0][source_index].size() == 1
            }
        } else if self.layer_x > self.layer_y {
            self.bottom_up_properties[source_layer.0][source_index].contains(assignment) && self.bottom_up_properties[source_layer.0][source_index].size() == 1
        } else {
            self.top_down_properties[source_layer.0][source_index].contains(assignment) && self.top_down_properties[source_layer.0][source_index].size() == 1
        }
    }

    fn add_node_in_layer(&mut self, layer: LayerIndex) {
        let top_down_property = SparseBitset::new(self.domains.iter().copied());
        let bottom_up_property = SparseBitset::new(self.domains.iter().copied());
        self.top_down_properties[layer.0].push(top_down_property);
        self.bottom_up_properties[layer.0].push(bottom_up_property);
    }
}
