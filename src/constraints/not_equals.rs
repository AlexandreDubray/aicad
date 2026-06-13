use super::*;
use crate::utils::SparseBitset;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::{Hash, Hasher};
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

    pub fn new(x: VariableIndex, y: VariableIndex) -> Self {
        Self {
            x,
            y,
            domains: FxHashSet::<isize>::default(),
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            layer_x: 0,
            layer_y: 0,
        }
    }

}

impl Constraint for NotEquals {

    fn init(&mut self, vars: &[Variable]) {
        for value in vars[*self.x].iter_domain() {
            self.domains.insert(value);
        }
        for value in vars[*self.y].iter_domain() {
            self.domains.insert(value);
        }
        self.top_down_properties = (0..vars.len() + 1).map(|_| {
            vec![SparseBitset::new(self.domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| {
            vec![SparseBitset::new(self.domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
    }

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        self.layer_x = ordering[self.x.0];
        self.layer_y = ordering[self.y.0];
    }

    fn reset_property_top_down(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.top_down_properties[layer][index].reset(0);
    }

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize)  {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        if self.is_layer_in_scope(source_layer) {
            self.top_down_properties[target_layer][target_index].insert(assignment);
        }
        let (td_properties_above, td_properties_below) = self.top_down_properties.split_at_mut(target_layer);
        td_properties_below[0][target_index].union(&td_properties_above[source_layer][source_index]);
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index].reset(0);
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        if self.is_layer_in_scope(source_layer) {
            self.bottom_up_properties[source_layer][source_index].insert(assignment);
        }
        let (bu_properties_above, bu_properties_below) = self.bottom_up_properties.split_at_mut(source_layer);
        bu_properties_above[target_layer][target_index].union(&bu_properties_below[0][source_index]);
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        layer == self.layer_x || layer == self.layer_y
    }

    fn is_assignment_invalid(&self, source: NodeIndex, _target: NodeIndex, decision: VariableIndex, assignment: isize) -> bool {
        let NodeIndex(source_layer, source_index) = source;

        if decision == self.x {
            if self.layer_x < self.layer_y {
                self.bottom_up_properties[source_layer][source_index].contains(assignment) && self.bottom_up_properties[source_layer][source_index].size() == 1
            } else {
                self.top_down_properties[source_layer][source_index].contains(assignment) && self.top_down_properties[source_layer][source_index].size() == 1
            }
        } else if self.layer_x > self.layer_y {
            self.bottom_up_properties[source_layer][source_index].contains(assignment) && self.bottom_up_properties[source_layer][source_index].size() == 1
        } else {
            self.top_down_properties[source_layer][source_index].contains(assignment) && self.top_down_properties[source_layer][source_index].size() == 1
        }
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        let top_down_property = SparseBitset::new(self.domains.iter().copied());
        let bottom_up_property = SparseBitset::new(self.domains.iter().copied());
        self.top_down_properties[layer].push(top_down_property);
        self.bottom_up_properties[layer].push(bottom_up_property);
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new([self.x, self.y].into_iter())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        assignment[*self.x] != assignment[*self.y]
    }

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        for word in self.top_down_properties[layer][index].words().iter().copied() {
            state.write_u64(word);
        }
        for word in self.bottom_up_properties[layer][index].words().iter().copied() {
            state.write_u64(word);
        }
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        self.top_down_properties[layer][index] == self.top_down_properties[olayer][oindex] &&
        self.bottom_up_properties[layer][index] == self.bottom_up_properties[olayer][oindex]
    }
}
