use crate::core::*;
use crate::core::problem::Problem;
use super::*;

/// Represents the index of a layer in the MDD
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct LayerIndex(usize);

/// Represents the index of a node in a layer of the MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize);

/// Represents the index of an edge in a layer of a MDD.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize);


/// Structure for the MDD. The MDD is organised in layers (one layer per variable in the problem)
/// and each layer contains the necessary information to propagate the constraint and generate
/// solutions.
#[derive(Default)]
pub struct Mdd {
    /// Layers of the MDD
    layers: Vec<Layer>,
}

impl Mdd {

    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(problem: &Problem, variable_ordering: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = vec![];
        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node.
        for layer_id in 0..problem.number_variables() + 1 {
            let is_in_constraint_scope = if layer_id == problem.number_variables() { vec![] } else { vec![false; problem.number_constraints() ] };
            let layer = Layer::new(is_in_constraint_scope);
            layers.push(layer);
        }

        for (variable_id, layer) in variable_ordering.iter().copied().enumerate() {
            layers[layer].set_decision(VariableIndex(variable_id));
        }

        // We add the local properties for each node
        for constraint in problem.iter_constraints() {
            let property = problem[constraint].get_properties();
            for layer in 0..layers.len() {
                layers[layer][NodeIndex(0)].add_top_down_property(property.clone());
                layers[layer][NodeIndex(0)].add_bottom_up_property(property.clone());
            }
        }

        // Next, we add the edges between the layers. There is edges only from one layer to the
        // next.
        for layer_from in 0..problem.number_variables() {
            // There is only one node per layer, so they are indexed at 0 in the layers
            let edge_source = NodeIndex(0);
            let edge_target = NodeIndex(0);
            let variable = layers[layer_from].decision();

            for value in (0..problem[variable].domain_size()).map(ValueIndex) {
                // The edges work as a linked chain, each node only keep a pointer to one of its
                // parent, and the pointer to the next one is stored in the edge.
                let assignment = problem[variable].get_value(value);
                let next_child = layers[layer_from][edge_source].first_child();
                let next_parent = layers[layer_from + 1][edge_target].first_parent();
                let edge = Edge::new(NodeIndex(0), NodeIndex(0), assignment, next_parent, next_child);
                let edge_index = layers[layer_from].add_edge(edge);
                layers[layer_from][edge_source].set_first_child(Some(edge_index));
                layers[layer_from + 1][edge_target].set_first_parent(Some(edge_index));
            }
        }
        Mdd {
            layers,
        }
    }

    pub fn refine(&mut self, problem: &Problem) {
        self.compute_local_properties_top_down(problem);
        self.compute_local_properties_bottom_up(problem);
    }

    fn compute_local_properties_top_down(&mut self, problem: &Problem) {
        for layer in (0..self.layers.len()).map(LayerIndex) {
            for node in (0..self[layer].number_nodes()).map(NodeIndex) {
                for constraint in (0..problem.number_constraints()).map(ConstraintIndex) {
                    problem[constraint].update_property_top_down(self, layer, node, constraint.0);
                }
            }
        }
    }

    fn compute_local_properties_bottom_up(&mut self, problem: &Problem) {
        for layer in (0..self.layers.len()).rev().map(LayerIndex) {
            for node in (0..self[layer].number_nodes()).map(NodeIndex) {
                for constraint in (0..problem.number_constraints()).map(ConstraintIndex) {
                    problem[constraint].update_property_top_down(self, layer, node, constraint.0);
                }
            }
        }
    }
}

/* ---- Various helper implementation to make life easier ---- */

impl Mdd {

    /// Return an iterator over the indexes of the layers
    fn iter_layers(&self) -> impl Iterator<Item = LayerIndex> {
        (0..self.layers.len()).map(LayerIndex)
    }

    pub fn as_graphviz(&self) ->  String {
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        for layer in (0..self.layers.len()).map(LayerIndex) {
            for node in (0..self[layer].number_nodes()).map(NodeIndex) {
                let id = format!("l{}n{}", layer.0, node.0);
                let variable = self[layer].decision().0;
                out.push_str(&format!("\t{id} [label=\"x{variable}\"];\n"));
            }
        }

        for layer in (0..self.layers.len()).map(LayerIndex) {
            for edge in (0..self[layer].number_edges()).map(EdgeIndex) {
                let edge = &self[layer][edge];
                let from = edge.from();
                let from_id = format!("l{}n{}", layer.0, from.0);
                let to = edge.to();
                let to_id = format!("l{}n{}", layer.0 + 1, to.0);
            out.push_str(&format!("\t{from_id} -> {to_id} [penwidth=1];\n"));
            }
        }
        out.push_str("}");
        out
    }
}

impl std::ops::Index<LayerIndex> for Mdd {
    type Output = Layer;

    fn index(&self, index: LayerIndex) -> &Self::Output {
        &self.layers[index.0]
    }
}

impl std::ops::IndexMut<LayerIndex> for Mdd {
    fn index_mut(&mut self, index: LayerIndex) -> &mut Self::Output {
        &mut self.layers[index.0]
    }
}
