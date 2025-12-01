use crate::modelling::*;
use super::*;

use std::fs;

/// Structure for the MDD. The MDD is organised in layers (one layer per variable in the problem)
/// and each layer contains the necessary information to propagate the constraint and generate
/// solutions.
#[derive(Default)]
pub struct Mdd {
    /// Nodes of the MDD.
    nodes: Vec<Node>,
    /// Edges of the MDD.
    edges: Vec<Edge>,
    /// Layers of the MDD
    layers: Vec<Layer>,
}

impl Mdd {

    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(problem: &Problem) -> Self {
        let mut layers: Vec<Layer> = vec![];
        let mut nodes: Vec<Node> = vec![];
        let mut edges: Vec<Edge> = vec![];

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer_id in 0..problem.number_variables() + 1 {
            // Indicates if the layer is in the scope of the constraints. One boolean per
            // constraint in each layer.
            let mut layer = Layer::new();
            // We have one node per layer
            let node = Node::new(LayerIndex(layer_id), 0);
            let node_index = NodeIndex(nodes.len());
            layer.add_node(node_index);

            nodes.push(node);
            layers.push(layer);
        }

        // We set the decision variable in each layer using the given ordering
        for (variable_id, layer) in problem.variable_ordering().iter().copied().enumerate() {
            layers[layer].set_decision(VariableIndex(variable_id));
        }

        // Next, we add the edges between the layers. There is edges only from one layer to the
        // next.
        for layer_from in 0..problem.number_variables() {
            // There is only one node per layer, so they are indexed at 0 in the layers
            let edge_source = layers[layer_from].node_at(0);
            let edge_target = layers[layer_from + 1].node_at(0);
            let variable = layers[layer_from].decision();

            for value in (0..problem[variable].domain_size()).map(ValueIndex) {
                // The edges work as a linked chain, each node only keep a pointer to one of its
                // parent, and the pointer to the next one is stored in the edge.
                let assignment = problem[variable].get_value(value);
                let next_child = nodes[edge_source.0].first_child();
                let next_parent = nodes[edge_target.0].first_parent();
                let edge = Edge::new(LayerIndex(layer_from), edge_source, edge_target, assignment, next_parent, next_child);
                let edge_index = EdgeIndex(edges.len());
                edges.push(edge);
                nodes[edge_source.0].set_first_child(Some(edge_index));
                nodes[edge_target.0].set_first_parent(Some(edge_index));
                layers[layer_from].add_edge(edge_index);
            }
        }
        Mdd {
            edges,
            nodes,
            layers,
        }
    }

    pub fn refine(&mut self, problem: &mut Problem) {
        for constraint in (0..problem.number_constraints()).map(ConstraintIndex) {
            problem[constraint].update_property_top_down(self);
            problem[constraint].update_property_bottom_up(self);
        }
        for layer in (0..self.layers.len() - 1).map(LayerIndex) {
            for constraint in problem.iter_constraints() {
                for i in (0..self[layer].number_edges()).rev() {
                    let edge = self[layer].edge_at(i);
                    if !problem[constraint].is_assignment_valid(self, edge) {
                        self[layer].swap_remove_edge(i);
                    }
                }
            }
        }
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn iter_layers(&self) -> impl DoubleEndedIterator<Item = LayerIndex> {
        (0..self.layers.len()).map(LayerIndex)
    }
}

/* ---- Various helper implementation to make life easier ---- */

impl Mdd {

    pub fn as_graphviz(&self) ->  String {
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        for layer in (0..self.layers.len()).map(LayerIndex) {
            for node in self[layer].iter_nodes() {
                let id = format!("{}", node.0);
                let variable = self[layer].decision().0;
                out.push_str(&format!("\t{id} [label=\"x{variable}\"];\n"));
            }
        }

        for layer in 0..self.layers.len() {
            for edge in self.layers[layer].iter_edges() {
                let edge = &self[edge];
                let from = edge.from();
                let to = edge.to();
                let assignment = edge.assignment();
                out.push_str(&format!("\t{} -> {} [penwidth=1, label=\"{}\"];\n", from.0, to.0, assignment));
            }
        }
        out.push_str("}");
        out
    }

    pub fn to_file(&self, filename: &str) {
        fs::write(filename, format!("{}", self.as_graphviz())).unwrap();
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

impl std::ops::Index<EdgeIndex> for Mdd {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0]
    }
}

impl std::ops::IndexMut<EdgeIndex> for Mdd {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edges[index.0]
    }
}

impl std::ops::Index<NodeIndex> for Mdd {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<NodeIndex> for Mdd {

    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

impl std::ops::Sub<usize> for LayerIndex {
    type Output = LayerIndex;

    fn sub(self, rhs: usize) -> Self::Output {
        LayerIndex(self.0 - rhs)
    }
}

impl std::ops::Add<usize> for LayerIndex {
    type Output = LayerIndex;

    fn add(self, rhs: usize) -> Self::Output {
        LayerIndex(self.0 + rhs)
    }
}
