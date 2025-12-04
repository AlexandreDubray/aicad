use crate::modelling::*;
use super::*;
use crate::utils::bitset::Bitset;

use std::fs;

/// Structure for the MDD. The MDD is organised in layers (one layer per variable in the problem)
/// and each layer contains the necessary information to propagate the constraint and generate
/// solutions.
pub struct Mdd {
    /// Nodes of the MDD.
    nodes: Vec<Node>,
    /// Edges of the MDD.
    edges: Vec<Edge>,
    /// Layers of the MDD
    layers: Vec<Layer>,
    /// Propagation queue for the constraints
    propagation_queue: Vec<ConstraintIndex>,
    scheduled_constraint: Bitset,
}

impl Mdd {

    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(problem: &Problem) -> Self {
        let mut mdd = Self {
            nodes: vec![],
            edges: vec![],
            layers: vec![Layer::default(); problem.number_variables() + 1],
            propagation_queue: vec![],
            scheduled_constraint: Bitset::new(problem.number_constraints()),
        };

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer in (0..mdd.number_layers()).map(LayerIndex) {
            mdd.add_node(layer);
        }

        // We set the decision variable in each layer using the given ordering
        for (variable_id, layer) in problem.variable_ordering().iter().copied().enumerate() {
            mdd[LayerIndex(layer)].set_decision(VariableIndex(variable_id));
        }

        // Next, we add the edges between the layers. There is edges only from one layer to the
        // next.
        for layer_from in (0..mdd.number_layers() - 1).map(LayerIndex) {
            // There is only one node per layer, so they are indexed at 0 in the layers
            let edge_source = mdd[layer_from].node_at(0);
            let edge_target = mdd[layer_from + 1].node_at(0);
            let variable = mdd[layer_from].decision();

            for value in (0..problem[variable].domain_size()).map(ValueIndex) {
                // The edges work as a linked chain, each node only keep a pointer to one of its
                // parent, and the pointer to the next one is stored in the edge.
                let assignment = problem[variable].get_value(value);
                mdd.add_edge(edge_source, edge_target, assignment);
            }
        }
        mdd
    }

    fn add_node(&mut self, layer: LayerIndex) {
        let index_in_layer = self[layer].number_nodes();
        let node = Node::new(layer, index_in_layer);
        let index = NodeIndex(self.nodes.len());
        self[layer].add_node(index);
        self.nodes.push(node);
    }

    fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, assignment: isize) {
        let first_child = self[from].first_child();
        let first_parent = self[to].first_parent();
        let index = EdgeIndex(self.edges.len());

        if let Some(edge) = first_child {
            self[edge].set_prev_child(Some(index));
        }
        if let Some(edge) = first_parent {
            self[edge].set_prev_parent(Some(index));
        }
        self[from].set_first_child(Some(index));
        self[to].set_first_parent(Some(index));
        let layer_from = self[from].layer();
        let edge = Edge::new(layer_from, from, to, assignment, first_parent, first_child);
        self.edges.push(edge);
    }

    // TODO: This is a very, very, very rough approach to constraint propagation that needs a lot
    // of work
    pub fn propagate_constraints(&mut self, problem: &mut Problem) {
        for constraint in problem.iter_constraints() {
            self.propagation_queue.push(constraint);
            self.scheduled_constraint.insert(constraint.0);
        }
        while let Some(constraint) = self.propagation_queue.pop() {
            self.scheduled_constraint.remove(constraint.0);
            problem[constraint].update_property_top_down(self);
            problem[constraint].update_property_bottom_up(self);
            for layer in (0..self.layers.len() - 1).map(LayerIndex) {
                if problem[constraint].is_layer_in_scope(layer) {
                    let mut to_schedule = false;
                    for node_index in 0..self[layer].number_nodes() {
                        let node = self[layer].node_at(node_index);
                        let mut edge_ptr = self[node].first_child();
                        while let Some(edge) = edge_ptr {
                            // We first update the pointer, in case the edge is removed
                            edge_ptr = self[edge].next_child();
                            println!("Checking assignment {} at layer {}",self[edge].assignment(), layer.0);
                            if problem[constraint].is_assignment_invalid(self, edge) {
                                self.remove_edge(edge);
                                to_schedule = true;
                            }
                        }
                    }
                    if to_schedule {
                        let decision = self[layer].decision();
                        for constraint in problem[decision].iter_constraints() {
                            if !self.scheduled_constraint.contains(constraint.0) {
                                self.scheduled_constraint.insert(constraint.0);
                                self.propagation_queue.push(constraint);
                            }
                        }
                    }
                }
            }
        }
    }

    fn remove_edge(&mut self, edge: EdgeIndex) {
        self[edge].deactivate();
        let prev_parent = self[edge].prev_parent();
        let next_parent = self[edge].next_parent();
        let prev_child = self[edge].prev_child();
        let next_child = self[edge].next_child();
        if let Some(parent) = prev_parent {
            self[parent].set_next_parent(next_parent);
        }
        if let Some(parent) = next_parent {
            self[parent].set_prev_parent(prev_parent);
        }
        if let Some(child) = prev_child {
            self[child].set_next_child(next_child);
        }
        if let Some(child) = next_child {
            self[child].set_prev_child(prev_child);
        }
        let from = self[edge].from();
        let to = self[edge].to();
        if edge == self[from].first_child().unwrap() {
            self[from].set_first_child(next_child);
        }
        if edge == self[to].first_parent().unwrap() {
            self[to].set_first_parent(next_parent);
        }
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn number_layers(&self) -> usize {
        self.layers.len()
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
            for node in self[layer].iter_nodes().filter(|n| self[*n].is_active()) {
                let id = format!("{}", node.0);
                let variable = self[layer].decision().0;
                out.push_str(&format!("\t{id} [label=\"x{variable}\"];\n"));
            }
        }

        for edge in self.edges.iter().filter(|e| e.is_active()) {
            let from = edge.from();
            let to = edge.to();
            let assignment = edge.assignment();
            out.push_str(&format!("\t{} -> {} [penwidth=1, label=\"{}\"];\n", from.0, to.0, assignment));
        }
        out.push('}');
        out
    }

    pub fn to_file(&self, filename: &str) {
        fs::write(filename, self.as_graphviz()).unwrap();
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

#[cfg(test)]
pub mod test_mdd {

    use rustc_hash::FxHashMap;
    use crate::modelling::*;
    use crate::mdd::*;

    pub fn count_number_solution(mdd: &Mdd) -> usize {
        let mut map_count = FxHashMap::<NodeIndex, usize>::default();
        // Only one node in the first and last layers
        let last_node = mdd.layers.last().unwrap().node_at(0);
        let first_node = mdd.layers[0].node_at(0);
        map_count.insert(first_node, 1);
        for layer in mdd.iter_layers().skip(1) {
            let number_nodes_prev_layer = mdd[layer - 1].number_nodes();
            for node in mdd[layer].iter_nodes() {
                let mut count_edge_from = vec![0; number_nodes_prev_layer];
                let mut edge_ptr = mdd[node].first_parent();
                while let Some(edge) = edge_ptr {
                    let from = mdd[edge].from();
                    let index_in_layer = mdd[from].index_in_layer();
                    count_edge_from[index_in_layer] += 1;
                    edge_ptr = mdd[edge].next_parent();
                }
                let count_path_to_node = mdd[layer - 1].iter_nodes().map(|n| {
                    let number_edges = count_edge_from[mdd[n].index_in_layer()];
                    let number_paths_to_parents = *map_count.get(&n).unwrap();
                    number_edges*number_paths_to_parents
                }).sum::<usize>();
                map_count.insert(node, count_path_to_node);
            }
        }
        *map_count.get(&last_node).unwrap()
    }

    pub fn node_possible_values(mdd: &Mdd, node: NodeIndex) -> Vec<isize> {
        let mut values = vec![];
        let mut edge_ptr = mdd[node].first_child();
        while let Some(edge) = edge_ptr {
            values.push(mdd[edge].assignment());
            edge_ptr = mdd[edge].next_child();
        }
        values.sort_unstable();
        values
    }

    #[test]
    pub fn mdd_creation() {
        let mut problem = Problem::default();
        problem.add_variable(vec![0, 1]);
        problem.add_variable(vec![0, 1]);
        problem.add_variable(vec![0, 1, 2]);
        problem.set_variable_ordering(vec![0, 1, 2]);

        let mdd = Mdd::new(&problem);
        assert!(mdd.number_nodes() == 4);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(2)) == vec![0, 1, 2]);
    }
}
