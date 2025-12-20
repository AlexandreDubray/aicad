use crate::modelling::*;
use super::*;
use crate::utils::bitset::Bitset;
use rustc_hash::{FxHasher, FxHashMap};
use std::hash::Hasher;

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
    /// Which constraint is scheduled for propagation
    scheduled_constraint: Bitset,
    cache: FxHashMap<u64, NodeIndex>,
}

impl Mdd {

    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(problem: &mut Problem) -> Self {
        let mut mdd = Self {
            nodes: vec![],
            edges: vec![],
            layers: vec![Layer::default(); problem.number_variables() + 1],
            propagation_queue: vec![],
            scheduled_constraint: Bitset::new(problem.number_constraints()),
            cache: FxHashMap::default(),
        };

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer in (0..mdd.number_layers()).map(LayerIndex) {
            mdd.add_node(problem, layer);
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

    fn add_node(&mut self, problem: &mut Problem, layer: LayerIndex) -> NodeIndex {
        let index_in_layer = self[layer].number_nodes();
        let node = Node::new(layer, index_in_layer);
        let index = NodeIndex(self.nodes.len());
        self[layer].add_node(index);
        self.nodes.push(node);
        for constraint in (0..problem.number_constraints()).map(ConstraintIndex) {
            problem[constraint].add_node_in_layer(layer);
        }
        index
    }

    fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, assignment: isize) {
        let edge_index = EdgeIndex(self.edges.len());
        self[from].add_child_edge(edge_index);
        self[to].add_parent_edge(edge_index);
        let layer_from = self[from].layer();
        let edge = Edge::new(layer_from, from, to, assignment);
        self.edges.push(edge);
    }

    pub fn refine(&mut self, problem: &mut Problem, max_width: usize) {
        for layer in 1..self.layers.len() - 1 {
            let width = self.layers[layer].number_nodes();
            for _ in width..max_width {
                let node_to_split = self.layers[layer].iter_nodes().find(|node| self[*node].number_parents() > 1);
                match node_to_split {
                    Some(node) => {
                        let new_node = self.split_node(problem, node);
                        self.propagate_constraints(problem);
                        let node_hash = self.hash_node(problem, new_node);
                        println!("Splitting node at layer {} with hash {}", layer, node_hash);
                        if let Some(n) = self.cache.get(&node_hash) {
                            self.merge_node(new_node, *n);
                        } else {
                            self.cache.insert(node_hash, new_node);
                        }
                    },
                    None => break,
                }
            }
        }
    }

    fn split_node(&mut self, problem: &mut Problem, node: NodeIndex) -> NodeIndex {
        let layer = self[node].layer();
        let new_node = self.add_node(problem, layer);
        // We split the parents accross the two nodes
        let n = self[node].number_parents();
        for i in (0..(n/2)).rev() {
            let edge = self[node].parent_edge_at(i);
            self[edge].deactivate();
            let from = self[edge].from();
            let assignment = self[edge].assignment();
            self.add_edge(from, new_node, assignment);
            self[node].swap_remove_parent_edge(i);
        }
        // Adds links from the new node to the children of the splitted node
        let n = self[node].number_children();
        for i in 0..n {
            let edge = self[node].child_edge_at(i);
            let to = self[edge].to();
            let assignment = self[edge].assignment();
            self.add_edge(new_node, to, assignment);
        }
        new_node
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
                        let n = self[node].number_children();
                        for i in (0..n).rev() {
                            let edge = self[node].child_edge_at(i);
                            if problem[constraint].is_assignment_invalid(self, edge) {
                                self.remove_child_of(node, i);
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

    fn remove_child_of(&mut self, node: NodeIndex, index: usize) {
        let edge = self[node].child_edge_at(index);
        self[node].swap_remove_child_edge(index);
        let to = self[edge].to();
        self[to].remove_parent_edge(edge);
        self[edge].deactivate();
        if self[node].number_children() == 0 {
            self.remove_node(node);
        }
        if self[to].number_parents() == 0 {
            self.remove_node(to);
        }
    }

    fn remove_parent_of(&mut self, node: NodeIndex, index: usize) {
        let edge = self[node].parent_edge_at(index);
        self[node].swap_remove_parent_edge(index);
        let from = self[edge].from();
        self[from].remove_child_edge(edge);
        self[edge].deactivate();
        if self[node].number_parents() == 0 {
            self.remove_node(node);
        }
        if self[from].number_children() == 0 {
            self.remove_node(from);
        }
    }

    fn remove_node(&mut self, node: NodeIndex) {
        self[node].deactivate();
        let n = self[node].number_children();
        for i in 0..n {
            self.remove_child_of(node, i);
        }
        let n = self[node].number_parents();
        for i in 0..n {
            self.remove_parent_of(node, i);
        }
    }

    fn merge_node(&mut self, node: NodeIndex, into: NodeIndex) {
        self[node].deactivate();
        let n = self[node].number_parents();
        for i in 0..n {
            let edge = self[node].parent_edge_at(i);
            self[edge].set_to(into);
            self[into].add_parent_edge(edge);
        }

        let n = self[node].number_children();
        for i in 0..n {
            self.remove_child_of(node, i);
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
        out.push_str("digraph {\nrankdir=TD;\ntranksep = 3;\n\n");

        let mut subgraph = String::new();
        subgraph.push_str("subgraph mdd {\n");
        let mut layer_labels = String::new();
        layer_labels.push_str("subgraph labels {\n");

        for layer in (0..self.layers.len()).map(LayerIndex) {
            let variable = self[layer].decision().0;
            layer_labels.push_str(&format!("\tL{} [shape=plaintext, label=\"x{}\"];\n", layer.0, variable));
        }

        for layer in (0..self.layers.len()).map(LayerIndex) {
            for node in self[layer].iter_nodes().filter(|n| self[*n].is_active()) {
                let id = format!("{{rank=same; {} [shape=point,width=0.05] L{}}}", node.0, layer.0);
                subgraph.push_str(&format!("\t{id};\n"));
            }
        }

        for edge in self.edges.iter().filter(|e| e.is_active()) {
            let from = edge.from();
            let to = edge.to();
            let assignment = edge.assignment();
            subgraph.push_str(&format!("\t{} -> {} [penwidth=1, label=\"{}\"];\n", from.0, to.0, assignment));
        }

        layer_labels.push_str("}\n");
        subgraph.push_str("}\n");

        out.push_str(&layer_labels);
        out.push_str(&subgraph);
        out.push('}');
        out
    }

    pub fn to_file(&self, filename: &str) {
        fs::write(filename, self.as_graphviz()).unwrap();
    }

    fn hash_node(&self, problem: &Problem, node: NodeIndex) -> u64 {
        let mut state = FxHasher::default();
        for constraint in problem.iter_constraints() {
            problem[constraint].hash_node(self, node, &mut state);
        }
        state.finish()
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
                for i in 0..mdd[node].number_parents() {
                    let edge = mdd[node].parent_edge_at(i);
                    let from = mdd[edge].from();
                    let index_in_layer = mdd[from].index_in_layer();
                    count_edge_from[index_in_layer] += 1;
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
        for i in 0..mdd[node].number_children() {
            let edge = mdd[node].child_edge_at(i);
            values.push(mdd[edge].assignment());
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

        let mdd = Mdd::new(&mut problem);
        assert!(mdd.number_nodes() == 4);
        assert!(node_possible_values(&mdd, NodeIndex(0)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(1)) == vec![0, 1]);
        assert!(node_possible_values(&mdd, NodeIndex(2)) == vec![0, 1, 2]);
    }

    #[test]
    pub fn mdd_refine() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1, 2]);
        let z = problem.add_variable(vec![1, 2]);

        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        problem.set_variable_ordering(vec![0, 1, 2]);

        let mut mdd = Mdd::new(&mut problem);
        mdd.refine(&mut problem, 10);
        mdd.to_file("mdd.txt");
        assert!(false);
    }
}
