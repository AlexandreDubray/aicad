use crate::modelling::*;
use super::*;
use super::heuristics::*;

use std::cell::RefCell;
use rand;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use rand::SeedableRng;

use std::fs;
use rustc_hash::{FxHashSet, FxHashMap};

thread_local! {
    static RNG: RefCell<Xoshiro256Plus> = RefCell::new(Xoshiro256Plus::from_rng(&mut rand::rng()));
}

/// Structure for the MDD. The MDD is organised in layers (one layer per variable in the problem)
/// and each layer contains the necessary information to propagate the constraint and generate
/// solutions.
pub struct Mdd {
    problem: Problem,
    /// Nodes of the MDD.
    nodes: Vec<Vec<Node>>,
    /// Edges of the MDD.
    edges: Vec<Vec<Edge>>,
    /// Branching order
    order: Vec<VariableIndex>,
    /// Max width allows during compilation
    max_width: usize,
    /// Heuristic used to score nodes during merging operation
    merge_heuristic: MergeHeuristic,
    /// Is the MDD unsat
    unsat: bool,
    /// Root of the MDD
    root: NodeIndex,
    /// Sink of the mdd
    sink: NodeIndex,
}

impl Mdd {

    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(problem: Problem, max_width: usize, order: OrderingHeuristic, merge_heuristic: MergeHeuristic) -> Self {
        let number_layers = problem.number_variables() + 1;
        let mut mdd = Self {
            nodes: vec![vec![]; problem.number_variables() + 1],
            edges: vec![vec![]; problem.number_variables()],
            order: vec![],
            max_width,
            merge_heuristic,
            problem,
            unsat: false,
            root: NodeIndex(0, 0),
            sink: NodeIndex(number_layers - 1, 0),
        };
        mdd.problem.init_constraints();

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer in 0..mdd.number_layers() {
            mdd.add_node(layer, layer != 0);
        }

        // Set the variable order in the MDD given the heuristics
        // We get for each layer its decision variable
        let var_order = order.get_order(&mdd.problem);
        // Reverse mapping order: for each variable, give its layer
        let mut var_order_inv = vec![0; var_order.len()];
        for (layer, variable) in var_order.iter().copied().enumerate() {
            var_order_inv[variable.0] = layer;
        }
        mdd.order = var_order;

        // For each constraint, update its variable order if necessary. For example, it is used in
        // the allDifferent constraint to compute hall sets
        for constraint in mdd.problem.iter_constraints().collect::<Vec<ConstraintIndex>>() {
            mdd.problem[constraint].update_variable_ordering(&var_order_inv);
        }

        // Next, we add the edges between the layers. There is edges only from one layer to the
        // next.
        for layer in 0..mdd.nodes.len() - 1 {
            let source = NodeIndex(layer, 0);
            let target = NodeIndex(layer + 1, 0);
            let variable = mdd.order[layer];
            for value in (0..mdd.problem[variable].domain_size()).map(ValueIndex) {
                mdd.add_edge(layer, source, target, value);
            }
        }
        mdd.propagate_constraints();
        if !mdd[mdd.root].is_active() || !mdd[mdd.sink].is_active() {
            mdd.unsat = true;
            return mdd;
        }
        mdd.clean();
        for layer in 1..mdd.number_layers() {
            let node = NodeIndex(layer, 0);
            if mdd[node].number_parents() == 1 {
                mdd[node].set_relaxed(false);
            } else {
                break;
            }
        }
        mdd
    }

    fn add_node(&mut self, layer: usize, relaxed: bool) -> NodeIndex {
        let index_in_layer = self.nodes[layer].len();
        let node = Node::new(layer, index_in_layer, relaxed);
        let index = NodeIndex(layer, index_in_layer);
        self.nodes[layer].push(node);
        for constraint in (0..self.problem.number_constraints()).map(ConstraintIndex) {
            self.problem[constraint].add_node_in_layer(layer);
        }
        index
    }

    fn add_edge(&mut self, layer: usize, from: NodeIndex, to: NodeIndex, assignment: ValueIndex) {
        let edge_index = EdgeIndex(layer, self.edges[layer].len());
        self[from].add_child_edge(edge_index);
        self[to].add_parent_edge(edge_index);
        let edge = Edge::new(from, to, assignment);
        self.edges[layer].push(edge);
    }

    pub fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        self.order[layer]
    }

    // --- split and refine strategy ---- //

    pub fn refine(&mut self) {
        if self.unsat {
            return;
        }
        for layer in 1..self.nodes.len() - 1 {
            if self.number_nodes_in_layer(layer) == self.max_width {
                continue;
            }
            let node = NodeIndex(layer, 0);
            self.split_node(node);
            self.propagate_constraints();
            if !self[self.root].is_active() || !self[self.sink].is_active() {
                self.unsat = true;
                return;
            }
            self.merge_layer(layer);
            self.clean();
        }
    }

    fn split_node(&mut self, node: NodeIndex) {
        let layer = self[node].layer();
        let n = self[node].number_parents();
        let outgoing_assignments = self[node]
            .iter_children()
            .filter(|edge| self[*edge].is_active())
            .map(|edge| (self[edge].to(), self[edge].assignment()))
            .collect::<Vec<(NodeIndex, ValueIndex)>>();
        self[node].set_relaxed(false);
        for i in (1..n).rev() {
            let new_node = self.add_node(layer, false);
            let edge = self[node].parent_edge_at(i);
            let from = self[edge].from();
            let assignment = self[edge].assignment();
            self.add_edge(layer - 1, from, new_node, assignment);
            for (child, outgoing_assignment) in outgoing_assignments.iter().copied() {
                self.add_edge(layer, new_node, child, outgoing_assignment);
            }
            self[edge].deactivate();
            self[node].swap_remove_parent_edge(i);
        }
    }


    pub fn propagate_constraints(&mut self) {
        let number_layers = self.nodes.len();

        // Top-down pass.
        for layer in 1..number_layers {
            let variable = self.order[layer - 1];
            let nodes_in_layer = self.nodes[layer].len();
            for i in 0..nodes_in_layer {
                let target = NodeIndex(layer, i);
                for constraint in (0..self.problem.number_constraints()).map(ConstraintIndex) {
                    self.problem[constraint].reset_property_top_down(target);
                    for j in 0..self[target].number_parents() {
                        let edge = self[target].parent_edge_at(j);
                        let source = self[edge].from();
                        let assignment = self.problem[variable].value(self[edge].assignment());
                        self.problem[constraint].update_property_top_down(source, target, assignment);
                    }

                }
            }
        }

        // We start by the bottom-up pass. We filter edges in this pass
        for layer in (0..number_layers - 1).rev() {
            let decision = self.order[layer];
            let nodes_in_layer = self.nodes[layer].len();
            for node_index in 0..nodes_in_layer {
                let target = NodeIndex(layer, node_index);
                if !self[target].is_active() {
                    continue;
                }
                for constraint in (0..self.problem.number_constraints()).map(ConstraintIndex) {
                    for edge_index in 0..self[target].number_children() {
                        if edge_index == 0 {
                            self.problem[constraint].reset_property_bottom_up(target);
                        }
                        let edge = self.nodes[layer][node_index].child_edge_at(edge_index);
                        let source = self[edge].to();
                        let assignment = self.problem[decision].value(self[edge].assignment());
                        self.problem[constraint].update_property_bottom_up(source, target, assignment);
                    }
                    for edge_index in (0..self[target].number_children()).rev() {
                        let edge = self.nodes[layer][node_index].child_edge_at(edge_index);
                        let source = self[edge].to();
                        let assignment = self.problem[decision].value(self[edge].assignment());
                        if self.problem[constraint].is_layer_in_scope(layer) && self.problem[constraint].is_assignment_invalid(target, source, decision, assignment) {
                            self[target].swap_remove_child_edge(edge_index);
                            if self[target].number_children() == 0 {
                                self.remove_node(target);
                            }
                            self[source].remove_parent_edge(edge);
                            if self[source].number_parents() == 0 {
                                self.remove_node(source);
                            }
                            self[edge].deactivate();
                        }
                    }
                }
            }
        }
    }

    fn remove_node(&mut self, node: NodeIndex) {
        if !self[node].is_active() {
            return;
        }
        self[node].deactivate();
        for i in 0..self[node].number_parents() {
            let edge = self[node].parent_edge_at(i);
            self[edge].deactivate();
            let parent = self[edge].from();
            self[parent].remove_child_edge(edge);
            if self[parent].number_children() == 0 {
                self.remove_node(parent);
            }
        }
        for i in 0..self[node].number_children() {
            let edge = self[node].child_edge_at(i);
            self[edge].deactivate();
            let child = self[edge].to();
            self[child].remove_parent_edge(edge);
            if self[child].number_parents() == 0 {
                self.remove_node(child);
            }
        }
    }

    fn merge_layer(&mut self, layer :usize) {
        let number_nodes = self.nodes[layer].len();
        if number_nodes <= self.max_width {
            return;
        }
        let node_ranks = self.merge_heuristic.rank_nodes(self, layer);
        let into = NodeIndex(layer, node_ranks[self.max_width - 1].1);
        self[into].set_relaxed(true);
        for i in self.max_width..number_nodes {
            let from = NodeIndex(layer, node_ranks[i].1);
            self.merge_nodes(from, into);
            self[from].deactivate();
        }
    }

    fn merge_nodes(&mut self, from: NodeIndex, into: NodeIndex) {
        self[into].set_relaxed(true);
        for i in 0..self[from].number_parents() {
            let edge = self[from].parent_edge_at(i);
            self[edge].set_to(into);
            self[into].add_parent_edge(edge);
        }

        let mut existing_children = FxHashSet::<(NodeIndex, ValueIndex)>::default();
        for i in 0..self[into].number_children() {
            let edge = self[into].child_edge_at(i);
            let child = self[edge].to();
            let assignment = self[edge].assignment();
            existing_children.insert((child, assignment));
        }

        for i in 0..self[from].number_children() {
            let edge = self[from].child_edge_at(i);
            let child = self[edge].to();
            let assignment = self[edge].assignment();
            if !existing_children.contains(&(child, assignment)) {
                self[edge].set_from(into);
                self[into].add_child_edge(edge);
            }
        }
    }

    fn clean(&mut self) {
        let mut map_node_index = FxHashMap::<NodeIndex, NodeIndex>::default();
        map_node_index.insert(self.root, self.root);
        map_node_index.insert(self.sink, self.sink);
        for layer in 1..self.nodes.len() - 1 {
            let mut new_index = 0;
            for index in 0..self.nodes[layer].len() {
                if self.nodes[layer][index].is_active() {
                    map_node_index.insert(NodeIndex(layer, index), NodeIndex(layer, new_index));
                    self.nodes[layer].swap(new_index, index);
                    new_index += 1;
                }
            }
            self.nodes[layer].truncate(new_index);
        }
        let mut map_edge_index = FxHashMap::<EdgeIndex, EdgeIndex>::default();
        for layer in 0..self.edges.len() {
            let mut new_index = 0;
            for index in 0..self.edges[layer].len() {
                let from = self.edges[layer][index].from();
                let to = self.edges[layer][index].to();
                if self.edges[layer][index].is_active() && !(map_node_index.get(&from).is_none() || map_node_index.get(&to).is_none()) {
                    map_edge_index.insert(EdgeIndex(layer, index), EdgeIndex(layer, new_index));
                    self.edges[layer].swap(new_index, index);
                    new_index += 1;
                }
            }
            self.edges[layer].truncate(new_index);
        }

        for layer in 0..self.nodes.len() {
            for index in 0..self.nodes[layer].len() {
                self.nodes[layer][index].update_edge_indices(&map_edge_index);
            }
            if layer > 0 {
                for index in 0..self.edges[layer - 1].len() {
                    self.edges[layer - 1][index].update_node_indices(&map_node_index);
                }
            }
        }
    }

    pub fn number_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.nodes[layer].len()
    }

    pub fn number_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn number_layers(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_solution(&self) -> Option<Vec<isize>> {
        let mut assignment = vec![0; self.nodes.len() - 1];
        let root = NodeIndex(0, 0);
        if self.extract_solution(root, &mut assignment) {
            Some(assignment)
        } else {
            None
        }
    }

    fn extract_solution(&self, node: NodeIndex, assignment: &mut Vec<isize>) -> bool {
        let layer = node.0;
        if layer == self.nodes.len() - 1 {
            return true;
        }
        if self[node].is_relaxed() {
            return false;
        }
        let variable = self.order[layer];
        for edge in self[node].iter_children() {
            if !self[edge].is_active() {
                continue;
            }
            let to = self[edge].to();
            let value = self.problem[variable].value(self[edge].assignment());
            assignment[*variable] = value;
            if self.extract_solution(to, assignment) {
                return true;
            }
        }
        false
    }
    
    pub fn is_solution(&self, assignment: &[isize])  -> bool {
        for constraint in self.problem.iter_constraints() {
            if !self.problem[constraint].is_satisfied(assignment) {
                return false;
            }
        }
        true
    }

    pub fn proportion_satisfied_constraints(&self, assignment: &[isize])  -> f64 {
        let number_constraints = self.problem.number_constraints() as f64;
        let satisfied = self.problem.iter_constraints().filter(|&constraint| self.problem[constraint].is_satisfied(assignment)).count() as f64;
        satisfied / number_constraints
    }

    pub fn is_unsat(&self) -> bool {
        self.unsat
    }

    pub fn set_probabilities(&mut self, probabilities: &[Vec<f64>]) {
        for variable in (0..self.number_layers() - 1).map(VariableIndex) {
            self.problem[variable].set_probabilities(&probabilities[variable.0]);
        }
    }

    pub fn sample(&self) -> Vec<isize> {
        let mut assignments = vec![0; self.number_layers() - 1];
        RNG.with_borrow_mut(|rng| {
            let mut cur_node = self.root;
            while cur_node != self.sink {
                let NodeIndex(layer, _) = cur_node;
                let variable = self.order[layer];
                let mut total_probability_mass = 0.0;
                for edge in self[cur_node].iter_children() {
                    let assignment = self[edge].assignment();
                    total_probability_mass += self.problem[variable].probability(assignment);
                }

                let mut target = rng.random_range(0.0..total_probability_mass);
                for edge in self[cur_node].iter_children() {
                    let assignment = self[edge].assignment();
                    target -= self.problem[variable].probability(assignment);
                    if target <= 0.0 {
                        assignments[variable.0] = self.problem[variable].value(assignment);
                        cur_node = self[edge].to();
                    }
                }
                if cur_node.0 != layer {
                    panic!("No edge sampled at layer {}", layer);
                }
            }
        });
        assignments
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

        for (layer, variable) in self.order.iter().copied().enumerate() {
            layer_labels.push_str(&format!("\tL{} [shape=plaintext, label=\"x{}\"];\n", layer, variable.0));
        }

        for layer in 0..self.nodes.len() {
            for index in (0..self.nodes[layer].len()).filter(|i| self[NodeIndex(layer, *i)].is_active()) {
                let id = format!("{{rank=same; N{}_{} [shape=point,width=0.05] L{}}}", layer, index, layer);
                subgraph.push_str(&format!("\t{id};\n"));
            }
        }

        for layer in 0..self.edges.len() {
            let variable = self.order[layer];
            for edge in self.edges[layer].iter().filter(|e| e.is_active()) {
                let NodeIndex(layer_from, index_from) = edge.from();
                let NodeIndex(layer_to, index_to) = edge.to();
                let assignment = self.problem[variable].value(edge.assignment());
                subgraph.push_str(&format!("\tN{}_{} -> N{}_{} [penwidth=1, label=\"{}\"];\n", layer_from, index_from, layer_to, index_to, assignment));
            }
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
}

impl std::ops::Index<EdgeIndex> for Mdd {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0][index.1]
    }
}

impl std::ops::IndexMut<EdgeIndex> for Mdd {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edges[index.0][index.1]
    }
}

impl std::ops::Index<NodeIndex> for Mdd {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0][index.1]
    }
}

impl std::ops::IndexMut<NodeIndex> for Mdd {

    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index.0][index.1]
    }
}

impl std::fmt::Debug for Mdd {

    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.unsat {
            write!(f, "UNSAT")?;
        } else {
            // First, we print the variable order
            let vorder_str = self.order.iter().map(|variable| format!("{}", variable.0)).collect::<Vec<String>>().join(" ");
            writeln!(f, "{}", vorder_str)?;
            let mut number_nodes = 0;
            let mut number_edges = 0;
            for layer in 0..self.nodes.len() {
                number_nodes += self.nodes[layer].len();
                if layer > 0 {
                    number_edges += self.edges[layer - 1].len();
                }
            }
            writeln!(f, "{} {}", number_nodes, number_edges)?;
            let mut map_node_id = FxHashMap::<NodeIndex, usize>::default();
            for layer in 0..self.nodes.len() {
                for i in 0..self.nodes[layer].len() {
                    let node = NodeIndex(layer, i);
                    let id = map_node_id.len();
                    writeln!(f, "{} {}", id, layer)?;
                    map_node_id.insert(node, id);
                }
            }
            for layer in 0..self.edges.len() {
                let variable = self.order[layer];
                for i in 0..self.edges[layer].len() {
                    let source = map_node_id[&self.edges[layer][i].from()];
                    let to = map_node_id[&self.edges[layer][i].to()];
                    let assignment = self.problem[variable].value(self.edges[layer][i].assignment());
                    if layer < self.edges.len() - 1 || i < self.edges[layer].len() - 1 {
                        writeln!(f, "{} {} {}", source, to, assignment)?;
                    } else {
                        write!(f, "{} {} {}", source, to, assignment)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod test_mdd {

    use crate::modelling::*;
    use crate::mdd::*;
    use crate::mdd::heuristics::*;

    pub fn get_all_solutions(mdd: &Mdd) -> Vec<Vec<isize>> {
        let mut solutions: Vec<Vec<isize>> = vec![];
        let mut current_solution: Vec<isize> = vec![0; mdd.number_layers() - 1];
        let root = NodeIndex(0, 0);
        _get_all_solutions(mdd, root, &mut solutions, &mut current_solution);
        solutions
    }

    fn _get_all_solutions(mdd: &Mdd, node: NodeIndex, solutions: &mut Vec<Vec<isize>>, current_solution: &mut Vec<isize>) {
        let NodeIndex(layer, _) = node;
        if layer == mdd.number_layers() - 1 {
            solutions.push(current_solution.clone());
            return;
        }
        let variable = mdd.decision_at_layer(layer);
        for edge in mdd[node].iter_children() {
            if mdd[edge].is_active() {
                let child = mdd[edge].to();
                let assignment = mdd.problem[variable].value(mdd[edge].assignment());
                current_solution[*variable] = assignment;
                _get_all_solutions(mdd, child, solutions, current_solution);
            }
        }
    }

    pub fn is_solution(solution: Vec<isize>, all_solutions: &[Vec<isize>]) -> bool {
        for sol in all_solutions.iter() {
            let mut eq = true;
            for i in 0..sol.len() {
                if sol[i] != solution[i] {
                    eq = false;
                    break
                }
            }
            if eq {
                return true;
            }
        }
        false
    }

    #[test]
    pub fn mdd_creation() {
        let mut problem = Problem::default();
        problem.add_variable(vec![0, 1], None);
        problem.add_variable(vec![0, 1], None);
        problem.add_variable(vec![0, 1, 2], None);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2*2*3);
        assert!(is_solution(vec![0, 0, 0], &solutions));
        assert!(is_solution(vec![0, 0, 1], &solutions));
        assert!(is_solution(vec![0, 0, 2], &solutions));
        assert!(is_solution(vec![0, 1, 0], &solutions));
        assert!(is_solution(vec![0, 1, 1], &solutions));
        assert!(is_solution(vec![0, 1, 2], &solutions));
        assert!(is_solution(vec![1, 0, 0], &solutions));
        assert!(is_solution(vec![1, 0, 1], &solutions));
        assert!(is_solution(vec![1, 0, 2], &solutions));
        assert!(is_solution(vec![1, 1, 0], &solutions));
        assert!(is_solution(vec![1, 1, 1], &solutions));
        assert!(is_solution(vec![1, 1, 2], &solutions));
    }

    #[test]
    pub fn mdd_refine() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![1, 2], None);

        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        // TODO assert?
    }
}
