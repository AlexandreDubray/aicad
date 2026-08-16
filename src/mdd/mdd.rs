use super::heuristics::*;
use super::*;
use crate::constraints::*;
use crate::modelling::*;
use crate::utils::MemoryReport;

use rand;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::cell::RefCell;

use rustc_hash::{FxHashMap, FxHashSet};
use std::fs;
use std::sync::Arc;

thread_local! {
    static RNG: RefCell<Xoshiro256Plus> = RefCell::new(Xoshiro256Plus::from_rng(&mut rand::rng()));
}

/// Structure for the MDD. The MDD is organised in layers (one layer per variable in the problem)
/// and each layer contains the necessary information to propagate the constraint and generate
/// solutions.
pub struct Mdd {
    /// Variables in the MDD scope. A variable is in the scope of the MDD if it is in the scope of
    /// one of the compiled constraint
    scope: Vec<VariableIndex>,
    /// Constraint being compiled in this MDD. Each MDD can compile any subset of the problem's
    /// constraints.
    constraints: Vec<Box<dyn Constraint>>,
    /// Problem being compiled
    problem: Arc<Problem>,
    /// Nodes of the MDD.
    nodes: Vec<Vec<Node>>,
    /// Edges of the MDD.
    edges: Vec<Vec<Edge>>,
    /// Branching order
    order: Vec<VariableIndex>,
    /// Heuristic used to score nodes during merging operation
    merge_heuristic: MergeHeuristic,
    /// Heuristic to select nodes to split
    select_heuristic: SelectHeuristic,
    /// Is the MDD unsat
    unsat: bool,
    /// Root of the MDD
    root: NodeIndex,
    /// Sink of the mdd
    sink: NodeIndex,
    /// Top down properties of the MDD's constraints
    top_down_properties: Vec<Vec<Vec<Box<dyn ConstraintProperty>>>>,
    /// Bottom up properties of the MDD's constraints
    bottom_up_properties: Vec<Vec<Vec<Box<dyn ConstraintProperty>>>>,
}

impl Mdd {
    /// Creates a new MDD for the given problem and variable ordering. The ordering array gives,
    /// for each variable, the layer at which it is branched on.
    pub fn new(
        problem: Arc<Problem>,
        order: OrderingHeuristic,
        merge_heuristic: MergeHeuristic,
        select_heuristic: SelectHeuristic,
        constraints: &[ConstraintIndex],
    ) -> Self {
        let mut in_scope = vec![false; problem.number_variables()];
        let mut mdd_scope = vec![];
        for constraint in constraints.iter().copied() {
            for variable in problem[constraint].iter_scope() {
                if !in_scope[*variable] {
                    mdd_scope.push(variable);
                    in_scope[*variable] = true;
                }
            }
        }
        let number_layers = mdd_scope.len();
        let constraints = constraints
            .iter()
            .map(|&constraint| problem[constraint].clone() as Box<dyn Constraint>)
            .collect::<Vec<Box<dyn Constraint>>>();
        let mut mdd = Self {
            scope: mdd_scope,
            constraints: constraints,
            nodes: vec![vec![]; number_layers + 1],
            edges: vec![vec![]; number_layers],
            order: vec![],
            merge_heuristic,
            select_heuristic,
            problem,
            unsat: false,
            root: NodeIndex(0, 0),
            sink: NodeIndex(number_layers, 0),
            top_down_properties: vec![vec![]; number_layers + 1],
            bottom_up_properties: vec![vec![]; number_layers + 1],
        };

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer in 0..mdd.number_layers() {
            mdd.add_node(layer, layer != 0);
        }

        // Set the variable order in the MDD given the heuristics
        // We get for each layer its decision variable
        let var_order = order.get_order(&mdd.problem, &mdd.scope);
        for i in 0..mdd.constraints.len() {
            mdd.constraints[i].update_variable_ordering(&var_order);
        }
        mdd.order = var_order;

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

        let is_root = layer == 0;
        let is_sink = layer == self.sink.0;

        self.top_down_properties[layer].push(
            (0..self.constraints.len())
                .map(|i| {
                    if is_root {
                        self.constraints[i].empty_property()
                    } else {
                        self.constraints[i].identity_property()
                    }
                })
                .collect(),
        );
        self.bottom_up_properties[layer].push(
            (0..self.constraints.len())
                .map(|i| {
                    if is_sink {
                        self.constraints[i].empty_property()
                    } else {
                        self.constraints[i].identity_property()
                    }
                })
                .collect(),
        );
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

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn iter_constraints(&self) -> impl Iterator<Item = &Box<dyn Constraint>> {
        self.constraints.iter()
    }

    // --- split and refine strategy ---- //

    /// Refines the MDD allowing max_width nodes in each layer
    pub fn refine(&mut self, max_width: usize) {
        if self.unsat {
            return;
        }
        for layer in 1..self.nodes.len() - 1 {
            if self.number_nodes_in_layer(layer) == max_width {
                continue;
            }
            if let Some(node) = self.select_heuristic.select_node(self, layer) {
                self.split_node(node);
                self.propagate_constraints();
                if !self[self.root].is_active() || !self[self.sink].is_active() {
                    self.unsat = true;
                    return;
                }
                self.collapse();
                self.merge_layer(layer, max_width);
                self.clean();
            }
        }
    }

    fn split_node(&mut self, node: NodeIndex) -> Vec<NodeIndex> {
        let mut nodes = vec![node];
        let NodeIndex(layer, _) = node;
        let n = self[node].number_parents();
        let outgoing_assignments = self[node]
            .iter_children()
            .filter(|edge| self[*edge].is_active())
            .map(|edge| (self[edge].to(), self[edge].assignment()))
            .collect::<Vec<(NodeIndex, ValueIndex)>>();
        self[node].set_relaxed(false);
        for i in (1..n).rev() {
            let new_node = self.add_node(layer, false);
            nodes.push(new_node);
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
        nodes
    }

    /// Recomputes every node's top-down and bottom-up constraint properties, then removes any
    /// edge that `is_assignment_invalid` rules out given the freshly recomputed properties on
    /// both of its endpoints.
    ///
    /// The two passes must run in this order: bottom-up filtering compares a node's *already
    /// up to date* top-down property against its child's freshly-computed bottom-up property,
    /// so the top-down pass has to be complete first.
    pub fn propagate_constraints(&mut self) {
        self.update_top_down_properties();
        self.update_bottom_up_properties_and_filter_edges();
    }

    /// Recomputes `top_down_properties` for every layer but the root (layer 0), whose top-down
    /// property is permanently `empty_property()` - see `add_node`.
    fn update_top_down_properties(&mut self) {
        for layer in 1..self.nodes.len() {
            let variable = self.order[layer - 1];
            for i in 0..self.nodes[layer].len() {
                let target = NodeIndex(layer, i);
                for constraint_index in 0..self.constraints.len() {
                    self.top_down_properties[layer][i][constraint_index] =
                        self.fold_property_over_parents(target, variable, constraint_index);
                }
            }
        }
    }

    /// Folds `target`'s parent edges through `identity_property()`, using each parent's own
    /// (already up to date) top-down property.
    fn fold_property_over_parents(
        &self,
        target: NodeIndex,
        variable: VariableIndex,
        constraint_index: usize,
    ) -> Box<dyn ConstraintProperty> {
        let mut property = self.constraints[constraint_index].identity_property();
        for j in 0..self[target].number_parents() {
            let edge = self[target].parent_edge_at(j);
            let NodeIndex(source_layer, source_index) = self[edge].from();
            let in_scope = self.constraints[constraint_index].is_layer_in_scope(source_layer);
            let assignment = self.problem[variable].value(self[edge].assignment());
            let parent_property =
                self.top_down_properties[source_layer][source_index][constraint_index].as_ref();
            property.update(parent_property, assignment, in_scope);
        }
        property
    }

    /// Recomputes `bottom_up_properties` for every layer but the sink, whose bottom-up property
    /// is permanently `empty_property()`. Once a node's bottom-up property (and its children's,
    /// since the pass runs layer by layer from the sink up) is up to date, edges out of it are
    /// filtered against the constraints in scope at that layer.
    fn update_bottom_up_properties_and_filter_edges(&mut self) {
        for layer in (0..self.nodes.len() - 1).rev() {
            let variable = self.order[layer];
            for node_index in 0..self.nodes[layer].len() {
                let target = NodeIndex(layer, node_index);
                if !self[target].is_active() {
                    continue;
                }
                for constraint_index in 0..self.constraints.len() {
                    self.bottom_up_properties[layer][node_index][constraint_index] =
                        self.fold_property_over_children(target, variable, constraint_index);
                    if self.constraints[constraint_index].is_layer_in_scope(layer) {
                        self.filter_invalid_edges(target, variable, constraint_index);
                    }
                }
            }
        }
    }

    /// Folds `target`'s child edges through `identity_property()`, using each child's own
    /// (already up to date) bottom-up property. This is exactly what `target`'s bottom-up
    /// property should become.
    fn fold_property_over_children(
        &self,
        target: NodeIndex,
        variable: VariableIndex,
        constraint_index: usize,
    ) -> Box<dyn ConstraintProperty> {
        let NodeIndex(layer, _) = target;
        let in_scope = self.constraints[constraint_index].is_layer_in_scope(layer);
        let mut property = self.constraints[constraint_index].identity_property();
        for edge_index in 0..self[target].number_children() {
            let edge = self[target].child_edge_at(edge_index);
            let NodeIndex(child_layer, child_index) = self[edge].to();
            let assignment = self.problem[variable].value(self[edge].assignment());
            let child_property =
                self.bottom_up_properties[child_layer][child_index][constraint_index].as_ref();
            property.update(child_property, assignment, in_scope);
        }
        property
    }

    /// Removes every child edge of `target` that `constraint_index` rules out, given `target`'s
    /// top-down property and the child's bottom-up property (both assumed up to date). Removing
    /// an edge can empty a node's remaining parents/children, in which case that node is removed
    /// too (cascading through `remove_node`).
    fn filter_invalid_edges(
        &mut self,
        target: NodeIndex,
        variable: VariableIndex,
        constraint_index: usize,
    ) {
        let NodeIndex(layer, node_index) = target;
        for edge_index in (0..self[target].number_children()).rev() {
            let edge = self[target].child_edge_at(edge_index);
            let child = self[edge].to();
            let NodeIndex(child_layer, child_index) = child;
            let assignment = self.problem[variable].value(self[edge].assignment());
            let parent_property =
                self.top_down_properties[layer][node_index][constraint_index].as_ref();
            let child_property =
                self.bottom_up_properties[child_layer][child_index][constraint_index].as_ref();
            let invalid = self.constraints[constraint_index].is_assignment_invalid(
                parent_property,
                child_property,
                layer,
                assignment,
            );
            if !invalid {
                continue;
            }
            self[target].swap_remove_child_edge(edge_index);
            if self[target].number_children() == 0 {
                self.remove_node(target);
            }
            self[child].remove_parent_edge(edge);
            if self[child].number_parents() == 0 {
                self.remove_node(child);
            }
            self[edge].deactivate();
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

    fn collapse(&mut self) {
        for layer in 1..self.nodes.len() - 1 {
            let mut map: FxHashMap<MergeKey, NodeIndex> = FxHashMap::default();
            for index in 0..self.nodes[layer].len() {
                let node = NodeIndex(layer, index);
                if !self[node].is_active() {
                    continue;
                }
                let key = MergeKey {
                    td_properties: &self.top_down_properties[layer][index],
                    bu_properties: &self.bottom_up_properties[layer][index],
                };
                if let Some(&primary_node) = map.get(&key) {
                    let NodeIndex(primary_layer, primary_index) = primary_node;

                    for i in 0..self[node].number_parents() {
                        let EdgeIndex(edge_layer, edge_index) = self[node].parent_edge_at(i);
                        self.edges[edge_layer][edge_index].set_to(primary_node);
                        self.nodes[primary_layer][primary_index]
                            .add_parent_edge(EdgeIndex(edge_layer, edge_index));
                    }

                    let mut existing_children = FxHashSet::<(NodeIndex, ValueIndex)>::default();
                    for i in 0..self[primary_node].number_children() {
                        let edge = self[primary_node].child_edge_at(i);
                        let child = self[edge].to();
                        let assignment = self[edge].assignment();
                        existing_children.insert((child, assignment));
                    }

                    for i in 0..self[node].number_children() {
                        let edge = self[node].child_edge_at(i);
                        let EdgeIndex(edge_layer, edge_index) = edge;
                        let child = self[edge].to();
                        let assignment = self[edge].assignment();
                        if !existing_children.contains(&(child, assignment)) {
                            self.edges[edge_layer][edge_index].set_to(primary_node);
                            self.nodes[primary_layer][primary_index].add_child_edge(edge);
                        }
                    }
                    self.nodes[layer][index].deactivate();
                } else {
                    map.insert(key, node);
                }
            }
        }
    }

    fn merge_layer(&mut self, layer: usize, max_width: usize) {
        let number_nodes = self.nodes[layer].len();
        if number_nodes <= max_width {
            return;
        }
        let node_ranks = self.merge_heuristic.rank_nodes(self, layer);
        let active_nodes = node_ranks.len();
        if active_nodes <= max_width {
            return;
        }
        if !self.merge_heuristic.bucket_merge() {
            let into = node_ranks[active_nodes - max_width].1;
            self[into].set_relaxed(true);
            for i in 0..active_nodes - max_width {
                let from = node_ranks[i].1;
                self.merge_nodes(from, into);
                self[from].deactivate();
            }
        } else {
            let q = node_ranks.len() / max_width;
            let r = node_ranks.len() % max_width;
            let mut bucket_sizes = vec![q; max_width - r];
            bucket_sizes.extend(vec![q + 1; r]);
            let mut i = 0;
            for _ in 0..max_width - r {
                let into = node_ranks[i].1;
                for j in (i + 1)..(i + q) {
                    let from = node_ranks[j].1;
                    self.merge_nodes(from, into);
                    self[from].deactivate();
                }
                i += q;
            }
            for _ in 0..r {
                let into = node_ranks[i].1;
                for j in (i + 1)..(i + q + 1) {
                    let from = node_ranks[j].1;
                    self.merge_nodes(from, into);
                    self[from].deactivate();
                }
                i += q + 1;
            }
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
            self.top_down_properties[layer].truncate(new_index);
            self.bottom_up_properties[layer].truncate(new_index);
        }
        let mut map_edge_index = FxHashMap::<EdgeIndex, EdgeIndex>::default();
        for layer in 0..self.edges.len() {
            let mut new_index = 0;
            for index in 0..self.edges[layer].len() {
                let from = self.edges[layer][index].from();
                let to = self.edges[layer][index].to();
                if self.edges[layer][index].is_active()
                    && !(map_node_index.get(&from).is_none() || map_node_index.get(&to).is_none())
                {
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
        self.nodes.iter().map(|layer| layer.len()).sum::<usize>()
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
        let sink = NodeIndex(self.nodes.len() - 1, 0);
        if self.extract_solution(sink, &mut assignment) {
            Some(assignment)
        } else {
            None
        }
    }

    fn extract_solution(&self, node: NodeIndex, assignment: &mut Vec<isize>) -> bool {
        let layer = node.0;
        if layer == 0 {
            return true;
        }
        for edge in self[node].iter_parents() {
            let from = self[edge].from();
            if !self[from].is_relaxed() {
                let variable = self.order[layer - 1];
                let value = self.problem[variable].value(self[edge].assignment());
                assignment[*variable] = value;
                return self.extract_solution(from, assignment);
            }
        }
        false
    }

    pub fn is_unsat(&self) -> bool {
        self.unsat
    }

    pub fn set_probabilities(&mut self, _probabilities: &[Vec<f64>]) {
        panic!("TODO");
    }

    pub fn get_edge_probability(&self, edge: EdgeIndex) -> f64 {
        let EdgeIndex(source_layer, _) = edge;
        let variable = self.decision_at_layer(source_layer);
        let assignment = self[edge].assignment();
        self.problem[variable].probability(assignment)
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
                if cur_node.0 == layer {
                    panic!("No edge sampled at layer {}", layer);
                }
            }
        });
        assignments
    }

    /// Returns a topological order of the MDD as a vector of (edge, src, variable, value)
    pub fn topological_order(&self) -> Vec<(usize, usize, usize, isize)> {
        let mut toporder: Vec<(usize, usize, usize, isize)> = vec![];
        let mut toporder_shift = vec![0; self.nodes.len()];
        for layer in 1..self.nodes.len() {
            toporder_shift[layer] += toporder_shift[layer - 1] + self.nodes[layer - 1].len();
        }
        for layer in 0..self.edges.len() {
            for index in 0..self.edges[layer].len() {
                let edge = &self.edges[layer][index];
                let variable = self.order[layer];
                let NodeIndex(from_layer, from_index) = edge.from();
                let NodeIndex(to_layer, to_index) = edge.to();
                let assignment = self.problem[variable].value(edge.assignment());
                let source_toporder = toporder_shift[from_layer] + from_index;
                let to_toporder = toporder_shift[to_layer] + to_index;
                toporder.push((source_toporder, to_toporder, variable.0, assignment));
            }
        }
        toporder
    }
}

/* ---- Various helper implementation to make life easier ---- */

impl Mdd {
    pub fn as_graphviz(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph {\nrankdir=TD;\ntranksep = 3;\n\n");

        let mut subgraph = String::new();
        subgraph.push_str("subgraph mdd {\n");
        let mut layer_labels = String::new();
        layer_labels.push_str("subgraph labels {\n");

        for (layer, variable) in self.order.iter().copied().enumerate() {
            layer_labels.push_str(&format!(
                "\tL{} [shape=plaintext, label=\"x{}\"];\n",
                layer, variable.0
            ));
        }

        for layer in 0..self.nodes.len() {
            for index in
                (0..self.nodes[layer].len()).filter(|i| self[NodeIndex(layer, *i)].is_active())
            {
                let id = format!(
                    "{{rank=same; N{}_{} [shape=point,width=0.05] L{}}}",
                    layer, index, layer
                );
                subgraph.push_str(&format!("\t{id};\n"));
            }
        }

        for layer in 0..self.edges.len() {
            let variable = self.order[layer];
            for edge in self.edges[layer].iter().filter(|e| e.is_active()) {
                let NodeIndex(layer_from, index_from) = edge.from();
                let NodeIndex(layer_to, index_to) = edge.to();
                let assignment = self.problem[variable].value(edge.assignment());
                subgraph.push_str(&format!(
                    "\tN{}_{} -> N{}_{} [penwidth=1, label=\"{}\"];\n",
                    layer_from, index_from, layer_to, index_to, assignment
                ));
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

    pub fn show_memory_footprint(&self) {
        log::info!(
            "Memory report for mdd with {} nodes",
            self.nodes.iter().map(|layer| layer.len()).sum::<usize>()
        );
        let report = MemoryReport::build(self.problem.constraints().iter());
        report.print(80);
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
            let vorder_str = self
                .order
                .iter()
                .map(|variable| format!("{}", variable.0))
                .collect::<Vec<String>>()
                .join(" ");
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
                    let assignment =
                        self.problem[variable].value(self.edges[layer][i].assignment());
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

    use crate::mdd::heuristics::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use std::sync::Arc;

    pub fn get_all_solutions(mdd: &Mdd) -> Vec<Vec<isize>> {
        let mut solutions: Vec<Vec<isize>> = vec![];
        let mut current_solution: Vec<isize> = vec![0; mdd.number_layers() - 1];
        let root = NodeIndex(0, 0);
        _get_all_solutions(mdd, root, &mut solutions, &mut current_solution);
        solutions
    }

    fn _get_all_solutions(
        mdd: &Mdd,
        node: NodeIndex,
        solutions: &mut Vec<Vec<isize>>,
        current_solution: &mut Vec<isize>,
    ) {
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
                    break;
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
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        // Mdd::new's scope is the union of the *given* constraints' scopes, so an MDD covering
        // all 3 (otherwise free) variables needs a constraint that pulls them into scope. An
        // unbounded gcc (no value bounds at all) imposes no actual restriction - see
        // `test_no_bound_restriction` in gcc.rs for the same pattern - so it's a stand-in for
        // "no real constraint" that still lets every variable enumerate its full domain.
        gcc(&mut problem, vec![x, y, z], vec![]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2 * 2 * 3);
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        // TODO assert?
    }
}
