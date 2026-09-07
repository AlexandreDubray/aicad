use super::heuristics::*;
use super::*;
use crate::constraints::*;
use crate::modelling::*;
use crate::utils::MemoryReport;

use num_bigint::BigUint;

use rand;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::cell::RefCell;

use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
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
                        self.constraints[i].empty_property_backward()
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
        let mut seen_fingerprints: HashSet<u64> = HashSet::default();
        seen_fingerprints.insert(self.fingerprint());
        loop {
            let mut any_split = false;
            for layer in 1..self.nodes.len() - 1 {
                if self.number_nodes_in_layer(layer) == max_width {
                    continue;
                }
                if let Some(node) = self.select_heuristic.select_node(self, layer) {
                    any_split = true;
                    let new_nodes = self.split_node(node);
                    for &new_node in &new_nodes {
                        self[new_node].set_property_flag();
                    }
                    let deepest_touched = self.update_top_down(layer);
                    for &new_node in &new_nodes {
                        self[new_node].set_property_flag();
                    }
                    self.update_bottom_up(deepest_touched);
                    if !self[self.root].is_active() || !self[self.sink].is_active() {
                        self.unsat = true;
                        return;
                    }
                    self.collapse();
                    self.merge_layer(layer, max_width);
                    self.collapse();
                    self.clean();
                }
            }
            if !any_split {
                break;
            }
            if !seen_fingerprints.insert(self.fingerprint()) {
                break;
            }
        }
    }

    fn fingerprint(&self) -> u64 {
        let mut total: u64 = 0;
        for layer in 0..self.nodes.len() {
            let mut layer_signature: u64 = 0;
            for index in 0..self.nodes[layer].len() {
                if !self.nodes[layer][index].is_active() {
                    continue;
                }
                let key = MergeKey {
                    td_properties: &self.top_down_properties[layer][index],
                    bu_properties: &self.bottom_up_properties[layer][index],
                };
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                layer_signature ^= hasher.finish();
            }
            total = total
                .wrapping_mul(1_000_000_007)
                .wrapping_add(layer_signature);
        }
        total
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

    fn update_top_down(&mut self, start_layer: usize) -> usize {
        let mut deepest_changed_layer = start_layer.max(1) - 1;
        for layer in start_layer.max(1)..self.nodes.len() {
            let parent_variable = self.order[layer - 1];
            let mut any_changed = false;
            for index in 0..self.nodes[layer].len() {
                let target = NodeIndex(layer, index);
                if !self[target].is_property_flag() {
                    continue;
                }
                self[target].unset_property_flag();
                if !self[target].is_active() {
                    continue;
                }
                let mut changed = false;
                for constraint_index in 0..self.constraints.len() {
                    let new_property =
                        self.fold_property_over_parents(target, parent_variable, constraint_index);
                    if !ConstraintProperty::eq(
                        new_property.as_ref(),
                        self.top_down_properties[layer][index][constraint_index].as_ref(),
                    ) {
                        changed = true;
                    }
                    self.top_down_properties[layer][index][constraint_index] = new_property;
                }
                if !changed {
                    continue;
                }
                any_changed = true;
                deepest_changed_layer = deepest_changed_layer.max(layer);
                if layer < self.nodes.len() - 1 {
                    let children_before: Vec<NodeIndex> = self[target]
                        .iter_children()
                        .map(|edge| self[edge].to())
                        .collect();
                    let variable = self.order[layer];
                    for constraint_index in 0..self.constraints.len() {
                        if self.constraints[constraint_index].is_layer_in_scope(layer) {
                            self.filter_invalid_edges(target, variable, constraint_index);
                        }
                    }
                    for child in children_before {
                        if self[child].is_active() {
                            self[child].set_property_flag();
                        }
                    }
                }
                self[target].set_property_flag();
            }
            if !any_changed {
                break;
            }
        }
        deepest_changed_layer
    }

    fn update_bottom_up(&mut self, start_layer: usize) {
        for layer in (0..=start_layer.min(self.nodes.len() - 2)).rev() {
            let variable = self.order[layer];
            let mut any_changed = false;
            for index in 0..self.nodes[layer].len() {
                let target = NodeIndex(layer, index);
                if !self[target].is_property_flag() {
                    continue;
                }
                self[target].unset_property_flag();
                if !self[target].is_active() {
                    continue;
                }
                let mut changed = false;
                for constraint_index in 0..self.constraints.len() {
                    let new_property =
                        self.fold_property_over_children(target, variable, constraint_index);
                    if !ConstraintProperty::eq(
                        new_property.as_ref(),
                        self.bottom_up_properties[layer][index][constraint_index].as_ref(),
                    ) {
                        changed = true;
                    }
                    self.bottom_up_properties[layer][index][constraint_index] = new_property;
                }
                let parents_before: Vec<NodeIndex> = if layer > 0 {
                    self[target]
                        .iter_parents()
                        .map(|edge| self[edge].from())
                        .collect()
                } else {
                    Vec::new()
                };
                for constraint_index in 0..self.constraints.len() {
                    if self.constraints[constraint_index].is_layer_in_scope(layer) {
                        self.filter_invalid_edges(target, variable, constraint_index);
                    }
                }
                let target_removed = !self[target].is_active();
                if !changed && !target_removed {
                    continue;
                }
                any_changed = true;
                {
                    for parent in parents_before {
                        if self[parent].is_active() {
                            self[parent].set_property_flag();
                        }
                    }
                }
            }
            if !any_changed {
                break;
            }
        }
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
            property.update_backward(child_property, assignment, in_scope);
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
            let mut merges: Vec<(NodeIndex, NodeIndex)> = vec![];
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
                    merges.push((node, primary_node));
                } else {
                    map.insert(key, node);
                }
            }
            for (node, primary_node) in merges {
                self.merge_nodes_with_flag(node, primary_node, false);
            }
        }
    }

    /// Ranks this layer's active nodes according to `self.merge_heuristic`
    fn rank_nodes(&self, layer: usize) -> Vec<NodeIndex> {
        match self.merge_heuristic {
            MergeHeuristic::LessRelaxed => {
                let mut scores: Vec<(f64, NodeIndex)> = vec![];
                for i in 0..self.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, i);
                    if self[node].is_active() {
                        let number_parents = self[node].number_parents() as f64;
                        let number_parents_relaxed = self[node]
                            .iter_parents()
                            .map(|edge| self[edge].from())
                            .filter(|parent| !self[*parent].is_relaxed())
                            .count() as f64;
                        scores.push((number_parents_relaxed / number_parents, node));
                    }
                }
                scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                scores.into_iter().map(|(_, node)| node).collect()
            }
            MergeHeuristic::MostLikely => {
                let mut scores: Vec<(f64, NodeIndex)> = vec![];
                for i in 0..self.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, i);
                    if self[node].is_active() {
                        let number_parents = self[node].number_parents() as f64;
                        let aggregate_probabilities = self[node]
                            .iter_parents()
                            .map(|edge| self.get_edge_probability(edge))
                            .sum::<f64>();
                        scores.push((aggregate_probabilities / number_parents, node));
                    }
                }
                scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                scores.into_iter().map(|(_, node)| node).collect()
            }
            MergeHeuristic::StateSimilarity => {
                // One composite key per node: every constraint's top-down `order_key()` then
                // its bottom-up `order_key()`, in the MDD's own (local) constraint order.
                // Sorting lexicographically -- rather than collapsing to one weighted scalar --
                // avoids letting one constraint's scale (e.g. a Gcc key in the hundreds)
                // dominate another's (e.g. a Regular popcount in the single digits): nodes
                // agreeing on more constraints end up genuinely adjacent in the sort.
                let mut keyed: Vec<(Vec<f64>, NodeIndex)> = vec![];
                for i in 0..self.number_nodes_in_layer(layer) {
                    let node = NodeIndex(layer, i);
                    if !self[node].is_active() {
                        continue;
                    }
                    let mut key = Vec::with_capacity(2 * self.constraints.len());
                    for c in 0..self.constraints.len() {
                        key.extend(self.top_down_properties[layer][i][c].order_key());
                        key.extend(self.bottom_up_properties[layer][i][c].order_key());
                    }
                    keyed.push((key, node));
                }
                keyed.sort_unstable_by(|a, b| {
                    for (x, y) in a.0.iter().zip(b.0.iter()) {
                        match x.total_cmp(y) {
                            std::cmp::Ordering::Equal => continue,
                            other => return other,
                        }
                    }
                    std::cmp::Ordering::Equal
                });
                keyed.into_iter().map(|(_, node)| node).collect()
            }
        }
    }

    fn merge_layer(&mut self, layer: usize, max_width: usize) {
        let number_nodes = self.nodes[layer].len();
        if number_nodes <= max_width {
            return;
        }
        let node_ranks = self.rank_nodes(layer);
        let active_nodes = node_ranks.len();
        if active_nodes <= max_width {
            return;
        }
        if !self.merge_heuristic.bucket_merge() {
            let into = node_ranks[active_nodes - max_width];
            for i in 0..active_nodes - max_width {
                let from = node_ranks[i];
                self.merge_nodes(from, into);
            }
        } else {
            let q = node_ranks.len() / max_width;
            let r = node_ranks.len() % max_width;
            let mut bucket_sizes = vec![q; max_width - r];
            bucket_sizes.extend(vec![q + 1; r]);
            let mut i = 0;
            for _ in 0..max_width - r {
                let into = node_ranks[i];
                for j in (i + 1)..(i + q) {
                    let from = node_ranks[j];
                    self.merge_nodes(from, into);
                }
                i += q;
            }
            for _ in 0..r {
                let into = node_ranks[i];
                for j in (i + 1)..(i + q + 1) {
                    let from = node_ranks[j];
                    self.merge_nodes(from, into);
                }
                i += q + 1;
            }
        }
    }

    fn merge_nodes(&mut self, from: NodeIndex, into: NodeIndex) {
        self.merge_nodes_with_flag(from, into, true);
    }

    fn merge_nodes_with_flag(&mut self, from: NodeIndex, into: NodeIndex, mark_relaxed: bool) {
        let mut worklist = vec![(from, into, mark_relaxed)];
        while let Some((from, into, mark_relaxed)) = worklist.pop() {
            if from == into {
                continue;
            }
            if mark_relaxed {
                self[into].set_relaxed(true);
            }
            for i in 0..self[from].number_parents() {
                let edge = self[from].parent_edge_at(i);
                self[edge].set_to(into);
                self[into].add_parent_edge(edge);
            }

            let mut existing_children = FxHashMap::<ValueIndex, NodeIndex>::default();
            for i in 0..self[into].number_children() {
                let edge = self[into].child_edge_at(i);
                existing_children.insert(self[edge].assignment(), self[edge].to());
            }

            for i in 0..self[from].number_children() {
                let edge = self[from].child_edge_at(i);
                let assignment = self[edge].assignment();
                let child = self[edge].to();
                match existing_children.get(&assignment).copied() {
                    None => {
                        self[edge].set_from(into);
                        self[into].add_child_edge(edge);
                        existing_children.insert(assignment, child);
                    }
                    Some(existing_child) if existing_child == child => {
                        self[child].remove_parent_edge(edge);
                        self[edge].deactivate();
                    }
                    Some(existing_child) => {
                        self[child].remove_parent_edge(edge);
                        self[edge].deactivate();
                        worklist.push((child, existing_child, true));
                    }
                }
            }
            Self::merge_properties(&mut self.top_down_properties[from.0], from.1, into.1);
            Self::merge_properties(&mut self.bottom_up_properties[from.0], from.1, into.1);
            self[from].deactivate();
        }
    }

    fn merge_properties(
        properties: &mut [Vec<Box<dyn ConstraintProperty>>],
        from_index: usize,
        into_index: usize,
    ) {
        let hi = from_index.max(into_index);
        let lo = from_index.min(into_index);
        let (left, right) = properties.split_at_mut(hi);
        let (from_properties, into_properties) = if from_index < into_index {
            (&left[lo], &mut right[0])
        } else {
            (&right[0], &mut left[lo])
        };
        for (into_property, from_property) in into_properties.iter_mut().zip(from_properties.iter())
        {
            into_property.merge(from_property.as_ref());
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
                    self.top_down_properties[layer].swap(new_index, index);
                    self.bottom_up_properties[layer].swap(new_index, index);
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

    pub fn root(&self) -> NodeIndex {
        self.root
    }

    pub fn sink(&self) -> NodeIndex {
        self.sink
    }

    pub fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes[layer].len()).map(move |index| NodeIndex(layer, index))
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

    /// Counts the number of root-to-sink paths (i.e. encoded assignments) in the MDD, via a
    /// topological path-counting DP: `count[sink] = 1`, `count[node] = sum(count[child])` over
    /// its outgoing edges, computed layer by layer from the sink back to the root. This relies
    /// on the same invariant `topological_order()` does -- that `clean()` has already dropped
    /// every inactive node/edge, so every node and edge still present is active -- rather than
    /// checking `is_active()` on each.
    pub fn count_solutions(&self) -> BigUint {
        if self.unsat {
            return BigUint::from(0u32);
        }
        let last_layer = self.nodes.len() - 1;
        let mut counts: Vec<Vec<BigUint>> = self
            .nodes
            .iter()
            .map(|layer| vec![BigUint::from(0u32); layer.len()])
            .collect();
        for i in 0..self.nodes[last_layer].len() {
            counts[last_layer][i] = BigUint::from(1u32);
        }
        for layer in (0..last_layer).rev() {
            for i in 0..self.nodes[layer].len() {
                let node = NodeIndex(layer, i);
                let mut total = BigUint::from(0u32);
                for edge in self[node].iter_children() {
                    let NodeIndex(to_layer, to_index) = self[edge].to();
                    total += &counts[to_layer][to_index];
                }
                counts[layer][i] = total;
            }
        }
        counts[0][0].clone()
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
    use num_bigint::BigUint;
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

    #[test]
    pub fn merge_nodes_merges_children_that_collide_on_the_same_label() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        gcc(&mut problem, vec![x, y], vec![]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );

        let into = NodeIndex(1, 0);
        let c1 = NodeIndex(2, 0);
        let from = mdd.add_node(1, true);
        let c2 = mdd.add_node(2, true);
        let shared_assignment = ValueIndex(0);
        mdd.add_edge(1, from, c2, shared_assignment);

        mdd.merge_nodes_with_flag(from, into, true);

        assert!(!mdd[from].is_active());
        assert!(mdd[into].is_active());
        assert!(mdd[c1].is_active());
        assert!(!mdd[c2].is_active());

        let mut seen_assignments = std::collections::HashSet::new();
        for edge in mdd[into].iter_children() {
            if !mdd[edge].is_active() {
                continue;
            }
            assert!(seen_assignments.insert(mdd[edge].assignment()));
        }
        assert_eq!(seen_assignments.len(), 2);
    }

    #[test]
    pub fn incremental_refine_matches_full_recompute_on_all_different() {
        for n in 3..=7 {
            let mut problem = Problem::default();
            let vars: Vec<_> = (0..n)
                .map(|_| problem.add_variable((0..n as isize).collect(), None))
                .collect();
            all_different(&mut problem, vars);
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
            let solutions = get_all_solutions(&mdd);
            let mut factorial = 1usize;
            for k in 1..=n {
                factorial *= k;
            }
            assert_eq!(
                solutions.len(),
                factorial,
                "n={} produced {} solutions, expected {} (all permutations)",
                n,
                solutions.len(),
                factorial
            );
            let mut seen = std::collections::HashSet::new();
            for s in &solutions {
                assert!(
                    seen.insert(s.clone()),
                    "duplicate solution {:?} for n={}",
                    s,
                    n
                );
            }
        }
    }

    #[test]
    pub fn refine_reaches_fixed_point_under_width_cap() {
        for n in 4..=6 {
            for max_width in [2usize, 3, 4, 5, 6] {
                let merge_h = MergeHeuristic::LessRelaxed;
                let mut problem = Problem::default();
                let vars: Vec<_> = (0..n)
                    .map(|_| problem.add_variable((0..n as isize).collect(), None))
                    .collect();
                all_different(&mut problem, vars.clone());
                if n >= 5 {
                    sum(&mut problem, vars[0..3].to_vec(), (n as isize) - 1);
                }
                let problem = Arc::new(problem);
                let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
                let mut mdd = Mdd::new(
                    problem,
                    OrderingHeuristic::MinDomMaxLinked,
                    merge_h,
                    SelectHeuristic::Greedy,
                    &constraints,
                );
                mdd.refine(max_width);
                for layer in 1..mdd.number_layers() - 1 {
                    let width = mdd.number_nodes_in_layer(layer);
                    let has_relaxed_room = (0..width)
                        .map(|i| NodeIndex(layer, i))
                        .any(|node| mdd[node].is_active() && mdd[node].is_relaxed());
                    assert!(
                        width == max_width || !has_relaxed_room,
                        "n={} max_width={} merge={:?} layer={} width={} still has a relaxed node left unsplit",
                        n,
                        max_width,
                        merge_h,
                        layer,
                        width
                    );
                }
            }
        }
    }

    fn permutations_of(n: usize) -> Vec<Vec<isize>> {
        fn go(n: usize, used: &mut Vec<bool>, current: &mut Vec<isize>, all: &mut Vec<Vec<isize>>) {
            if current.len() == n {
                all.push(current.clone());
                return;
            }
            for value in 0..n {
                if !used[value] {
                    used[value] = true;
                    current.push(value as isize);
                    go(n, used, current, all);
                    current.pop();
                    used[value] = false;
                }
            }
        }
        let mut used = vec![false; n];
        let mut current = vec![];
        let mut all = vec![];
        go(n, &mut used, &mut current, &mut all);
        all
    }

    #[test]
    pub fn merged_relaxed_mdd_still_contains_every_true_all_different_solution() {
        for n in 4..=6 {
            for max_width in [1usize, 2, 3] {
                let mut problem = Problem::default();
                let vars: Vec<_> = (0..n)
                    .map(|_| problem.add_variable((0..n as isize).collect(), None))
                    .collect();
                all_different(&mut problem, vars.clone());
                let problem = Arc::new(problem);
                let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
                let mut mdd = Mdd::new(
                    problem,
                    OrderingHeuristic::MinDomMaxLinked,
                    MergeHeuristic::LessRelaxed,
                    SelectHeuristic::Greedy,
                    &constraints,
                );
                mdd.refine(max_width);
                let candidate_solutions = get_all_solutions(&mdd);
                for solution in permutations_of(n) {
                    assert!(
                        is_solution(solution.clone(), &candidate_solutions),
                        "n={} max_width={} true all-different solution {:?} missing from relaxed mdd -- unsound property merge",
                        n,
                        max_width,
                        solution
                    );
                }
            }
        }
    }

    fn all_assignments(n: usize, domain_size: usize) -> Vec<Vec<isize>> {
        let mut all = vec![vec![]];
        for _ in 0..n {
            let mut next = vec![];
            for prefix in all.iter() {
                for v in 0..domain_size as isize {
                    let mut extended = prefix.clone();
                    extended.push(v);
                    next.push(extended);
                }
            }
            all = next;
        }
        all
    }

    #[test]
    pub fn state_similarity_merge_caps_width_and_stays_sound_for_gcc() {
        // Gcc's order_key is the richest of the six (one axis per bounded value), so it
        // exercises the lexicographic multi-key sort the most; forcing max_width well below the
        // natural node count means merge_layer's bucket path must actually run.
        let n = 5;
        let candidates = all_assignments(n, n);
        for max_width in [1usize, 2, 3] {
            let mut problem = Problem::default();
            let vars: Vec<_> = (0..n)
                .map(|_| problem.add_variable((0..n as isize).collect(), None))
                .collect();
            // Every value can appear at most twice among the 5 variables.
            let bounds: Vec<(isize, usize, usize)> = (0..n as isize).map(|v| (v, 0, 2)).collect();
            gcc(&mut problem, vars.clone(), bounds);
            let problem = Arc::new(problem);
            let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            let mut mdd = Mdd::new(
                Arc::clone(&problem),
                OrderingHeuristic::MinDomMaxLinked,
                MergeHeuristic::StateSimilarity,
                SelectHeuristic::Greedy,
                &constraints,
            );
            mdd.refine(max_width);

            for layer in 1..mdd.number_layers() {
                assert!(
                    mdd.number_nodes_in_layer(layer) <= max_width,
                    "layer {} has {} nodes, exceeding max_width={}",
                    layer,
                    mdd.number_nodes_in_layer(layer),
                    max_width
                );
            }

            let candidate_solutions = get_all_solutions(&mdd);
            for assignment in candidates.iter() {
                if problem.is_solution(assignment) {
                    assert!(
                        is_solution(assignment.clone(), &candidate_solutions),
                        "max_width={} true gcc solution {:?} missing from relaxed mdd -- unsound state-similarity merge",
                        max_width,
                        assignment
                    );
                }
            }
        }
    }

    #[test]
    pub fn count_solutions_matches_brute_force_when_exact() {
        let n = 5;
        let candidates = all_assignments(n, n);
        let mut problem = Problem::default();
        let vars: Vec<_> = (0..n)
            .map(|_| problem.add_variable((0..n as isize).collect(), None))
            .collect();
        let bounds: Vec<(isize, usize, usize)> = (0..n as isize).map(|v| (v, 0, 2)).collect();
        gcc(&mut problem, vars.clone(), bounds);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            Arc::clone(&problem),
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);

        let true_count = candidates
            .iter()
            .filter(|assignment| problem.is_solution(assignment))
            .count();
        let path_count = get_all_solutions(&mdd).len();
        assert_eq!(
            path_count, true_count,
            "get_all_solutions found {} but brute force found {}",
            path_count, true_count
        );
        assert_eq!(
            mdd.count_solutions(),
            BigUint::from(true_count),
            "count_solutions() should exactly match the brute-force count once compiled exactly \
             (no merging means the diagram is tree-shaped: every path is a distinct assignment)"
        );
    }

    #[test]
    pub fn count_solutions_is_zero_when_unsat() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![1], None);
        gcc(&mut problem, vars, vec![(1, 0, 1)]); // both forced to 1, but at most one allowed
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        assert!(mdd.is_unsat());
        assert_eq!(mdd.count_solutions(), BigUint::from(0u32));
    }

    #[test]
    pub fn count_solutions_upper_bounds_the_true_count_under_relaxed_width() {
        // A relaxed diagram over-approximates: it should never encode *fewer* paths than there
        // are true solutions, since every true solution must still be present (soundness).
        let n = 5;
        let candidates = all_assignments(n, n);
        for max_width in [1usize, 2, 3] {
            let mut problem = Problem::default();
            let vars: Vec<_> = (0..n)
                .map(|_| problem.add_variable((0..n as isize).collect(), None))
                .collect();
            let bounds: Vec<(isize, usize, usize)> = (0..n as isize).map(|v| (v, 0, 2)).collect();
            gcc(&mut problem, vars.clone(), bounds);
            let problem = Arc::new(problem);
            let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            let mut mdd = Mdd::new(
                Arc::clone(&problem),
                OrderingHeuristic::MinDomMaxLinked,
                MergeHeuristic::StateSimilarity,
                SelectHeuristic::Greedy,
                &constraints,
            );
            mdd.refine(max_width);
            let true_count = candidates
                .iter()
                .filter(|assignment| problem.is_solution(assignment))
                .count();
            assert!(
                mdd.count_solutions() >= BigUint::from(true_count),
                "max_width={} count_solutions()={} is less than the true count {} -- unsound relaxation",
                max_width,
                mdd.count_solutions(),
                true_count
            );
        }
    }
}
