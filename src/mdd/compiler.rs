//! The compiler of decision diagrams: every algorithm that builds one and changes it.
//!
//! # How it works.
//!
//! A [`Compiler`] starts from the fully relaxed universe MDD (one node per layer) and tighten it with
//! [`Compiler::refine`] up to a given maximum width.
//! To tighten it, we leverage constraint propagation on the MDD. Each constraint keeps a summary
//! (a `ConstraintProperty`) of the paths that reach a node from the root (top-down properties) and
//! the paths that leave that node to the sink (bottom-up properties).
//! An edge leaving a node can be filtered when the node's properties prove that no feasible path
//! use that edge. For a complete description of the constraints' propagator, see the
//! [`constraint`](crate::constraints) module.
//!
//! The compilation starts from a fully relaxed mdd, with one node per layer, and apply constraint
//! propagation. Then, during refinement, a node is selected for splitting (i.e., it is duplicated
//! and its incoming edges are splited among the copies). Modifying the MDD's topology this way
//! results in modified properties; and constraint propagation can be reapplied once the properties
//! have been updated.
//! If after this split operation the number of nodes in a layer exceed the maximum width, the
//! merge heuristic is applied.
//! This split-and-refine procedure is applied until a fixed point is reached, or the MDD is proven
//! to be unsat.
//!
//! The compiler owns the MDD during compilation. It can be moved out the compiler when compilation
//! is done, and the compiler is then dropped and its memory freed.
//!
//! ## Properties and filtering
//!
//! A `ConstraintProperty` is a small summary that a constraint keeps of a set of paths. The
//! top-down property of a node summarises the paths from the root to it, the bottom-up property
//! the paths from it to the sink. Each is built by folding over the edges around the node:
//! starting from the constraint's identity property, `update` absorbs a parent's top-down
//! property through the edge's value, and `update_backward` a child's bottom-up property. The root
//! starts from the constraint's `empty_property`, the sink from `empty_property_backward`. An
//! edge leaving a node `u` into a node `v` is removed as soon as the constraint's
//! `is_assignment_invalid` says, from the top-down property of `u` and the bottom-up property of
//! `v`, that no solution goes through it. A layer outside the scope of a constraint leaves its
//! properties unchanged.
//!
//! Merging two nodes joins their properties (`merge`): the result describes at least the paths of
//! both, so it accepts more, never less. This is what keeps the diagram a relaxation: a solution
//! is a path that no sound constraint ever removes, and merging only adds paths.
//!
//! ## Refinement
//!
//! [`Compiler::refine`] splits nodes whose parents bring different properties (see "Exact and
//! relaxed nodes" below) until none is left or the layers are full. For a fixed `max_width` it is
//! a fix-point loop:
//!
//! ```text
//! repeat
//!     for each layer 1 .. last - 1 that has fewer than max_width nodes
//!         pick a splittable node of the layer with the select heuristic; skip the layer if none
//!         split it                          (one exact node per incoming edge)
//!         propagate the change              (update_top_down, then update_bottom_up)
//!         collapse                          (exact merge of identical nodes)
//!         merge_layer                       (lossy merge down to max_width nodes)
//!         repair_after_merges               (the nodes around a merge)
//!         clean
//!         update the relaxed flags of the layer
//! until a whole sweep splits nothing, or the diagram repeats an earlier one
//! settle                                    (recompute all properties, tighten)
//! ```
//!
//! With `max_width = usize::MAX` nothing is ever merged lossily and the loop runs until no
//! splittable node is left: the diagram is as exact as the properties allow. With a smaller width the
//! loop stops when every layer is as full as allowed and the nodes left cannot be split without
//! going over it. Splitting then merging can rebuild a diagram that was seen before, which would
//! loop forever; `fingerprint` is a hash of the nodes of each layer (their states and where their
//! edges lead), and the loop stops when one comes back. Then `settle` recomputes every property
//! from scratch, removes the edges they invalidate and collapses the nodes that became alike; if
//! that changed the diagram and no lossy merge was made in this call, the loop starts again, since
//! finer properties can allow more splits. Refining
//! again with a larger width continues from the diagram left by the previous call, so
//! refining with growing widths and ending with `usize::MAX` gives an exact diagram.
//!
//! ## Exact and relaxed nodes
//!
//! A node is *exact* if every path that reaches it leads to the same state of the constraints
//! and its parents are exact : a node below a relaxed one is reached by paths that the
//! relaxed node mixes. A node that is not exact is *relaxed*, and the diagram is exact if no node
//! is. The state of a node is the top-down property of each constraint, and "the paths lead to the
//! same state" is `ConstraintProperty::is_single_state` for each of them.
//! `update_exactness` computes the flag of a layer from the flags of the layer above and from the
//! properties, once the propagation of the layer is done; the root is exact.
//!
//! The node to split is not chosen by this flag. Splitting a node gives each parent edge its own
//! copy, which helps only if the properties that the parents bring differ; a node with a relaxed
//! parent and a single parent edge stays relaxed whatever is done to it. `is_splittable` is true
//! for the nodes whose parents do not all bring the same property.
//!
//! ## Incremental propagation
//!
//! After a split only the neighbourhood of the new nodes can change, so the properties are not
//! recomputed everywhere. The *property flag* of a node says that its properties are out of date.
//! `update_top_down` sweeps the layers downwards from the split layer: in each layer it recomputes
//! the top-down property of the flagged nodes by folding over their parents. If a node did not
//! change it is left alone, and its children are not flagged. If it did, it filters its outgoing
//! edges with the new property (an edge removed may remove nodes in turn, see `remove_node`),
//! flags its children for the next layer and flags itself for the upward pass. The sweep stops at
//! the first layer where nothing changed and returns the deepest layer that did.
//! `update_bottom_up` then sweeps upwards from that layer in the same way: it recomputes the
//! bottom-up property of the flagged nodes from their children, filters their outgoing edges, and
//! flags the parents of a node that changed or was removed. It also stops at the first layer where
//! nothing changed and no node of the next layer is flagged. A property left out of date by this
//! is coarser than the paths it stands for (an edge removed above a node, for instance, is not
//! followed by a recomputation below it).`settle` recomputes everything at the end of `refine`
//! because property finer than its paths would lose solutions: see `repair_after_merges`.
//!
//! ## Merging
//!
//! `collapse` merges, in each layer, the nodes whose top-down and bottom-up properties are all
//! equal (their `MergeKey`). They are interchangeable for every constraint, so the merge loses
//! nothing. `merge_layer` is the lossy merge: if the layer has more than `max_width` nodes it
//! asks the merge heuristic for `max_width` groups of nodes and merges each group into its first
//! node. Whether the result is relaxed depends on the states it joins, see below. A merge joins
//! the properties of the nodes it merges, but the nodes below them now hold more paths than their
//! properties say, and the nodes above have a child that holds more: `repair_after_merges`
//! recomputes them, downwards then upwards, before anything is filtered against them.
//!
//! ## Cleaning
//!
//! Nothing is ever deleted while the diagram changes: nodes and edges are deactivated, so that the
//! indices held by the propagation and by the worklists stay valid. Every split adds nodes and
//! edges and every merge or filter leaves dead ones behind, so without care the vectors keep
//! growing. `clean` drops the inactive ones: in each layer between the root and the sink it moves
//! the active nodes to the front, keeping their order, and truncates the vector, doing the same for
//! the properties and the edges, then rewrites every `NodeIndex` and `EdgeIndex` stored in
//! nodes and edges with the new positions.
//!
//! ## Unsatisfiability
//!
//! If propagation removes the root or the sink, the constraints have no solution. `Compiler::unsat`
//! then clears every structure, so that an unsatisfiable diagram has no layer at all and
//! `is_unsat` is `number_layers() == 0`.
//!
//! # Design notes
//!
//! - **The constraints are copies.** Each constraint keeps state that depends on the branching
//!   order of the diagram it is compiled into (which layers are in its scope), so [`Compiler::new`]
//!   clones the constraints of the problem and gives the clone the order.
//! - **The root and the sink are not stored.** They are the first node of the first layer and of
//!   the last layer; [`Mdd::debug_assert_root_and_sink`] checks that those layers have a single node.
//! - **The heuristics only choose.** They name the node to split and the groups to merge, and
//!   never change the diagram themselves, so no choice can make it unsound.
//! - **The branching order is repaired, not trusted.** Heuristics know nothing of constraints that
//!   need a specific sequence of variables (`Regular`); `Compiler::new` reorders for them and panics
//!   if two constraints ask for contradictory sequences.
//! - **A diagram is only as exact as its constraints' filtering.** Each constraint documents
//!   what its properties can and cannot detect.
//! - The MDD is separate from the compiler for at least two reasons. First, when compilation
//!   finishes and the MDD must be used for other tasks (e.g., learning module) then all
//!   compilation-related data are not needed anymore (e.g., constraints' properties are not
//!   useful). Then, the MDD structure can be extracted and used as-is. Moreover, a compiler
//!   structure is useful when multiple turn of refine are called (e.g., with increased maximum
//!   width).

use super::heuristics::*;
use super::*;
use crate::modelling::*;

use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The compilation of some constraints of a problem into a decision diagram, and the state it
/// needs to go on.
pub struct Compiler {
    /// The diagram being built.
    mdd: Mdd,
    /// Constraints compiled in this diagram, in the order given to [`Compiler::new`]. They are
    /// copies of the problem's constraints: each one keeps the state that depends on this
    /// diagram's variable order (which layers it is in the scope of).
    constraints: Vec<Box<dyn Constraint>>,
    /// Heuristic used to score nodes during merging operation
    merge_heuristic: MergeHeuristic,
    /// Heuristic to select nodes to split
    select_heuristic: SelectHeuristic,
    /// Top down properties of the constraints: `[layer][position][constraint]`
    top_down_properties: Vec<Vec<Vec<Box<dyn ConstraintProperty>>>>,
    /// Bottom up properties of the constraints: `[layer][position][constraint]`
    bottom_up_properties: Vec<Vec<Vec<Box<dyn ConstraintProperty>>>>,
}

/// Repairs `var_order` (as produced by an `OrderingHeuristic`) so it satisfies every
/// `precedence_edges` pair from `constraints` -- e.g. `Regular`'s requirement that its scope be
/// branched in the exact sequence its automaton walks it.
///
/// Implemented as a "stable" topological sort (Kahn's algorithm, breaking ties by the heuristic's
/// own rank): among the variables with no unsatisfied precedence-predecessor remaining, always
/// picks the one the heuristic placed earliest. This satisfies every precedence pair while
/// otherwise disturbing the heuristic's chosen order as little as possible.
///
/// Panics if the combined precedence edges are contradictory (e.g. two `Regular` constraints over
/// the same scope declared in incompatible sequences).
/// That's a genuine modelling conflict between the constraints being compiled into this MDD, not
/// something any variable ordering can resolve.
pub(crate) fn enforce_precedence_order(
    var_order: Vec<VariableIndex>,
    constraints: &[Box<dyn Constraint>],
) -> Vec<VariableIndex> {
    let edges: HashSet<(VariableIndex, VariableIndex)> = constraints
        .iter()
        .flat_map(|c| c.precedence_edges())
        .collect();
    if edges.is_empty() {
        return var_order;
    }

    let rank: FxHashMap<VariableIndex, usize> =
        var_order.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let mut successors: FxHashMap<VariableIndex, Vec<VariableIndex>> = FxHashMap::default();
    let mut in_degree: FxHashMap<VariableIndex, usize> =
        var_order.iter().map(|&v| (v, 0)).collect();
    for (before, after) in edges {
        successors.entry(before).or_default().push(after);
        *in_degree.get_mut(&after).expect(
            "precedence_edges must only reference variables in this constraint's (and hence \
             the MDD's) own scope",
        ) += 1;
    }

    let mut heap: BinaryHeap<Reverse<(usize, VariableIndex)>> = var_order
        .iter()
        .filter(|&&v| in_degree[&v] == 0)
        .map(|&v| Reverse((rank[&v], v)))
        .collect();

    let mut result = Vec::with_capacity(var_order.len());
    while let Some(Reverse((_, v))) = heap.pop() {
        result.push(v);
        for &succ in successors.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
            let deg = in_degree.get_mut(&succ).unwrap();
            *deg -= 1;
            if *deg == 0 {
                heap.push(Reverse((rank[&succ], succ)));
            }
        }
    }

    assert_eq!(
        result.len(),
        var_order.len(),
        "conflicting precedence requirements between constraints being compiled into the same \
         MDD -- no variable ordering can satisfy every order-sensitive constraint's declared \
         sequence simultaneously"
    );
    result
}

impl Compiler {
    /// Creates the diagram of `constraints` (indices into `problem`) and propagates them once.
    pub fn new(
        problem: Arc<Problem>,
        heuristic: CompilationHeuristic,
        constraints: &[ConstraintIndex],
    ) -> Self {
        let CompilationHeuristic {
            ordering,
            merge: merge_heuristic,
            selection: select_heuristic,
        } = heuristic;
        merge_heuristic.check(&problem);
        let mut in_scope = vec![false; problem.number_variables()];
        // Scope is the union of the constraints' scope.
        let mut scope = vec![];
        for constraint in constraints.iter().copied() {
            for variable in problem[constraint].iter_scope() {
                if !in_scope[*variable] {
                    scope.push(variable);
                    in_scope[*variable] = true;
                }
            }
        }
        let number_layers = scope.len();
        let constraints = constraints
            .iter()
            .map(|&constraint| problem[constraint].clone() as Box<dyn Constraint>)
            .collect::<Vec<Box<dyn Constraint>>>();
        let mut compiler = Self {
            mdd: Mdd::with_layers(problem, number_layers),
            constraints,
            merge_heuristic,
            select_heuristic,
            top_down_properties: vec![vec![]; number_layers + 1],
            bottom_up_properties: vec![vec![]; number_layers + 1],
        };

        // First, we create each layer. There is n + 1 layers, with n the number of variables. The
        // last layer is the sink node. Each layer has one node at creation.
        for layer in 0..compiler.mdd.number_layers() {
            compiler.add_node(layer);
        }

        // Set the variable order in the MDD given the heuristics
        // We get for each layer its decision variable
        let var_order = ordering.get_order(&compiler.mdd.problem, &scope);
        // Repair the heuristic's order to take into account order-sensitive constraints (e.g.,
        // Regular)
        let var_order = enforce_precedence_order(var_order, &compiler.constraints);
        for i in 0..compiler.constraints.len() {
            compiler.constraints[i].update_variable_ordering(&var_order);
        }
        compiler.mdd.order = var_order;

        // Next, we add the edges between the layers. There is edges only from one layer to the
        // next.
        for layer in 0..compiler.mdd.number_layers() - 1 {
            let source = NodeIndex(layer, 0);
            let target = NodeIndex(layer + 1, 0);
            let variable = compiler.mdd.order[layer];
            for value in (0..compiler.mdd.problem[variable].domain_size()).map(ValueIndex) {
                compiler.add_edge(layer, source, target, value);
            }
        }
        compiler.propagate_constraints();
        if !compiler.mdd[compiler.mdd.root()].is_active()
            || !compiler.mdd[compiler.mdd.sink()].is_active()
        {
            compiler.unsat();
            return compiler;
        }
        compiler.clean();
        for layer in 0..compiler.mdd.number_layers() {
            compiler.update_exactness(layer);
        }
        compiler.mdd.debug_assert_root_and_sink();
        compiler
    }

    /// The diagram as it is now. It is a relaxation of the constraints' solutions, tighter after
    /// each [`Compiler::refine`].
    pub fn mdd(&self) -> &Mdd {
        &self.mdd
    }

    /// Gives the diagram away and drops everything that only the compilation needed: the
    /// constraints, their properties and the heuristics. The diagram cannot be refined any more.
    pub fn into_mdd(self) -> Mdd {
        self.mdd
    }

    /// Turns the diagram into the unsatisfiable one: nothing is kept, no layer, no node, no edge,
    /// no order and no property, so that [`Mdd::is_unsat`] holds. This is the only place that
    /// decides what an unsatisfiable diagram looks like; call it as soon as the constraints are
    /// found to have no solution.
    fn unsat(&mut self) {
        self.mdd.clear();
        self.top_down_properties.clear();
        self.bottom_up_properties.clear();
    }

    /// Adds a node to `layer` and returns its index. Its properties start at the identity of the
    /// constraints, except those of the root (top-down) and of the sink (bottom-up), which start
    /// at the empty property: the empty path.
    fn add_node(&mut self, layer: usize) -> NodeIndex {
        let index_in_layer = self.mdd.nodes[layer].len();
        let node = Node::new(layer, index_in_layer, false);
        let index = NodeIndex(layer, index_in_layer);
        self.mdd.nodes[layer].push(node);

        let is_root = layer == 0;
        let is_sink = layer == self.mdd.number_layers() - 1;

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

    /// Adds an edge from `from` (in `layer`) to `to` (in the next layer) labelled with the value
    /// `assignment` of the variable of `layer`, and registers it on both endpoints.
    fn add_edge(&mut self, layer: usize, from: NodeIndex, to: NodeIndex, assignment: ValueIndex) {
        let edge_index = EdgeIndex(layer, self.mdd.edges[layer].len());
        self.mdd[from].add_child_edge(edge_index);
        self.mdd[to].add_parent_edge(edge_index);
        let edge = Edge::new(from, to, assignment);
        self.mdd.edges[layer].push(edge);
    }

    /// Number of constraints compiled in this diagram.
    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// The constraints compiled in this diagram.
    pub fn iter_constraints(&self) -> impl Iterator<Item = &Box<dyn Constraint>> {
        self.constraints.iter()
    }

    // --- split and refine strategy ---- //

    /// Tightens the diagram by splitting the nodes that mix paths of different states, with at
    /// most `max_width` nodes per layer. The relaxed flags of the diagram are up to date when it
    /// returns.
    pub fn refine(&mut self, max_width: usize) {
        if self.mdd.is_unsat() {
            return;
        }
        let select_heuristic = self.select_heuristic;
        let mut merged_lossily = false;
        let mut seen_fingerprints: HashSet<u64> = HashSet::default();
        seen_fingerprints.insert(self.fingerprint());
        loop {
            let mut any_split = false;
            for layer in 1..self.mdd.nodes.len() - 1 {
                let selected = if self.mdd.number_nodes_in_layer(layer) == max_width {
                    None
                } else {
                    select_heuristic.select_node(self, layer)
                };
                if let Some(node) = selected {
                    any_split = true;
                    let new_nodes = self.split_node(node);
                    for &new_node in &new_nodes {
                        self.mdd[new_node].set_property_flag();
                    }
                    let deepest_touched = self.update_top_down(layer);
                    for &new_node in &new_nodes {
                        self.mdd[new_node].set_property_flag();
                    }
                    self.update_bottom_up(deepest_touched);
                    if !self.mdd[self.mdd.root()].is_active()
                        || !self.mdd[self.mdd.sink()].is_active()
                    {
                        self.unsat();
                        return;
                    }
                    // Merge equivalent nodes
                    let merged = self.collapse(layer);
                    self.repair_after_merges(&merged);
                    // Reduce the size to the max_width, if needed
                    let merged = self.merge_layer(layer, max_width);
                    merged_lossily |= !merged.is_empty();
                    self.repair_after_merges(&merged);
                    if !self.mdd[self.mdd.root()].is_active()
                        || !self.mdd[self.mdd.sink()].is_active()
                    {
                        self.unsat();
                        return;
                    }
                    // Reclame space used by inactive data
                    self.clean();
                }
                // The next layers read the flags of this one (and so does the merge heuristic).
                self.update_exactness(layer);
            }
            self.update_exactness(self.mdd.nodes.len() - 1);
            let cycled = any_split && !seen_fingerprints.insert(self.fingerprint());
            if !any_split || cycled {
                // Nothing more to split, or going in circles: tighten what the updates left
                // coarse. Finer properties can allow more splits, so go on if that changed the
                // diagram. Not after a lossy merge, which keeps the diagram moving without
                // settling, nor if it was going in circles or the result was seen before.
                let changed = self.settle();
                if self.mdd.nodes.is_empty() {
                    return;
                }
                if cycled
                    || !changed
                    || merged_lossily
                    || !seen_fingerprints.insert(self.fingerprint())
                {
                    break;
                }
            }
        }
    }

    /// Recompute every properties based on the actual paths encoded in the MDD and tightens it.
    /// Much of the updates are doing the refine, but some operations don't lead to incremental
    /// property update (e.g., edge removal) because then it would be too costly and the fix-point
    /// handle many of these cases by iterating over the diagram anyway.
    /// Hence, some properties may be more relaxed than possible, leading to sub-optimal
    /// propagation and the MDD could still be tighten.
    /// Returns whethe the diagram changed. If the MDD becomes UNSAT, clears it.
    fn settle(&mut self) -> bool {
        // Computes the number of nodes and edges in the MDD.
        let size = |compiler: &Self| -> (usize, usize) {
            (
                compiler
                    .mdd
                    .nodes
                    .iter()
                    .flatten()
                    .filter(|node| node.is_active())
                    .count(),
                compiler
                    .mdd
                    .edges
                    .iter()
                    .flatten()
                    .filter(|edge| edge.is_active())
                    .count(),
            )
        };
        let before = size(self);
        loop {
            let edges = size(self).1;
            self.propagate_constraints();
            if !self.mdd[self.mdd.root()].is_active() || !self.mdd[self.mdd.sink()].is_active() {
                self.unsat();
                return true;
            }
            if size(self).1 == edges {
                break;
            }
        }
        self.clean();
        for layer in 1..self.mdd.nodes.len() - 1 {
            let merged = self.collapse(layer);
            self.repair_after_merges(&merged);
        }
        if !self.mdd[self.mdd.root()].is_active() || !self.mdd[self.mdd.sink()].is_active() {
            self.unsat();
            return true;
        }
        self.clean();
        for layer in 0..self.mdd.nodes.len() {
            self.update_exactness(layer);
        }
        size(self) != before
    }

    /// A hash of the diagram: for each layer, the hashes of the `MergeKey` of its active nodes,
    /// with where its active child edges lead, combined without regard to their order; the layers
    /// are then combined in order. `refine` uses it to notice that it is going in circles, so a
    /// sweep that only moves an edge must change it: that is progress.
    fn fingerprint(&self) -> u64 {
        let mut total: u64 = 0;
        for layer in 0..self.mdd.nodes.len() {
            let mut layer_signature: u64 = 0;
            for index in 0..self.mdd.nodes[layer].len() {
                if !self.mdd.nodes[layer][index].is_active() {
                    continue;
                }
                let key = MergeKey {
                    td_properties: &self.top_down_properties[layer][index],
                    bu_properties: &self.bottom_up_properties[layer][index],
                };
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                index.hash(&mut hasher);
                for edge in self.mdd.nodes[layer][index].iter_children() {
                    if self.mdd[edge].is_active() {
                        self.mdd[edge].assignment().hash(&mut hasher);
                        self.mdd[edge].to().hash(&mut hasher);
                    }
                }
                layer_signature ^= hasher.finish();
            }
            total = total
                .wrapping_mul(1_000_000_007)
                .wrapping_add(layer_signature);
        }
        total
    }

    /// Sets the relaxed flag of the nodes of `layer` from their top-down properties and the flags
    /// of their parents, which must be up to date.
    fn update_exactness(&mut self, layer: usize) {
        for index in 0..self.mdd.nodes[layer].len() {
            let node = NodeIndex(layer, index);
            if !self.mdd[node].is_active() {
                continue;
            }
            // Nothing is decided after the sink, so the states of its paths do not matter.
            let is_sink = layer == self.mdd.nodes.len() - 1;
            let relaxed = layer != 0
                && (self.mdd[node]
                    .iter_parents()
                    .any(|edge| self.mdd[self.mdd[edge].from()].is_relaxed())
                    || (!is_sink
                        && !self.top_down_properties[layer][index]
                            .iter()
                            .all(|property| property.is_single_state())));
            self.mdd[node].set_relaxed(relaxed);
        }
    }

    /// True if splitting `node` would separate paths that its properties mix: it has at least two
    /// parent edges and, for some constraint, they do not all bring the same top-down property.
    /// The heuristic that selects the node to split only chooses among these. A node whose
    /// parents agree, or that has a single parent, gains nothing from a split: its copies would
    /// have the same properties, and `collapse` would merge them back.
    pub(crate) fn is_splittable(&self, node: NodeIndex) -> bool {
        if self.mdd[node].number_parents() < 2 {
            return false;
        }
        (0..self.constraints.len()).any(|constraint_index| {
            let mut contributions = self.mdd[node]
                .iter_parents()
                .map(|edge| self.property_through_edge(edge, constraint_index));
            let first = contributions
                .next()
                .expect("a node with two parents has a first one");
            contributions.any(|other| !ConstraintProperty::eq(first.as_ref(), other.as_ref()))
        })
    }

    /// The top-down property that the parent edge `edge` brings to its child for the constraint
    /// `constraint_index`: the identity property updated with the parent's property and the
    /// edge's value. Folding the properties of all the parent edges of a node gives its property.
    fn property_through_edge(
        &self,
        edge: EdgeIndex,
        constraint_index: usize,
    ) -> Box<dyn ConstraintProperty> {
        let NodeIndex(source_layer, source_index) = self.mdd[edge].from();
        let variable = self.mdd.order[source_layer];
        let assignment = self.mdd.problem[variable].value(self.mdd[edge].assignment());
        let in_scope = self.constraints[constraint_index].is_layer_in_scope(source_layer);
        let mut property = self.constraints[constraint_index].identity_property();
        property.update(
            self.top_down_properties[source_layer][source_index][constraint_index].as_ref(),
            assignment,
            in_scope,
        );
        property
    }

    /// Splits `node`, which has `n` parents, into `n` nodes, one per parent edge, and
    /// returns them (the original first). The original keeps its first parent edge, and each other
    /// parent edge is deactivated and replaced by a new edge, with the same value, into a new node
    /// that has a copy of every active child edge of the original. The properties of the new nodes
    /// are not computed here: they start at the identity, and the caller must flag them and
    /// propagate.
    fn split_node(&mut self, node: NodeIndex) -> Vec<NodeIndex> {
        let mut nodes = vec![node];
        let NodeIndex(layer, _) = node;
        let n = self.mdd[node].number_parents();
        let outgoing_assignments = self.mdd[node]
            .iter_children()
            .filter(|edge| self.mdd[*edge].is_active())
            .map(|edge| (self.mdd[edge].to(), self.mdd[edge].assignment()))
            .collect::<Vec<(NodeIndex, ValueIndex)>>();
        for i in (1..n).rev() {
            let new_node = self.add_node(layer);
            nodes.push(new_node);
            let edge = self.mdd[node].parent_edge_at(i);
            let from = self.mdd[edge].from();
            let assignment = self.mdd[edge].assignment();
            self.add_edge(layer - 1, from, new_node, assignment);
            for (child, outgoing_assignment) in outgoing_assignments.iter().copied() {
                self.add_edge(layer, new_node, child, outgoing_assignment);
            }
            self.mdd[edge].deactivate();
            self.mdd[node].swap_remove_parent_edge(i);
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
    pub(crate) fn propagate_constraints(&mut self) {
        self.update_top_down_properties();
        self.update_bottom_up_properties_and_filter_edges();
    }

    /// Recomputes the top-down properties of the flagged nodes, layer by layer from `start_layer`
    /// (but at least the first layer after the root), and filters the edges that leave the nodes
    /// whose property changed. Flags the children of a changed node for the next layer, and the
    /// node itself for [`Mdd::update_bottom_up`]. Stops at the first layer where no node changed and none is flagged in the next one.
    /// Returns the deepest layer where one did. See the module documentation.
    fn update_top_down(&mut self, start_layer: usize) -> usize {
        let mut deepest_changed_layer = start_layer.max(1) - 1;
        for layer in start_layer.max(1)..self.mdd.nodes.len() {
            let parent_variable = self.mdd.order[layer - 1];
            let mut any_changed = false;
            for index in 0..self.mdd.nodes[layer].len() {
                let target = NodeIndex(layer, index);
                if !self.mdd[target].is_property_flag() {
                    continue;
                }
                self.mdd[target].unset_property_flag();
                if !self.mdd[target].is_active() {
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
                if layer < self.mdd.nodes.len() - 1 {
                    let children_before: Vec<NodeIndex> = self.mdd[target]
                        .iter_children()
                        .map(|edge| self.mdd[edge].to())
                        .collect();
                    let variable = self.mdd.order[layer];
                    for constraint_index in 0..self.constraints.len() {
                        if self.constraints[constraint_index].is_layer_in_scope(layer) {
                            self.filter_invalid_edges(target, variable, constraint_index);
                        }
                    }
                    for child in children_before {
                        if self.mdd[child].is_active() {
                            self.mdd[child].set_property_flag();
                        }
                    }
                }
                self.mdd[target].set_property_flag();
            }
            if !any_changed && !self.has_flagged_node(layer + 1) {
                break;
            }
        }
        deepest_changed_layer
    }

    /// Recomputes the bottom-up properties of the flagged nodes, layer by layer upwards from
    /// `start_layer` (at most the layer before the sink), and filters the edges that leave them.
    /// Flags the parents of a node whose property changed, or that was removed. Stops at the first
    /// layer where no node changed and none is flagged in the next one. See the module
    /// documentation.
    fn update_bottom_up(&mut self, start_layer: usize) {
        for layer in (0..=start_layer.min(self.mdd.nodes.len() - 2)).rev() {
            let variable = self.mdd.order[layer];
            let mut any_changed = false;
            for index in 0..self.mdd.nodes[layer].len() {
                let target = NodeIndex(layer, index);
                if !self.mdd[target].is_property_flag() {
                    continue;
                }
                self.mdd[target].unset_property_flag();
                if !self.mdd[target].is_active() {
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
                    self.mdd[target]
                        .iter_parents()
                        .map(|edge| self.mdd[edge].from())
                        .collect()
                } else {
                    Vec::new()
                };
                for constraint_index in 0..self.constraints.len() {
                    if self.constraints[constraint_index].is_layer_in_scope(layer) {
                        self.filter_invalid_edges(target, variable, constraint_index);
                    }
                }
                let target_removed = !self.mdd[target].is_active();
                if !changed && !target_removed {
                    continue;
                }
                any_changed = true;
                {
                    for parent in parents_before {
                        if self.mdd[parent].is_active() {
                            self.mdd[parent].set_property_flag();
                        }
                    }
                }
            }
            if !any_changed && (layer == 0 || !self.has_flagged_node(layer - 1)) {
                break;
            }
        }
    }

    /// Whether a node of `layer` is flagged for a property update. False beyond the last layer.
    fn has_flagged_node(&self, layer: usize) -> bool {
        self.mdd
            .nodes
            .get(layer)
            .is_some_and(|nodes| nodes.iter().any(|node| node.is_property_flag()))
    }

    /// Recomputes `top_down_properties` for every layer but the root (layer 0), whose top-down
    /// property is permanently `empty_property()` - see `add_node`.
    fn update_top_down_properties(&mut self) {
        for layer in 1..self.mdd.nodes.len() {
            let variable = self.mdd.order[layer - 1];
            for i in 0..self.mdd.nodes[layer].len() {
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
        for j in 0..self.mdd[target].number_parents() {
            let edge = self.mdd[target].parent_edge_at(j);
            let NodeIndex(source_layer, source_index) = self.mdd[edge].from();
            let in_scope = self.constraints[constraint_index].is_layer_in_scope(source_layer);
            let assignment = self.mdd.problem[variable].value(self.mdd[edge].assignment());
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
        for layer in (0..self.mdd.nodes.len() - 1).rev() {
            let variable = self.mdd.order[layer];
            for node_index in 0..self.mdd.nodes[layer].len() {
                let target = NodeIndex(layer, node_index);
                if !self.mdd[target].is_active() {
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
        for edge_index in 0..self.mdd[target].number_children() {
            let edge = self.mdd[target].child_edge_at(edge_index);
            let NodeIndex(child_layer, child_index) = self.mdd[edge].to();
            let assignment = self.mdd.problem[variable].value(self.mdd[edge].assignment());
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
        for edge_index in (0..self.mdd[target].number_children()).rev() {
            let edge = self.mdd[target].child_edge_at(edge_index);
            let child = self.mdd[edge].to();
            let NodeIndex(child_layer, child_index) = child;
            let assignment = self.mdd.problem[variable].value(self.mdd[edge].assignment());
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
            self.mdd[target].swap_remove_child_edge(edge_index);
            if self.mdd[target].number_children() == 0 {
                self.remove_node(target);
            }
            self.mdd[child].remove_parent_edge(edge);
            if self.mdd[child].number_parents() == 0 {
                self.remove_node(child);
            }
            self.mdd[edge].deactivate();
        }
    }

    /// Deactivates `node` and all its edges, then removes in turn every neighbour left without a
    /// parent or without a child, since such a node is on no root-to-sink path. Does nothing if
    /// the node is already inactive.
    fn remove_node(&mut self, node: NodeIndex) {
        if !self.mdd[node].is_active() {
            return;
        }
        self.mdd[node].deactivate();
        for i in 0..self.mdd[node].number_parents() {
            let edge = self.mdd[node].parent_edge_at(i);
            self.mdd[edge].deactivate();
            let parent = self.mdd[edge].from();
            self.mdd[parent].remove_child_edge(edge);
            if self.mdd[parent].number_children() == 0 {
                self.remove_node(parent);
            }
        }
        for i in 0..self.mdd[node].number_children() {
            let edge = self.mdd[node].child_edge_at(i);
            self.mdd[edge].deactivate();
            let child = self.mdd[edge].to();
            self.mdd[child].remove_parent_edge(edge);
            if self.mdd[child].number_parents() == 0 {
                self.remove_node(child);
            }
        }
    }

    /// Merges, in `layer`, the nodes whose top-down and bottom-up properties are all equal. The
    /// merge loses nothing: the nodes are interchangeable for every constraint. Returns the nodes
    /// that took over others, see [`Compiler::merge_nodes`].
    fn collapse(&mut self, layer: usize) -> Vec<NodeIndex> {
        let mut map: FxHashMap<MergeKey, NodeIndex> = FxHashMap::default();
        let mut merges: Vec<(NodeIndex, NodeIndex)> = vec![];
        for index in 0..self.mdd.nodes[layer].len() {
            let node = NodeIndex(layer, index);
            if !self.mdd[node].is_active() {
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
        let mut merged = vec![];
        for (node, primary_node) in merges {
            merged.extend(self.merge_nodes(node, primary_node));
        }
        merged
    }

    /// Merges the nodes of `layer` that the merge heuristic picks until at most `max_width` remain.
    /// The heuristic is only consulted when the layer has more active nodes than that. Returns the
    /// nodes that took over others, see [`Compiler::merge_nodes`].
    fn merge_layer(&mut self, layer: usize, max_width: usize) -> Vec<NodeIndex> {
        let active_nodes = self
            .mdd
            .nodes_in_layer(layer)
            .filter(|&node| self.mdd[node].is_active())
            .count();
        if active_nodes <= max_width {
            return vec![];
        }
        let groups = self.merge_heuristic.group_nodes(self, layer, max_width);
        let mut merged = vec![];
        for group in groups {
            if let Some((&into, froms)) = group.split_first() {
                for &from in froms {
                    merged.extend(self.merge_nodes(from, into));
                }
            }
        }
        merged
    }

    /// Merges `from` into `into`: the parent edges of `from` now enter `into`, its child edges are
    /// moved, dropped or, where `into` has an edge of the same value towards another child, cause
    /// those two children to be merged in turn. The properties are joined and `from` is
    /// deactivated. Whether the result is relaxed is not decided here: see
    /// [`Compiler::update_exactness`]. See the module documentation.
    ///
    /// Returns the nodes that took over others (`into`, and the children merged in turn), which
    /// may have been merged away since: the properties of what lies above and below them are
    /// to be repaired by [`Compiler::repair_after_merges`].
    fn merge_nodes(&mut self, from: NodeIndex, into: NodeIndex) -> Vec<NodeIndex> {
        let mut worklist = vec![(from, into)];
        let mut took_over = vec![];
        // Nodes merged away so far, with the node that took them over. An entry of the worklist
        // can name a node that an earlier entry has already absorbed: it is followed to the node
        // that holds its paths now, since merging into a deactivated node would lose them.
        let mut absorbed = FxHashMap::<NodeIndex, NodeIndex>::default();
        while let Some((from, into)) = worklist.pop() {
            let resolve = |mut node: NodeIndex| {
                while let Some(&next) = absorbed.get(&node) {
                    node = next;
                }
                node
            };
            let (from, into) = (resolve(from), resolve(into));
            if from == into {
                continue;
            }
            took_over.push(into);
            for i in 0..self.mdd[from].number_parents() {
                let edge = self.mdd[from].parent_edge_at(i);
                self.mdd[edge].set_to(into);
                self.mdd[into].add_parent_edge(edge);
            }

            let mut existing_children = FxHashMap::<ValueIndex, NodeIndex>::default();
            for i in 0..self.mdd[into].number_children() {
                let edge = self.mdd[into].child_edge_at(i);
                if !self.mdd[edge].is_active() {
                    continue;
                }
                existing_children.insert(self.mdd[edge].assignment(), self.mdd[edge].to());
            }

            for i in 0..self.mdd[from].number_children() {
                let edge = self.mdd[from].child_edge_at(i);
                if !self.mdd[edge].is_active() {
                    continue;
                }
                let assignment = self.mdd[edge].assignment();
                let child = self.mdd[edge].to();
                match existing_children.get(&assignment).copied() {
                    None => {
                        self.mdd[edge].set_from(into);
                        self.mdd[into].add_child_edge(edge);
                        existing_children.insert(assignment, child);
                    }
                    Some(existing_child) if existing_child == child => {
                        self.mdd[child].remove_parent_edge(edge);
                        self.mdd[edge].deactivate();
                    }
                    Some(existing_child) => {
                        self.mdd[child].remove_parent_edge(edge);
                        self.mdd[edge].deactivate();
                        worklist.push((child, existing_child));
                    }
                }
            }
            Self::merge_properties(&mut self.top_down_properties[from.0], from.1, into.1);
            Self::merge_properties(&mut self.bottom_up_properties[from.0], from.1, into.1);
            self.mdd[from].deactivate();
            absorbed.insert(from, into);
        }
        took_over
    }

    /// Recomputes properties of nodes impacted by a merge operation. When merging nodes, their
    /// properties are merge using the relaxation operators (when they are not equivalent);
    /// however, this changes the properties of the nodes above and below them in the diagram.
    /// Propagating constraint on these tighter properties may lead to unwanted edge removal and
    /// lead to loss of solutions.
    fn repair_after_merges(&mut self, took_over: &[NodeIndex]) {
        let took_over: Vec<NodeIndex> = took_over
            .iter()
            .copied()
            .filter(|&node| self.mdd[node].is_active())
            .collect();
        let (Some(first), Some(last)) = (
            took_over.iter().map(|node| node.0).min(),
            took_over.iter().map(|node| node.0).max(),
        ) else {
            return;
        };
        for &node in &took_over {
            for edge in self.mdd[node].iter_children().collect::<Vec<_>>() {
                let child = self.mdd[edge].to();
                if self.mdd[edge].is_active() && self.mdd[child].is_active() {
                    self.mdd[child].set_property_flag();
                }
            }
        }
        let deepest_changed = self.update_top_down(first + 1).max(last);
        for &node in &took_over {
            if !self.mdd[node].is_active() {
                continue;
            }
            self.mdd[node].set_property_flag();
            for edge in self.mdd[node].iter_parents().collect::<Vec<_>>() {
                let parent = self.mdd[edge].from();
                if self.mdd[edge].is_active() && self.mdd[parent].is_active() {
                    self.mdd[parent].set_property_flag();
                }
            }
        }
        self.update_bottom_up(deepest_changed);
    }

    /// Joins, in the layer `properties`, the properties of the node at `from_index` into those of
    /// the node at `into_index`.
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

    /// Drops the inactive nodes and edges and renumbers the others, keeping their order, in every
    /// vector (nodes, edges, properties), and rewrites the indices stored in the nodes and edges.
    /// The root and the sink are kept even if inactive. See the module documentation.
    fn clean(&mut self) {
        let mut map_node_index = FxHashMap::<NodeIndex, NodeIndex>::default();
        let (root, sink) = (self.mdd.root(), self.mdd.sink());
        map_node_index.insert(root, root);
        map_node_index.insert(sink, sink);
        for layer in 1..self.mdd.nodes.len() - 1 {
            let mut new_index = 0;
            for index in 0..self.mdd.nodes[layer].len() {
                if self.mdd.nodes[layer][index].is_active() {
                    map_node_index.insert(NodeIndex(layer, index), NodeIndex(layer, new_index));
                    self.mdd.nodes[layer].swap(new_index, index);
                    self.top_down_properties[layer].swap(new_index, index);
                    self.bottom_up_properties[layer].swap(new_index, index);
                    new_index += 1;
                }
            }
            self.mdd.nodes[layer].truncate(new_index);
            self.top_down_properties[layer].truncate(new_index);
            self.bottom_up_properties[layer].truncate(new_index);
        }
        let mut map_edge_index = FxHashMap::<EdgeIndex, EdgeIndex>::default();
        for layer in 0..self.mdd.edges.len() {
            let mut new_index = 0;
            for index in 0..self.mdd.edges[layer].len() {
                let from = self.mdd.edges[layer][index].from();
                let to = self.mdd.edges[layer][index].to();
                if self.mdd.edges[layer][index].is_active()
                    && map_node_index.contains_key(&from)
                    && map_node_index.contains_key(&to)
                {
                    map_edge_index.insert(EdgeIndex(layer, index), EdgeIndex(layer, new_index));
                    self.mdd.edges[layer].swap(new_index, index);
                    new_index += 1;
                }
            }
            self.mdd.edges[layer].truncate(new_index);
        }

        for layer in 0..self.mdd.nodes.len() {
            for index in 0..self.mdd.nodes[layer].len() {
                self.mdd.nodes[layer][index].update_edge_indices(&map_edge_index);
            }
            if layer > 0 {
                for index in 0..self.mdd.edges[layer - 1].len() {
                    self.mdd.edges[layer - 1][index].update_node_indices(&map_node_index);
                }
            }
        }
    }

    /// The top-down properties of `node`, one per compiled constraint.
    pub(crate) fn top_down_properties(&self, node: NodeIndex) -> &[Box<dyn ConstraintProperty>] {
        &self.top_down_properties[node.0][node.1]
    }

    /// The bottom-up properties of `node`, one per compiled constraint.
    pub(crate) fn bottom_up_properties(&self, node: NodeIndex) -> &[Box<dyn ConstraintProperty>] {
        &self.bottom_up_properties[node.0][node.1]
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
        let compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        let solutions = get_all_solutions(compiler.mdd());
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

    /// Regression test for `enforce_precedence_order`: `MinDomMaxLinked` has no notion of
    /// `Regular`'s required scope order, so without the repair pass in `Mdd::new`, a `Regular`
    /// constraint whose scope isn't the one `MinDomMaxLinked` would naturally pick first panics
    /// in `Regular::update_variable_ordering`. This inflates `vars[3]`'s constraint degree well
    /// above `vars[0..3]`'s so `MinDomMaxLinked`'s greedy score/tie-break picks `vars[3]` FIRST
    /// -- directly ahead of `vars[0]`, which the trivial automaton below requires to be branched
    /// before `vars[1]`, `vars[2]`, and `vars[3]`.
    #[test]
    pub fn min_dom_max_linked_repairs_order_for_regular_constraints() {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = (0..4)
            .map(|_| problem.add_variable(vec![0, 1], None))
            .collect();
        // One state, self-loop on every value, always accepting: imposes no restriction on
        // solutions, only exercises the ordering requirement (vars[0] < vars[1] < vars[2] <
        // vars[3]).
        regular(
            &mut problem,
            vars.clone(),
            vec![vec![Some(0), Some(0)]],
            0,
            vec![0],
        );
        for _ in 0..3 {
            let dummy = problem.add_variable(vec![0, 1], None);
            not_equals(&mut problem, dummy, vars[3]);
        }

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        // Should not panic -- see the doc above.
        let compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );

        let positions: Vec<usize> = (0..compiler.mdd().number_layers() - 1)
            .map(|layer| (compiler.mdd().decision_at_layer(layer), layer))
            .filter(|(v, _)| vars.contains(v))
            .map(|(_, layer)| layer)
            .collect();
        assert_eq!(positions.len(), vars.len());
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "Regular's declared scope order was not respected: layers {positions:?}"
        );
    }

    /// Two `Regular` constraints declaring contradictory sequences over the same 2 variables
    /// (v0 before v1, and v1 before v0) is a genuine modelling conflict no variable ordering can
    /// satisfy -- `enforce_precedence_order` should detect the cycle and panic with a clear
    /// message rather than silently picking one order and violating the other constraint.
    #[test]
    #[should_panic(expected = "conflicting precedence requirements")]
    pub fn conflicting_regular_orderings_panics_with_a_clear_message() {
        let mut problem = Problem::default();
        let v0 = problem.add_variable(vec![0, 1], None);
        let v1 = problem.add_variable(vec![0, 1], None);
        let trivial = vec![vec![Some(0), Some(0)]];
        regular(&mut problem, vec![v0, v1], trivial.clone(), 0, vec![0]);
        regular(&mut problem, vec![v1, v0], trivial, 0, vec![0]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let _ = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
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
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        compiler.refine(usize::MAX);
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
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );

        let into = NodeIndex(1, 0);
        let c1 = NodeIndex(2, 0);
        let from = compiler.add_node(1);
        let c2 = compiler.add_node(2);
        let shared_assignment = ValueIndex(0);
        compiler.add_edge(1, from, c2, shared_assignment);

        compiler.merge_nodes(from, into);

        assert!(!compiler.mdd()[from].is_active());
        assert!(compiler.mdd()[into].is_active());
        assert!(compiler.mdd()[c1].is_active());
        assert!(!compiler.mdd()[c2].is_active());

        let mut seen_assignments = std::collections::HashSet::new();
        for edge in compiler.mdd()[into].iter_children() {
            if !compiler.mdd()[edge].is_active() {
                continue;
            }
            assert!(seen_assignments.insert(compiler.mdd()[edge].assignment()));
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
            let mut compiler = Compiler::new(
                problem,
                CompilationHeuristic {
                    ordering: OrderingHeuristic::MinDomMaxLinked,
                    merge: MergeHeuristic::LessRelaxed,
                    selection: SelectHeuristic::Greedy,
                },
                &constraints,
            );
            compiler.refine(usize::MAX);
            let solutions = get_all_solutions(compiler.mdd());
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

    /// Regression test: a merge cascade can name a node that an earlier entry of the same cascade
    /// already absorbed. Merging into that dead node lost its paths: with `StateSimilarity`, six
    /// `AllDifferent` variables of six values and a width of 4 kept only 289 of the 720 solutions.
    #[test]
    pub fn merging_follows_nodes_absorbed_earlier_in_the_same_cascade() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(6, (0..6).collect(), None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::StateSimilarity,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        compiler.refine(4);
        assert!(compiler.mdd().count_solutions() >= BigUint::from(720u32));
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
                let mut compiler = Compiler::new(
                    problem,
                    CompilationHeuristic {
                        ordering: OrderingHeuristic::MinDomMaxLinked,
                        merge: MergeHeuristic::LessRelaxed,
                        selection: SelectHeuristic::Greedy,
                    },
                    &constraints,
                );
                compiler.refine(max_width);
                let candidate_solutions = get_all_solutions(compiler.mdd());
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
            let mut compiler = Compiler::new(
                Arc::clone(&problem),
                CompilationHeuristic {
                    ordering: OrderingHeuristic::MinDomMaxLinked,
                    merge: MergeHeuristic::StateSimilarity,
                    selection: SelectHeuristic::Greedy,
                },
                &constraints,
            );
            compiler.refine(max_width);

            for layer in 1..compiler.mdd().number_layers() {
                assert!(
                    compiler.mdd().number_nodes_in_layer(layer) <= max_width,
                    "layer {} has {} nodes, exceeding max_width={}",
                    layer,
                    compiler.mdd().number_nodes_in_layer(layer),
                    max_width
                );
            }

            let candidate_solutions = get_all_solutions(compiler.mdd());
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
        let mut compiler = Compiler::new(
            Arc::clone(&problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        compiler.refine(usize::MAX);

        let true_count = candidates
            .iter()
            .filter(|assignment| problem.is_solution(assignment))
            .count();
        let path_count = get_all_solutions(compiler.mdd()).len();
        assert_eq!(
            path_count, true_count,
            "get_all_solutions found {} but brute force found {}",
            path_count, true_count
        );
        assert_eq!(
            compiler.mdd().count_solutions(),
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
        let compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        assert!(compiler.mdd().is_unsat());
        assert_eq!(compiler.mdd().count_solutions(), BigUint::from(0u32));
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
            let mut compiler = Compiler::new(
                Arc::clone(&problem),
                CompilationHeuristic {
                    ordering: OrderingHeuristic::MinDomMaxLinked,
                    merge: MergeHeuristic::StateSimilarity,
                    selection: SelectHeuristic::Greedy,
                },
                &constraints,
            );
            compiler.refine(max_width);
            let true_count = candidates
                .iter()
                .filter(|assignment| problem.is_solution(assignment))
                .count();
            assert!(
                compiler.mdd().count_solutions() >= BigUint::from(true_count),
                "max_width={} count_solutions()={} is less than the true count {} -- unsound relaxation",
                max_width,
                compiler.mdd().count_solutions(),
                true_count
            );
        }
    }

    #[test]
    fn merging_does_not_read_removed_edges() {
        // Three variables with the domain {0, 1, 2}, and at least two of them take a value in
        // {0, 1}. With the order [2, 1, 0] and no width limit, `merge_nodes` used to
        // take the edges that a split had removed for live ones of the node it merged into,
        // and the solutions [2, 0, 0] and [2, 0, 1] disappeared from the MDD.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        at_least(&mut problem, vars, vec![0, 1], 2);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut compiler = Compiler::new(
            Arc::clone(&problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::Custom(vec![2, 1, 0]),
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        compiler.refine(usize::MAX);
        let solutions = get_all_solutions(compiler.mdd());
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let assignment = vec![x, y, z];
                    let satisfied = assignment.iter().filter(|&&v| v < 2).count() >= 2;
                    assert_eq!(
                        is_solution(assignment.clone(), &solutions),
                        satisfied,
                        "{assignment:?}"
                    );
                }
            }
        }
    }

    /// An all-different over three variables of two values: no solution.
    fn unsatisfiable() -> Compiler {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        )
    }

    #[test]
    fn an_unsatisfiable_diagram_holds_nothing() {
        let compiler = unsatisfiable();
        let mdd = compiler.mdd();
        assert!(mdd.is_unsat());
        assert!(mdd.nodes.is_empty());
        assert!(mdd.edges.is_empty());
        assert!(mdd.order.is_empty());
        assert!(compiler.top_down_properties.is_empty());
        assert!(compiler.bottom_up_properties.is_empty());
        assert_eq!(mdd.number_layers(), 0);
        assert_eq!(mdd.get_solution(), None);
        assert_eq!(format!("{mdd:?}"), "UNSAT");
    }

    #[test]
    fn refining_an_unsatisfiable_diagram_changes_nothing() {
        let mut compiler = unsatisfiable();
        compiler.refine(usize::MAX);
        compiler.refine(2);
        assert!(compiler.mdd().is_unsat());
        assert!(compiler.mdd().nodes.is_empty());
    }

    #[test]
    fn the_root_and_the_sink_stay_alone_in_their_layer_at_every_width() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        for width in [1, 2, 3, usize::MAX] {
            let mut compiler = Compiler::new(
                Arc::clone(&problem),
                CompilationHeuristic {
                    ordering: OrderingHeuristic::MinDomMaxLinked,
                    merge: MergeHeuristic::LessRelaxed,
                    selection: SelectHeuristic::Greedy,
                },
                &constraints,
            );
            compiler.refine(width);
            compiler.mdd().debug_assert_root_and_sink();
            let last_layer = compiler.mdd().number_layers() - 1;
            assert_eq!(compiler.mdd().root(), NodeIndex(0, 0), "width {width}");
            assert_eq!(
                compiler.mdd().sink(),
                NodeIndex(last_layer, 0),
                "width {width}"
            );
            assert_eq!(compiler.mdd().number_nodes_in_layer(0), 1, "width {width}");
            assert_eq!(
                compiler.mdd().number_nodes_in_layer(last_layer),
                1,
                "width {width}"
            );
        }
    }

    /// Five variables that must all differ, compiled but not refined.
    fn five_different_variables() -> Compiler {
        let mut problem = Problem::default();
        let vars = problem.add_variables(5, vec![0, 1, 2, 3, 4], None);
        let constraint = crate::constraints::all_different(&mut problem, vars);
        Compiler::new(
            Arc::new(problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &[constraint],
        )
    }

    #[test]
    fn refining_with_growing_widths_keeps_every_solution_and_respects_the_width() {
        let mut compiler = five_different_variables();
        assert!(compiler.mdd().count_solutions() >= 120u32.into());
        for width in [1, 2, 4, 8, 16] {
            compiler.refine(width);
            let mdd = compiler.mdd();
            let layers = mdd.number_layers();
            assert!((0..layers).all(|layer| mdd.number_nodes_in_layer(layer) <= width));
            // Relaxing never loses a solution: the 120 permutations stay in the diagram.
            assert!(mdd.count_solutions() >= 120u32.into());
        }
    }

    /// Whether `assignment` (one value per variable) is a path of `mdd`.
    fn is_a_path(mdd: &Mdd, assignment: &[isize]) -> bool {
        let mut node = mdd.root();
        for layer in 0..mdd.number_layers() - 1 {
            let variable = mdd.decision_at_layer(layer);
            let next = mdd[node].iter_children().find(|&edge| {
                mdd[edge].is_active()
                    && mdd.problem()[variable].value(mdd[edge].assignment())
                        == assignment[variable.0]
            });
            match next {
                Some(edge) => node = mdd[edge].to(),
                None => return false,
            }
        }
        true
    }

    #[test]
    fn a_solution_names_the_variables_in_the_branching_order() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        let constraint = not_equals(&mut problem, vars[3], vars[1]);
        let mut compiler = Compiler::new(
            Arc::new(problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::Custom(vec![3, 1]),
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &[constraint],
        );
        compiler.refine(usize::MAX);
        let (values, variables) = compiler.mdd().get_solution().expect("satisfiable");
        assert_eq!(variables, vec![VariableIndex(3), VariableIndex(1)]);
        assert_eq!(values.len(), 2);
        assert_ne!(values[0], values[1]);
    }

    /// Checks that the stored properties describe at least the paths of the diagram: each one
    /// joined with the property recomputed from the neighbouring nodes is itself. A property that
    /// describes less would filter out solutions. With `same`, they must be equal, as when
    /// nothing is left coarse. The stored properties are put back afterwards.
    fn assert_properties_cover_the_paths(compiler: &mut Compiler, same: bool, context: &str) {
        let stored_top_down = compiler.top_down_properties.clone();
        let stored_bottom_up = compiler.bottom_up_properties.clone();
        compiler.update_top_down_properties();
        for layer in (0..compiler.mdd.nodes.len() - 1).rev() {
            let variable = compiler.mdd.order[layer];
            for index in 0..compiler.mdd.nodes[layer].len() {
                let node = NodeIndex(layer, index);
                if !compiler.mdd[node].is_active() {
                    continue;
                }
                for constraint in 0..compiler.constraints.len() {
                    compiler.bottom_up_properties[layer][index][constraint] =
                        compiler.fold_property_over_children(node, variable, constraint);
                }
            }
        }
        let check = |stored: &[Vec<Vec<Box<dyn ConstraintProperty>>>],
                     fresh: &[Vec<Vec<Box<dyn ConstraintProperty>>>],
                     layers: std::ops::Range<usize>,
                     direction: &str| {
            for layer in layers {
                for index in 0..stored[layer].len() {
                    if !compiler.mdd[NodeIndex(layer, index)].is_active() {
                        continue;
                    }
                    for (stored, fresh) in stored[layer][index].iter().zip(&fresh[layer][index]) {
                        let mut joined = stored.clone();
                        joined.merge(fresh.as_ref());
                        assert!(
                            ConstraintProperty::eq(joined.as_ref(), stored.as_ref()),
                            "{context}: a {direction} property of node ({layer}, {index}) \
                             describes fewer paths than the node holds"
                        );
                        assert!(
                            !same || ConstraintProperty::eq(stored.as_ref(), fresh.as_ref()),
                            "{context}: a {direction} property of node ({layer}, {index}) is \
                             coarser than the node's paths"
                        );
                    }
                }
            }
        };
        // The root's top-down and the sink's bottom-up properties are fixed, and not recomputed.
        let layers = stored_top_down.len();
        check(
            &stored_top_down,
            &compiler.top_down_properties,
            1..layers,
            "top-down",
        );
        check(
            &stored_bottom_up,
            &compiler.bottom_up_properties,
            0..layers - 1,
            "bottom-up",
        );
        compiler.top_down_properties = stored_top_down;
        compiler.bottom_up_properties = stored_bottom_up;
    }

    #[test]
    fn no_schedule_of_refinements_loses_a_solution() {
        use crate::constraints::{all_different, among, at_least, gcc, not_equals, sum};
        let variables = 5;
        let domain: Vec<isize> = (0..5).collect();
        for case in 0..5 {
            let mut problem = Problem::default();
            let vars = problem.add_variables(variables, domain.clone(), None);
            let constraints = match case {
                0 => vec![gcc(
                    &mut problem,
                    vars,
                    vec![(0, 1, 2), (1, 0, 2), (2, 1, 3)],
                )],
                1 => vec![sum(&mut problem, vars, 9)],
                2 => vec![all_different(&mut problem, vars)],
                3 => vec![
                    all_different(&mut problem, vars[..3].to_vec()),
                    not_equals(&mut problem, vars[3], vars[4]),
                ],
                _ => vec![
                    at_least(&mut problem, vars.clone(), vec![2], 2),
                    among(&mut problem, vars, vec![0, 1], 2, 4),
                ],
            };
            let problem = Arc::new(problem);
            // The solutions, by brute force.
            let mut solutions: Vec<Vec<isize>> = vec![];
            let mut assignment = vec![0isize; variables];
            loop {
                if constraints
                    .iter()
                    .all(|&constraint| problem[constraint].is_satisfied(&assignment))
                {
                    solutions.push(assignment.clone());
                }
                let mut position = 0;
                while position < variables {
                    assignment[position] += 1;
                    if assignment[position] < 5 {
                        break;
                    }
                    assignment[position] = 0;
                    position += 1;
                }
                if position == variables {
                    break;
                }
            }
            let mut seed = 7u64;
            let mut next = |bound: u64| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) % bound
            };
            let mut schedules: Vec<Vec<usize>> =
                vec![vec![1, 10], vec![5, 9], vec![6, 9], vec![3, 3, 7, 7]];
            for _ in 0..25 {
                let calls = 2 + next(3) as usize;
                schedules.push((0..calls).map(|_| 1 + next(12) as usize).collect());
            }
            for merge in [MergeHeuristic::LessRelaxed, MergeHeuristic::StateSimilarity] {
                for schedule in &schedules {
                    let mut compiler = Compiler::new(
                        Arc::clone(&problem),
                        CompilationHeuristic {
                            ordering: OrderingHeuristic::MinDomMaxLinked,
                            merge: merge.clone(),
                            selection: SelectHeuristic::Greedy,
                        },
                        &constraints,
                    );
                    for &width in schedule.iter().chain(&[usize::MAX]) {
                        compiler.refine(width);
                        let mdd = compiler.mdd();
                        assert!(
                            solutions.iter().all(|solution| is_a_path(mdd, solution)),
                            "case {case}, {merge:?}, widths {schedule:?} then {width}: a solution was lost"
                        );
                        assert_properties_cover_the_paths(
                            &mut compiler,
                            width == usize::MAX,
                            &format!("case {case}, {merge:?}, widths {schedule:?} then {width}"),
                        );
                    }
                    assert_eq!(
                        compiler.mdd().count_solutions(),
                        solutions.len().into(),
                        "case {case}, {merge:?}, widths {schedule:?} then MAX: not exact"
                    );
                }
            }
        }
    }

    #[test]
    fn refining_in_several_calls_ends_as_exact_as_refining_once() {
        let exact = {
            let mut compiler = five_different_variables();
            compiler.refine(usize::MAX);
            compiler.into_mdd()
        };
        let widths = |mdd: &Mdd| -> Vec<usize> {
            (0..mdd.number_layers())
                .map(|l| mdd.number_nodes_in_layer(l))
                .collect()
        };
        for schedule in [
            vec![1, usize::MAX],
            vec![2, usize::MAX],
            vec![4, usize::MAX],
            vec![1, 10, usize::MAX],
            vec![2, 4, 8, usize::MAX],
            vec![3, 3, 7, 7, usize::MAX],
            vec![2, usize::MAX, usize::MAX],
        ] {
            let mut compiler = five_different_variables();
            for &width in &schedule {
                compiler.refine(width);
            }
            let mdd = compiler.into_mdd();
            assert_eq!(
                mdd.count_solutions(),
                120u32.into(),
                "schedule {schedule:?}"
            );
            assert!(
                (0..mdd.number_layers())
                    .flat_map(|layer| mdd.nodes_in_layer(layer))
                    .all(|node| !mdd[node].is_relaxed()),
                "schedule {schedule:?}"
            );
            assert!(
                widths(&mdd)
                    .iter()
                    .zip(widths(&exact))
                    .all(|(w, e)| *w >= e)
            );
        }
    }

    #[test]
    fn the_diagram_can_be_read_between_refinements_and_taken_at_the_end() {
        let mut compiler = five_different_variables();
        compiler.refine(16);
        let coarse_layers = compiler.mdd().number_layers();
        compiler.refine(usize::MAX);
        let mdd = compiler.into_mdd();
        assert_eq!(mdd.number_layers(), coarse_layers);
        assert_eq!(mdd.count_solutions(), 120u32.into());
        let (mut solution, _) = mdd.get_solution().expect("the diagram is satisfiable");
        solution.sort();
        assert_eq!(solution, vec![0, 1, 2, 3, 4]);
    }

    /// A compiler of two different variables, the merge heuristic being `merge`.
    fn two_variables_with(merge: MergeHeuristic) -> Compiler {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        )
    }

    #[test]
    fn most_likely_takes_the_probabilities_it_is_given() {
        let probabilities = vec![vec![0.2, 0.3, 0.5], vec![1.0, 0.0, 0.0]];
        let mut compiler = two_variables_with(MergeHeuristic::MostLikely(probabilities));
        compiler.refine(1);
        assert!(!compiler.mdd().is_unsat());
    }

    #[test]
    #[should_panic(expected = "one row of probabilities per variable")]
    fn most_likely_needs_a_row_per_variable() {
        two_variables_with(MergeHeuristic::MostLikely(vec![vec![0.2, 0.3, 0.5]]));
    }

    #[test]
    #[should_panic(expected = "they sum to")]
    fn most_likely_needs_distributions() {
        two_variables_with(MergeHeuristic::MostLikely(vec![
            vec![0.2, 0.3, 0.5],
            vec![0.5, 0.5, 0.5],
        ]));
    }

    #[test]
    #[should_panic(expected = "2 probabilities were given for a domain of 3 values")]
    fn most_likely_needs_a_probability_per_value() {
        two_variables_with(MergeHeuristic::MostLikely(vec![
            vec![0.5, 0.5],
            vec![0.2, 0.3, 0.5],
        ]));
    }

    #[test]
    fn refining_in_several_calls_gives_the_solutions_of_a_single_refinement_for_every_constraint() {
        use crate::constraints::{among, at_least, gcc, not_equals, sum};
        for case in 0..4 {
            let build = || {
                let mut problem = Problem::default();
                let vars = problem.add_variables(6, vec![0, 1, 2, 3], None);
                let constraints = match case {
                    0 => vec![gcc(
                        &mut problem,
                        vars,
                        vec![(0, 1, 2), (1, 0, 2), (2, 1, 3)],
                    )],
                    1 => vec![sum(&mut problem, vars, 9)],
                    2 => vec![
                        crate::constraints::all_different(&mut problem, vars[..4].to_vec()),
                        not_equals(&mut problem, vars[4], vars[5]),
                    ],
                    _ => vec![
                        at_least(&mut problem, vars.clone(), vec![2], 2),
                        among(&mut problem, vars, vec![0, 1], 2, 4),
                    ],
                };
                Compiler::new(
                    Arc::new(problem),
                    CompilationHeuristic {
                        ordering: OrderingHeuristic::MinDomMaxLinked,
                        merge: MergeHeuristic::LessRelaxed,
                        selection: SelectHeuristic::Greedy,
                    },
                    &constraints,
                )
            };
            let mut once = build();
            once.refine(usize::MAX);
            let expected = once.mdd().count_solutions();
            for schedule in [
                vec![1, usize::MAX],
                vec![3, usize::MAX],
                vec![2, 5, usize::MAX],
            ] {
                let mut compiler = build();
                for &width in &schedule {
                    compiler.refine(width);
                }
                assert_eq!(
                    compiler.mdd().count_solutions(),
                    expected,
                    "case {case}, schedule {schedule:?}"
                );
            }
        }
    }

    /// For each node, the distinct sets of values on the paths from the root to it.
    fn value_sets_of_paths(mdd: &Mdd) -> Vec<Vec<std::collections::BTreeSet<Vec<usize>>>> {
        let mut sets: Vec<Vec<std::collections::BTreeSet<Vec<usize>>>> = (0..mdd.number_layers())
            .map(|layer| vec![Default::default(); mdd.number_nodes_in_layer(layer)])
            .collect();
        sets[0][0].insert(vec![]);
        for layer in 0..mdd.number_layers() - 1 {
            for node in mdd.nodes_in_layer(layer) {
                let here = sets[layer][node.1].clone();
                for edge in mdd[node].iter_children() {
                    let NodeIndex(to_layer, to_position) = mdd[edge].to();
                    for set in &here {
                        let mut set = set.clone();
                        set.push(mdd[edge].assignment().0);
                        set.sort();
                        sets[to_layer][to_position].insert(set);
                    }
                }
            }
        }
        sets
    }

    #[test]
    fn a_node_is_relaxed_exactly_when_its_paths_differ_or_a_parent_is_relaxed() {
        // The definition of an exact node: every path to it leads to the same state (for
        // `AllDifferent`, takes the same set of values) and its parents are exact.
        for width in [1, 2, 3, 4, 5, 6, 8, 12, 16, usize::MAX] {
            let mut compiler = five_different_variables();
            compiler.refine(width);
            let mdd = compiler.mdd();
            let sets = value_sets_of_paths(mdd);
            let mut expected: Vec<Vec<bool>> = vec![];
            for layer in 0..mdd.number_layers() {
                let relaxed = mdd
                    .nodes_in_layer(layer)
                    .map(|node| {
                        layer != 0
                            && (sets[layer][node.1].len() > 1
                                || mdd[node]
                                    .iter_parents()
                                    .any(|edge| expected[layer - 1][mdd[edge].from().1]))
                    })
                    .collect();
                expected.push(relaxed);
            }
            for (layer, expected_in_layer) in expected.iter().enumerate() {
                for node in mdd.nodes_in_layer(layer) {
                    assert_eq!(
                        mdd[node].is_relaxed(),
                        expected_in_layer[node.1],
                        "width {width}, node {node:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_exact_diagram_has_no_relaxed_node() {
        let mut compiler = five_different_variables();
        compiler.refine(usize::MAX);
        let mdd = compiler.mdd();
        assert!(
            (0..mdd.number_layers())
                .flat_map(|layer| mdd.nodes_in_layer(layer))
                .all(|node| !mdd[node].is_relaxed())
        );
    }
}
