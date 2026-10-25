//! Structure representing a decision diagram itself, as compiled by the
//! [`Compiler`](super::Compiler)
//!
//! # Meaning
//!
//! An [`Mdd`] is what a [`Compiler`](super::Compiler) produces: the nodes and edges of the
//! diagram, the variable branched on at each layer and the problem that gives the variables and
//! their values a meaning. It carries nothing that only the compilation needs (the constraints,
//! their properties, the heuristics), so a finished diagram is small, and the counting, sampling and learning code reads it directly.
//!
//! # How it works
//!
//! `nodes[l]` holds the nodes of layer `l` and `edges[l]` the edges from layer `l` to layer
//! `l + 1`, each labelled with a value of the variable `order[l]`. The last layer holds the sink.
//! The fields are visible to the rest of the `mdd` module, which builds and reads them; outside
//! of it a diagram is read through the methods below.
//!
//! A diagram has these properties, which the code that reads it relies on:
//!
//! - the layers are numbered from the root's (0) to the sink's, and `number_layers` counts the
//!   sink's layer (one more than the number of variables);
//! - the root is the only node of layer 0 and the sink the only node of the last layer, so they
//!   are always `NodeIndex(0, 0)` and `NodeIndex(last_layer, 0)`;
//! - an unsatisfiable diagram holds nothing: it has no layer at all, and has neither root nor sink;
//! - every node and edge present is active (the diagram has been cleaned), and an edge goes from
//!   a layer to the next one. It means that the compiler *must* clean the diagram at some point
//!   during the compilation process.

use super::*;
use crate::modelling::*;

use num_bigint::BigUint;
use std::sync::Arc;

/// A decision diagram of the solutions of some constraints of a problem. A [`Compiler`](crate::mdd::Compiler)
/// builds and refines it; see [`mdd`](crate::mdd) for how.
pub struct Mdd {
    /// The problem whose variables and values the diagram speaks about.
    pub(super) problem: Arc<Problem>,
    /// Nodes of the diagram.
    pub(super) nodes: Vec<Vec<Node>>,
    /// Edges of the diagram.
    pub(super) edges: Vec<Vec<Edge>>,
    /// Variable branched on at each layer.
    pub(super) order: Vec<VariableIndex>,
}

impl Mdd {
    /// A diagram of `number_layers` variables with every layer empty, and one more (the sink's).
    /// The order is empty: the builder sets it before it adds any edge.
    pub(super) fn with_layers(problem: Arc<Problem>, number_layers: usize) -> Self {
        Self {
            problem,
            nodes: vec![vec![]; number_layers + 1],
            edges: vec![vec![]; number_layers],
            order: vec![],
        }
    }

    /// A diagram made of the given nodes and edges, with no order and an empty problem, to test
    /// the code that reads diagrams on shapes that the compiler never produces.
    #[cfg(test)]
    pub(crate) fn from_parts(nodes: Vec<Vec<Node>>, edges: Vec<Vec<Edge>>) -> Self {
        Self {
            problem: Arc::new(Problem::default()),
            nodes,
            edges,
            order: vec![],
        }
    }

    /// Total number of nodes, in every layer.
    #[cfg(test)]
    pub(crate) fn number_nodes(&self) -> usize {
        self.nodes.iter().map(|layer| layer.len()).sum()
    }

    /// Total number of edges, in every layer.
    #[cfg(test)]
    pub(crate) fn number_edges(&self) -> usize {
        self.edges.iter().map(|layer| layer.len()).sum()
    }

    /// Turns the diagram into the unsatisfiable one: no layer, no node, no edge and no order.
    pub(super) fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.order.clear();
    }

    /// Number of layers, the sink's included; 0 for an unsatisfiable diagram.
    pub fn number_layers(&self) -> usize {
        self.nodes.len()
    }

    /// Number of nodes in `layer`.
    pub fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.nodes[layer].len()
    }

    /// `true` if the MDD is unsat. An unsat MDD has been cleared, so it has no nodes.
    pub fn is_unsat(&self) -> bool {
        self.number_layers() == 0
    }

    /// Returns the root of the MDD. Panics if the MDD is unsat.
    pub fn root(&self) -> NodeIndex {
        assert!(!self.is_unsat(), "an unsatisfiable diagram has no root");
        NodeIndex(0, 0)
    }

    /// Returns the sink of the MDD. Panics if the MDD is unsat.
    pub fn sink(&self) -> NodeIndex {
        assert!(!self.is_unsat(), "an unsatisfiable diagram has no sink");
        NodeIndex(self.number_layers() - 1, 0)
    }

    /// Iterates over the nodes of `layer`.
    pub fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.number_nodes_in_layer(layer)).map(move |index| NodeIndex(layer, index))
    }

    /// Checks, in debug builds only, that the first layer and the last layer hold a single node
    /// each, which is what makes the root and the sink `NodeIndex(0, 0)` and
    /// `NodeIndex(last_layer, 0)`. Does nothing for an unsatisfiable diagram.
    pub fn debug_assert_root_and_sink(&self) {
        if self.is_unsat() {
            return;
        }
        let last_layer = self.number_layers() - 1;
        debug_assert_eq!(
            self.number_nodes_in_layer(0),
            1,
            "the root layer must have exactly one node"
        );
        debug_assert_eq!(
            self.number_nodes_in_layer(last_layer),
            1,
            "the sink layer must have exactly one node"
        );
    }

    /// Counts the number of root-to-sink paths (i.e. encoded assignments) of the diagram, which is
    /// the number of solutions if the diagram is exact and an upper bound if it is relaxed.
    ///
    /// It is a dynamic program over the layers: the sink counts 1 and a node counts the sum of the
    /// counts of its children, computed from the sink up to the root. Runs in O(nodes + edges).
    /// Returns 0 for an unsatisfiable diagram.
    pub fn count_solutions(&self) -> BigUint {
        if self.is_unsat() {
            return BigUint::from(0u32);
        }
        let last_layer = self.number_layers() - 1;
        let mut counts: Vec<Vec<BigUint>> = (0..=last_layer)
            .map(|layer| {
                let value = BigUint::from(u32::from(layer == last_layer));
                vec![value; self.number_nodes_in_layer(layer)]
            })
            .collect();
        for layer in (0..last_layer).rev() {
            for node in self.nodes_in_layer(layer) {
                let mut total = BigUint::from(0u32);
                for edge in self[node].iter_children() {
                    let NodeIndex(to_layer, to_index) = self[edge].to();
                    total += &counts[to_layer][to_index];
                }
                counts[layer][node.1] = total;
            }
        }
        counts[0][0].clone()
    }

    /// Reads one solution off the diagram: the value index chosen at each layer (but the sink's)
    /// along a path whose nodes are all exact, i.e. not relaxed, or `None` if there is no such
    /// path.
    ///
    /// The layers are visited from the sink up. A node can *continue* if it is exact and has a
    /// child edge into the sink or into a node that can continue; the first such edge is kept as
    /// the node's way on. This is one active path per node: a node with no exact way down is
    /// dropped, and so are the paths that reach it. The root can continue exactly when an exact
    /// path exists, and following the kept edges from it reads the path. It runs in
    /// O(nodes + edges) and needs no backtracking.
    ///
    /// A path of exact nodes is a solution of the compiled constraints, since none of its nodes is
    /// the result of a lossy merge.
    pub fn exact_path(&self) -> Option<Vec<ValueIndex>> {
        if self.is_unsat() {
            return None;
        }
        let sink_layer = self.number_layers() - 1;
        // `way_on[layer][position]` is the child edge that the node continues by, if it can.
        let mut way_on: Vec<Vec<Option<EdgeIndex>>> = (0..sink_layer)
            .map(|layer| vec![None; self.number_nodes_in_layer(layer)])
            .collect();
        for layer in (0..sink_layer).rev() {
            for node in self.nodes_in_layer(layer) {
                if self[node].is_relaxed() {
                    continue;
                }
                way_on[layer][node.1] = self[node].iter_children().find(|&edge| {
                    let NodeIndex(child_layer, child_position) = self[edge].to();
                    child_layer == sink_layer || way_on[child_layer][child_position].is_some()
                });
            }
        }
        let mut path = Vec::with_capacity(sink_layer);
        let mut node = self.root();
        while node.0 < sink_layer {
            match way_on[node.0][node.1] {
                Some(edge) => {
                    path.push(self[edge].assignment());
                    node = self[edge].to();
                }
                None => return None,
            };
        }
        Some(path)
    }

    /// Number of variables branched on, which is the number of layers but the sink's.
    pub fn number_decisions(&self) -> usize {
        self.order.len()
    }

    /// The variable branched on at `layer`. Panics if `layer` is the sink's layer or beyond, or if
    /// the diagram is unsatisfiable (it then has no layers).
    pub fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        self.order[layer]
    }

    /// The problem this diagram was compiled from.
    pub fn problem(&self) -> &Problem {
        &self.problem
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use crate::mdd::heuristics::*;
    use crate::mdd::{Compiler, Mdd, Node};
    use crate::modelling::Problem;
    use std::sync::Arc;

    fn compiled(unsat: bool) -> Mdd {
        let mut problem = Problem::default();
        let domain = if unsat { vec![0] } else { vec![0, 1, 2] };
        let vars = problem.add_variables(3, domain, None);
        let c = all_different(&mut problem, vars);
        let mut compiler = Compiler::new(
            Arc::new(problem),
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &[c],
        );
        compiler.refine(usize::MAX);
        compiler.into_mdd()
    }

    #[test]
    fn the_first_layer_holds_the_root_and_the_last_the_sink() {
        let mdd = compiled(false);
        assert_eq!(mdd.root(), NodeIndex(0, 0));
        assert_eq!(mdd.number_layers(), 4);
        assert_eq!(mdd.sink().0, mdd.number_layers() - 1);
        assert_eq!(mdd.number_nodes_in_layer(0), 1);
        assert_eq!(mdd.number_nodes_in_layer(3), 1);
        assert!(!mdd.is_unsat());
    }

    #[test]
    fn the_order_names_each_variable_once() {
        let mdd = compiled(false);
        let mut seen: Vec<_> = (0..mdd.number_decisions())
            .map(|l| *mdd.decision_at_layer(l))
            .collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[test]
    fn an_unsatisfiable_diagram_holds_nothing() {
        let mdd = compiled(true);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.number_layers(), 0);
        assert_eq!(mdd.count_solutions().to_string(), "0");
        assert_eq!(mdd.exact_path(), None);
        mdd.debug_assert_root_and_sink();
    }

    #[test]
    #[should_panic(expected = "an unsatisfiable diagram has no root")]
    fn an_unsatisfiable_diagram_has_no_root() {
        compiled(true).root();
    }

    #[test]
    #[should_panic(expected = "an unsatisfiable diagram has no sink")]
    fn an_unsatisfiable_diagram_has_no_sink() {
        compiled(true).sink();
    }

    #[test]
    fn the_solutions_are_counted_and_an_exact_one_is_read_off() {
        let mdd = compiled(false);
        assert_eq!(mdd.count_solutions().to_string(), "6");
        let path = mdd.exact_path().expect("the diagram is exact");
        assert_eq!(path.len(), 3);
        // The values of a permutation of three values are all different.
        let mut values: Vec<usize> = path.iter().map(|value| value.0).collect();
        values.sort();
        assert_eq!(values, vec![0, 1, 2]);
    }

    /// A diagram of one variable with two values, with `roots` nodes in the first layer.
    fn structure_with_roots(roots: usize) -> Mdd {
        let mut nodes: Vec<Vec<Node>> = vec![(0..roots).map(|i| Node::new(0, i, false)).collect()];
        nodes.push(vec![Node::new(1, 0, true)]);
        Mdd::from_parts(nodes, vec![vec![]])
    }

    #[test]
    fn a_single_root_and_sink_pass_the_check() {
        structure_with_roots(1).debug_assert_root_and_sink();
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "the root layer must have exactly one node")]
    fn two_roots_fail_the_check() {
        structure_with_roots(2).debug_assert_root_and_sink();
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "the sink layer must have exactly one node")]
    fn two_sinks_fail_the_check() {
        let mut structure = structure_with_roots(1);
        structure.nodes[1].push(Node::new(1, 1, true));
        structure.debug_assert_root_and_sink();
    }

    /// A diagram of two variables whose nodes are given by `relaxed` (`[layer 1, layer 2]`, two
    /// nodes in each) and whose edges are `R -> Z (0), R -> W (1)` in the first layer,
    /// `Z -> C (0), W -> E (0)` in the second and `C -> sink (0), E -> sink (1)` in the third,
    /// where `Z`, `W` are the nodes of layer 1 and `C`, `E` those of layer 2. The sink lists `C`
    /// first.
    fn fork(relaxed: [[bool; 2]; 2]) -> Mdd {
        let node = |layer: usize, index: usize, relaxed: bool| Node::new(layer, index, relaxed);
        let mut nodes = vec![
            vec![node(0, 0, false)],
            vec![node(1, 0, relaxed[0][0]), node(1, 1, relaxed[0][1])],
            vec![node(2, 0, relaxed[1][0]), node(2, 1, relaxed[1][1])],
            vec![node(3, 0, true)],
        ];
        let links = [
            (NodeIndex(0, 0), NodeIndex(1, 0), 0),
            (NodeIndex(0, 0), NodeIndex(1, 1), 1),
            (NodeIndex(1, 0), NodeIndex(2, 0), 0),
            (NodeIndex(1, 1), NodeIndex(2, 1), 0),
            (NodeIndex(2, 0), NodeIndex(3, 0), 0),
            (NodeIndex(2, 1), NodeIndex(3, 0), 1),
        ];
        let mut edges: Vec<Vec<Edge>> = vec![vec![]; 3];
        for (from, to, value) in links {
            let edge_index = EdgeIndex(from.0, edges[from.0].len());
            nodes[from.0][from.1].add_child_edge(edge_index);
            nodes[to.0][to.1].add_parent_edge(edge_index);
            edges[from.0].push(Edge::new(from, to, ValueIndex(value)));
        }
        Mdd::from_parts(nodes, edges)
    }

    #[test]
    fn an_exact_path_is_found_even_when_the_first_branch_is_a_dead_end() {
        // Z is relaxed, so the branch through C (the first parent of the sink) is a dead end; the
        // path through W and E is exact.
        let structure = fork([[true, false], [false, false]]);
        assert_eq!(
            structure.exact_path(),
            Some(vec![ValueIndex(1), ValueIndex(0), ValueIndex(1)])
        );
    }

    #[test]
    fn the_first_exact_path_is_the_one_returned() {
        let structure = fork([[false, false], [false, false]]);
        assert_eq!(
            structure.exact_path(),
            Some(vec![ValueIndex(0), ValueIndex(0), ValueIndex(0)])
        );
    }

    #[test]
    fn there_is_no_exact_path_when_every_branch_crosses_a_relaxed_node() {
        assert_eq!(fork([[true, false], [false, true]]).exact_path(), None);
        assert_eq!(fork([[true, true], [false, false]]).exact_path(), None);
        assert_eq!(fork([[false, false], [true, true]]).exact_path(), None);
    }

    #[test]
    fn a_diagram_without_variables_has_the_empty_path() {
        let structure = Mdd::from_parts(vec![vec![Node::new(0, 0, false)]], vec![]);
        assert_eq!(structure.exact_path(), Some(vec![]));
    }
}
