//! Ways to look at an `Mdd` in terms of its problem: solutions, Graphviz, a text dump, memory.
//!
//! # Meaning
//!
//! The methods of this module read a diagram and translate it with the variables and values of the
//! problem it was compiled from, which the diagram alone does not tell. None of them
//! changes the diagram.
//!
//! # Design notes
//!
//! - **Printing goes through the caller.** `as_graphviz` returns the text and `to_file` writes it
//!   where asked; the memory report goes through `log`, so that the caller chooses what is shown.

use super::memory::MemoryReport;
use super::*;
use crate::modelling::{ValueIndex, VariableIndex};

use rustc_hash::FxHashMap;
use std::fs;

impl Mdd {
    /// Reads one solution off the diagram, or `None` if the diagram has no path of exact nodes (in
    /// particular if it is unsatisfiable). A relaxed diagram can hold solutions that this does
    /// not find: see [`Mdd::exact_path`], which finds a path whenever there is one.
    ///
    /// The solution is given as in [`Mdd::solution_from_path`]: the values, and the variables
    /// they are the values of.
    pub fn get_solution(&self) -> Option<(Vec<isize>, Vec<VariableIndex>)> {
        Some(self.solution_from_path(&self.exact_path()?))
    }

    /// The assignment that a path of the diagram stands for, given as the value index taken at
    /// each layer (as returned by [`Mdd::exact_path`] or `sampling::sample_solution`). It is
    /// returned as the values taken and the variables they are the values of, both in the
    /// branching order of the diagram: the first value is the one of the variable of the first
    /// layer. A diagram compiled from constraints that do not cover every variable of the
    /// problem holds only the variables in their scope, and its branching order is not the order
    /// the variables were declared in, so the variables are returned too.
    ///
    /// Panics if `path` is not as long as the number of variables branched on.
    pub fn solution_from_path(&self, path: &[ValueIndex]) -> (Vec<isize>, Vec<VariableIndex>) {
        assert_eq!(
            path.len(),
            self.order.len(),
            "a path needs a value for each of the {} variables of the diagram",
            self.order.len()
        );
        path.iter()
            .zip(&self.order)
            .map(|(&value, &variable)| (self.problem[variable].value(value), variable))
            .unzip()
    }

    /// The diagram in Graphviz's DOT format: one point per node, one labelled edge per value.
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

    /// Writes [`Mdd::as_graphviz`] to the file `filename`. Panics if it cannot be written.
    pub fn to_file(&self, filename: &str) {
        fs::write(filename, self.as_graphviz()).unwrap();
    }

    /// Logs, at `info` level, which constraint types take the most memory.
    pub fn show_memory_footprint(&self) {
        log::info!(
            "Memory report for mdd with {} nodes",
            self.nodes.iter().map(|layer| layer.len()).sum::<usize>()
        );
        let report = MemoryReport::build(self.problem.constraints().iter());
        report.print(80);
    }
}

/// The text dump of the diagram: `UNSAT`, or the variable order, the numbers of nodes and edges,
/// the layer of each node and the edges as `source target value`.
impl std::fmt::Debug for Mdd {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_unsat() {
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
