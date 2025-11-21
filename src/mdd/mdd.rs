use crate::core::*;
use crate::core::problem::Problem;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct NodeIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EdgeIndex(pub usize);


#[derive(Default)]
pub struct Mdd {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    root: NodeIndex,
}

pub struct Node {
    variable: VariableIndex,
    first_edge: Option<EdgeIndex>,
}

impl Node {

    pub fn new(variable: VariableIndex) -> Self {
        Self {
            variable,
            first_edge: None,
        }
    }
}

pub struct Edge {
    from: NodeIndex,
    to: NodeIndex,
    value: ValueIndex,
    next: Option<EdgeIndex>,
}

impl Edge {
}

impl Mdd {

    pub fn new(problem: &Problem) -> Self {
        let mut nodes: Vec<Node> = (0..problem.number_variables()).map(VariableIndex).map(|v| Node::new(v)).collect();
        let mut edges = vec![];
        for i in 0..nodes.len() - 1 {
            let from = NodeIndex(i);
            let to = NodeIndex(i+1);
            let variable = VariableIndex(i);
            for value in (0..problem[variable].domain_size()).map(ValueIndex) {
                let next = nodes[from.0].first_edge;
                let edge = Edge { from, to, value, next };
                nodes[from.0].first_edge = Some(EdgeIndex(edges.len()));
                edges.push(edge);
            }
        }
        let root = NodeIndex(0);
        Mdd {
            nodes,
            edges,
            root,
        }
    }

    pub fn refine(&mut self, problem: &Problem) {
    }
}

impl Mdd {
    
    pub fn as_graphviz(&self) ->  String {
        let mut out = String::new();
        out.push_str("digraph {\ntranksep = 3;\n\n");

        for node in (0..self.nodes.len()).map(NodeIndex) {
            let id = node.0;
            let variable = self[node].variable.0;
            out.push_str(&format!("\t{id} [label=\"{variable}\"];\n"));
        }

        for edge in (0..self.edges.len()).map(EdgeIndex) {
            let from = self[edge].from.0;
            let to = self[edge].to.0;
            out.push_str(&format!("\t{from} -> {to} [penwidth=1];\n"));
        }
        out.push_str("}");
        out
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

impl std::ops::Index<EdgeIndex> for Mdd {
    type Output = Edge;

    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index.0]
    }
}
