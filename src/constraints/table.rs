//! `Table`: the scope takes one of the listed tuples.
//!
//! # Meaning
//!
//! Given a list of tuples $T \subseteq \mathbb{Z}^k$ (the *table*), constrains $x_1, \dots, x_k$
//! so that $(x_1, \dots, x_k) \in T$. This is the classical (positive, extensional) table
//! constraint: the tuples are the supports, anything else is forbidden. Duplicate tuples are
//! stored once.
//!
//! The tuples are given in the order the scope was **declared**: column $j$ holds the value of
//! the $j$-th variable of the scope. The compiler may branch the variables in another order; the
//! constraint reorders its own columns (see [`update_variable_ordering`](Constraint::update_variable_ordering)).
//!
//! Tuples are not checked against the domains: a tuple with a value that is not in the domain of
//! its variable is a support that no assignment can use, so it is harmless. An empty table is
//! unsatisfiable. With an empty scope the only possible tuple is the empty one, so the constraint
//! holds exactly when the table is not empty; as for [`Regular`], the MDD has no layer at which to
//! filter anything, so an empty `Table` is not seen by the compilation.
//!
//! A variable may appear only once in the scope, and every tuple must have one value per
//! variable: [`Table::new`] panics otherwise.
//!
//! # How it compiles
//!
//! When the compiler gives its variable order, the constraint builds, once, the *reduced MDD of
//! the table* for that order: a layered, deterministic diagram whose root-to-sink paths are
//! exactly the tuples, and in which two nodes that accept the same suffixes are one. It is the
//! smallest such diagram for the order (the table is a set of disjoint paths, which reduction
//! folds on the shared prefixes and suffixes), and every node lies on at least one tuple. The
//! nodes of the table MDD are numbered globally, layer after layer, so a node identifier tells its
//! layer, and are shared by all the properties through an [`Arc`].
//!
//! A node of the problem's MDD carries a set of nodes of the table MDD.
//!
//! - **Top-down**, the set $T$ of the nodes of the table MDD that the scope variables of some
//!   path from the root to the node can reach. The root holds the root of the table MDD.
//! - **Bottom-up**, the set $B$ of the nodes of the table MDD from which the scope variables of
//!   some path from the node to the sink can reach the sink of the table MDD. The sink holds the
//!   sink of the table MDD.
//!
//! A node with no path holds the empty set, so there is no sentinel to recognise.
//!
//! ## Property update
//!
//! On an edge of the scope with value $v$, the top-down set of the child gets
//! $\{\delta(u, v) \mid u \in T_{parent}\}$, where $\delta$ is the transition of the table MDD
//! (nothing if there is no edge labelled $v$), and the bottom-up set of the parent gets the
//! nodes $u$ with $\delta(u, v) \in B_{child}$. Out of scope layers copy the set. When a node has
//! several parents (or children) the results are united.
//!
//! ## Node merging
//!
//! Merging two nodes takes the union of their sets. The merged node accepts the paths of both, and
//! also the paths that mix a prefix of one with a suffix of the other when their sets overlap, so
//! it over-approximates them.
//!
//! ## Edge filtering
//!
//! An edge with value $v$ from a parent $T$ to a child $B$ is removed when no node $u \in T$ has
//! $\delta(u, v) \in B$. Since the table MDD is deterministic, a prefix that reaches $u$ and a
//! suffix that leaves $\delta(u, v)$ can always be glued through the edge: with exact sets, which
//! is the case in an MDD whose nodes are not merged, an edge is kept if and only if it lies on a
//! tuple, so nothing but solutions is left once the compilation is exact. With merged nodes the
//! test is sound but incomplete. The bottom-up sets are computed before the edges are filtered,
//! so one pass can leave sets slightly larger than the fixpoint; the next pass removes what is
//! left.
//!
//! ## Single state
//!
//! A node is a single state when its set holds at most one node of the table MDD: every path
//! reaches the same node, that is the same set of completions, or none leads anywhere and the node
//! is dead. Because the table MDD is deterministic, a single path always gives such a set from the
//! top (the bottom-up set of a single path may hold several nodes, which only the top-down side
//! of the exactness test reads).
//!
//! # Design notes
//!
//! - **The reduced MDD is the property's alphabet.** The sets are of node identifiers, kept
//!   sorted, so a union is a merge of two lists and membership a binary search. No other
//!   structure is needed on the path from a node to its tuples. The MDD is built in
//!   $O(k\,|T| \log |T|)$ from the tuples sorted in the compiler's order, and shared through an
//!   [`Arc`], so the properties of all the nodes point to the same copy.
//! - **The order is static.** The table MDD depends on the order of the layers, which is fixed
//!   before the constraint is used; asking for a new order rebuilds it. The size of the
//!   reduced MDD depends on the order, and finding the best one is NP-hard, so the order of the
//!   compiler is taken as given.
//! - **The structural key holds the tuples**, sorted and without duplicates, in the order of the
//!   layers of the scope once the order is known, and in the declared order before. A table is
//!   not symmetric in its scope, so the arena compiles it with its columns as layers, and a scope
//!   given in another order shares the structure of the same table. The shared templates assume
//!   that the value at index $j$ of a domain is the value $j$, which holds for domains $0..n$.
//! - **Negative tables are not supported**, nor compressed (starred) tuples.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, Table};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2]);
//! let table = Table::new(vars, vec![vec![0, 1, 2], vec![1, 1, 0], vec![2, 0, 2]]);
//!
//! assert!(table.is_satisfied(&[1, 1, 0]));
//! assert!(!table.is_satisfied(&[1, 1, 1]));
//! ```
//!
//! # References
//!
//! - K. C. K. Cheng and R. H. C. Yap. *An MDD-based generalized arc consistency algorithm for
//!   positive and negative table constraints and some global constraints.* Constraints 15(2),
//!   2010. The construction of the MDD of a table (mddc, mddify, mddReduce) and the fact that a
//!   fixed order is enough.
//! - G. Perez and J.-C. Régin. *Efficient operations on MDDs for building constraint programming
//!   models.* IJCAI 2015, pp. 374-380. The bottom-up reduction of a trie of tuples (pReduce), and
//!   the view of a table as an uncompressed MDD.
use super::*;
use crate::modelling::VariableIndex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

/// The reduced MDD of the tuples of a table, for a fixed order of the columns.
///
/// Nodes are numbered from 0 layer after layer, the root is 0 and the sink is the last. Inside a
/// layer, nodes follow the order in which the lexicographically sorted tuples first reach them.
/// Every node is on a path from the root to the sink, except in an empty table, where the root
/// and the sink are alone.
#[derive(Clone, deepsize::DeepSizeOf)]
struct TableMdd {
    /// First node of every layer, plus one entry that is the number of nodes. Layer `l` is the
    /// set of the nodes after reading `l` columns, so there are `arity + 1` layers.
    level_start: Vec<u32>,
    /// `children[child_range[u]..child_range[u + 1]]` are the edges out of `u`, as
    /// `(value, child)`, sorted by value.
    child_range: Vec<u32>,
    /// The edges out of every node, see `child_range`.
    children: Vec<(isize, u32)>,
    /// `parents[parent_range[u]..parent_range[u + 1]]` are the edges into `u`, as
    /// `(value, parent)`, sorted by value then parent.
    parent_range: Vec<u32>,
    /// The edges into every node, see `parent_range`.
    parents: Vec<(isize, u32)>,
}

impl TableMdd {
    /// Builds the reduced MDD of `tuples`, which must be sorted lexicographically, without
    /// duplicates and of length `arity`.
    fn build(tuples: &[Vec<isize>], arity: usize) -> Self {
        /// The nodes created so far, layer by layer, with the edges of every node as
        /// `(value, child)` where the child is its index in the next layer.
        struct Builder {
            /// The edges of every node of every layer.
            layers: Vec<Vec<Vec<(isize, u32)>>>,
            /// For every layer, the index of the nodes already created by their edges.
            interned: Vec<FxHashMap<Vec<(isize, u32)>, u32>>,
            /// Number of columns.
            arity: usize,
        }

        impl Builder {
            /// Builds the node reached after reading `level` columns, from the tuples that share
            /// the values read so far. Returns its index in its layer.
            fn node(&mut self, tuples: &[Vec<isize>], level: usize) -> u32 {
                if level == self.arity {
                    return 0;
                }
                let mut edges: Vec<(isize, u32)> = vec![];
                let mut start = 0;
                while start < tuples.len() {
                    let value = tuples[start][level];
                    let mut end = start + 1;
                    while end < tuples.len() && tuples[end][level] == value {
                        end += 1;
                    }
                    let child = self.node(&tuples[start..end], level + 1);
                    edges.push((value, child));
                    start = end;
                }
                if let Some(&id) = self.interned[level].get(&edges) {
                    return id;
                }
                let id = self.layers[level].len() as u32;
                self.layers[level].push(edges.clone());
                self.interned[level].insert(edges, id);
                id
            }
        }

        let mut builder = Builder {
            layers: vec![vec![]; arity + 1],
            interned: vec![FxHashMap::default(); arity + 1],
            arity,
        };
        // The sink, and the root of an empty table.
        builder.layers[arity].push(vec![]);
        if arity > 0 {
            if tuples.is_empty() {
                builder.layers[0].push(vec![]);
            } else {
                builder.node(tuples, 0);
            }
        }

        let mut level_start = vec![0u32];
        for layer in &builder.layers {
            level_start.push(level_start.last().unwrap() + layer.len() as u32);
        }
        let number_nodes = *level_start.last().unwrap() as usize;
        let mut child_range = Vec::with_capacity(number_nodes + 1);
        let mut children = vec![];
        let mut incoming: Vec<(u32, isize, u32)> = vec![];
        for (level, layer) in builder.layers.iter().enumerate() {
            for (index, edges) in layer.iter().enumerate() {
                child_range.push(children.len() as u32);
                let node = level_start[level] + index as u32;
                for &(value, child) in edges {
                    let child = level_start[level + 1] + child;
                    children.push((value, child));
                    incoming.push((child, value, node));
                }
            }
        }
        child_range.push(children.len() as u32);
        incoming.sort_unstable();
        let mut parent_range = vec![0u32; number_nodes + 1];
        for &(child, _, _) in &incoming {
            parent_range[child as usize + 1] += 1;
        }
        for node in 0..number_nodes {
            parent_range[node + 1] += parent_range[node];
        }
        let parents = incoming
            .iter()
            .map(|&(_, value, parent)| (value, parent))
            .collect();
        Self {
            level_start,
            child_range,
            children,
            parent_range,
            parents,
        }
    }

    /// The root.
    fn root(&self) -> u32 {
        0
    }

    /// The sink.
    fn sink(&self) -> u32 {
        *self.level_start.last().unwrap() - 1
    }

    /// The node reached from `node` by `value`, if there is an edge.
    fn child(&self, node: u32, value: isize) -> Option<u32> {
        let edges = &self.children[self.child_range[node as usize] as usize
            ..self.child_range[node as usize + 1] as usize];
        let at = edges.partition_point(|&(v, _)| v < value);
        edges.get(at).filter(|&&(v, _)| v == value).map(|&(_, c)| c)
    }

    /// The edges `(value, parent)` into `node` that are labelled `value`.
    fn parents_with(&self, node: u32, value: isize) -> &[(isize, u32)] {
        let edges = &self.parents[self.parent_range[node as usize] as usize
            ..self.parent_range[node as usize + 1] as usize];
        let from = edges.partition_point(|&(v, _)| v < value);
        let to = edges.partition_point(|&(v, _)| v <= value);
        &edges[from..to]
    }
}

/// Replaces the sorted list `into` by its union with the sorted list `other`.
fn union_into(into: &mut Vec<u32>, other: &[u32]) {
    if other.is_empty() {
        return;
    }
    let mut result = Vec::with_capacity(into.len() + other.len());
    let (mut i, mut j) = (0, 0);
    while i < into.len() && j < other.len() {
        match into[i].cmp(&other[j]) {
            std::cmp::Ordering::Less => {
                result.push(into[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(other[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(into[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&into[i..]);
    result.extend_from_slice(&other[j..]);
    *into = result;
}

/// Per-node state of [`Table`]: a set of nodes of the table MDD. Top-down, the nodes reachable
/// by the scope variables of some path from the root; bottom-up, the nodes from which the sink is
/// reachable by the scope variables of some path to the sink. The empty set means that no path
/// has been folded in, which is the identity of `merge`. See the [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
struct TableProperty {
    /// The table MDD. Shared with the constraint.
    mdd: Arc<TableMdd>,
    /// The nodes of the table MDD, sorted, without duplicates.
    nodes: Vec<u32>,
}

impl TableProperty {
    /// The set of `nodes`, which must be sorted and without duplicates.
    fn new(mdd: Arc<TableMdd>, nodes: Vec<u32>) -> Self {
        Self { mdd, nodes }
    }

    /// Absorbs the sorted, deduplicated list `reached`, built from unsorted ids.
    fn absorb(&mut self, mut reached: Vec<u32>) {
        reached.sort_unstable();
        reached.dedup();
        union_into(&mut self.nodes, &reached);
    }

    /// Downcasts `other`, panicking with the name of the operation if it is another kind.
    fn cast<'a>(&self, other: &'a dyn ConstraintProperty, operation: &str) -> &'a TableProperty {
        other
            .as_any()
            .downcast_ref::<TableProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling {operation} on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            })
    }
}

/// The constraint that the scope takes the value of one of the listed tuples. See the
/// [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Table {
    /// Scope of the constraint, without repetition, in the order of the columns of the tuples.
    variables: Vec<VariableIndex>,
    /// The tuples, sorted and without duplicates, in the order of `variables`.
    tuples: Arc<Vec<Box<[isize]>>>,
    /// For the k-th layer of the scope, the column of the tuples that it reads. Empty until
    /// `update_variable_ordering` is called.
    columns: Vec<usize>,
    /// The reduced MDD of the tuples for the order of the compiler. `None` until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    mdd: Option<Arc<TableMdd>>,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called.
    layer_in_scope: Vec<u64>,
}

impl Table {
    /// Builds the constraint that `variables` take the values of one of the `tuples`, column `j`
    /// of a tuple being the value of `variables[j]`. The tuples are sorted and deduplicated.
    ///
    /// # Panics
    ///
    /// - If a variable appears more than once in `variables`.
    /// - If a tuple does not have exactly one value per variable.
    pub fn new(variables: Vec<VariableIndex>, tuples: Vec<Vec<isize>>) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        if distinct.len() != variables.len() {
            panic!("Table does not support a variable repeated in its scope");
        }
        if let Some((index, tuple)) = tuples
            .iter()
            .enumerate()
            .find(|(_, tuple)| tuple.len() != variables.len())
        {
            panic!(
                "Table: tuple {index} has {} values for a scope of {} variables",
                tuple.len(),
                variables.len()
            );
        }
        let mut tuples: Vec<Box<[isize]>> = tuples.into_iter().map(Vec::into_boxed_slice).collect();
        tuples.sort_unstable();
        tuples.dedup();
        Self {
            variables,
            tuples: Arc::new(tuples),
            columns: vec![],
            mdd: None,
            layer_in_scope: vec![],
        }
    }

    /// The table MDD, which exists once the order is known.
    fn mdd(&self) -> Arc<TableMdd> {
        self.mdd
            .as_ref()
            .expect("Table: update_variable_ordering has not been called")
            .clone()
    }
}

impl Constraint for Table {
    /// The tuples in the order of the layers of the scope once the order is known (see
    /// [`update_variable_ordering`](Constraint::update_variable_ordering)), in the declared order
    /// before. The structure of a table depends on the order of its layers, so this is what lets
    /// the shared structures of the arena serve a scope in any order.
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let is_declared_order = self.columns.iter().enumerate().all(|(k, &c)| k == c);
        let tuples = if is_declared_order {
            Arc::clone(&self.tuples)
        } else {
            let mut reordered: Vec<Box<[isize]>> = self
                .tuples
                .iter()
                .map(|tuple| self.columns.iter().map(|&c| tuple[c]).collect())
                .collect();
            reordered.sort_unstable();
            Arc::new(reordered)
        };
        ConstraintShapeKey::Table {
            arity: self.variables.len(),
            tuples,
        }
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let declared: FxHashMap<VariableIndex, usize> = self
            .variables
            .iter()
            .enumerate()
            .map(|(column, &variable)| (variable, column))
            .collect();
        self.layer_in_scope = vec![0; order.len() / 64 + 1];
        // `columns[k]` is the column of the tuples read by the k-th layer of the scope.
        let mut columns: Vec<usize> = Vec::with_capacity(self.variables.len());
        for (layer, variable) in order.iter().enumerate() {
            if let Some(&column) = declared.get(variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                columns.push(column);
            }
        }
        if columns.len() != self.variables.len() {
            panic!("Table: a variable of the scope is not in the variable ordering");
        }
        let mut reordered: Vec<Vec<isize>> = self
            .tuples
            .iter()
            .map(|tuple| columns.iter().map(|&column| tuple[column]).collect())
            .collect();
        reordered.sort_unstable();
        self.mdd = Some(Arc::new(TableMdd::build(&reordered, columns.len())));
        self.columns = columns;
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(
            !self.layer_in_scope.is_empty(),
            "update_variable_ordering has not been called"
        );
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let projected: Vec<isize> = self
            .variables
            .iter()
            .map(|variable| assignment[**variable])
            .collect();
        self.tuples
            .binary_search_by(|tuple| (**tuple).cmp(projected.as_slice()))
            .is_ok()
    }

    fn name(&self) -> &'static str {
        "Table"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        _layer: usize,
        assignment: isize,
    ) -> bool {
        let parent = parent.as_any().downcast_ref::<TableProperty>().unwrap_or_else(|| {
            panic!(
                "Calling is_assignment_invalid on parent property of type {} instead of TableProperty",
                parent.name()
            );
        });
        let child = child.as_any().downcast_ref::<TableProperty>().unwrap_or_else(|| {
            panic!(
                "Calling is_assignment_invalid on child property of type {} instead of TableProperty",
                child.name()
            );
        });
        !parent.nodes.iter().any(|&node| {
            parent
                .mdd
                .child(node, assignment)
                .is_some_and(|next| child.nodes.binary_search(&next).is_ok())
        })
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(TableProperty::new(self.mdd(), vec![]))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        let mdd = self.mdd();
        let root = mdd.root();
        Box::new(TableProperty::new(mdd, vec![root]))
    }

    fn empty_property_backward(&self) -> Box<dyn ConstraintProperty> {
        let mdd = self.mdd();
        let sink = mdd.sink();
        Box::new(TableProperty::new(mdd, vec![sink]))
    }
}

impl ConstraintProperty for TableProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = self.cast(other, "update");
        if in_scope {
            let reached: Vec<u32> = other
                .nodes
                .iter()
                .filter_map(|&node| self.mdd.child(node, assignment))
                .collect();
            self.absorb(reached);
        } else {
            union_into(&mut self.nodes, &other.nodes);
        }
    }

    fn update_backward(
        &mut self,
        other: &dyn ConstraintProperty,
        assignment: isize,
        in_scope: bool,
    ) {
        let other = self.cast(other, "update_backward");
        if in_scope {
            let reached: Vec<u32> = other
                .nodes
                .iter()
                .flat_map(|&node| self.mdd.parents_with(node, assignment))
                .map(|&(_, parent)| parent)
                .collect();
            self.absorb(reached);
        } else {
            union_into(&mut self.nodes, &other.nodes);
        }
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = self.cast(other, "merge");
        union_into(&mut self.nodes, &other.nodes);
    }

    /// The first and the last node of the set, and its size. The nodes of a layer are numbered
    /// in the lexicographic order of the prefixes that reach them, so sets with close ends hold
    /// similar tuples. The empty set has its own coordinates.
    fn order_key(&self) -> Vec<f64> {
        match (self.nodes.first(), self.nodes.last()) {
            (Some(&first), Some(&last)) => vec![first as f64, last as f64, self.nodes.len() as f64],
            _ => vec![-1.0, -1.0, 0.0],
        }
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for &node in &self.nodes {
            hasher.write_u32(node);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        self.nodes == self.cast(other, "eq").nodes
    }

    /// One node of the table MDD, or none: a path that no tuple continues is dead, and dead paths
    /// are interchangeable since nothing can follow them.
    fn is_single_state(&self) -> bool {
        self.nodes.len() <= 1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "TableProperty"
    }
}

#[cfg(test)]
mod test_table {
    use super::*;
    use crate::constraints::{AllDifferent, ConstraintShapeKey, table};
    use crate::mdd::compiler::test_mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHasher;
    use std::sync::Arc;

    // ----------------------------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------------------------

    /// The tiny deterministic generator used for the random instances below.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
    }

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    /// A problem of `n` variables of domain `0..d`.
    fn problem_of(n: usize, d: isize) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(d));
        (problem, vars)
    }

    /// The reduced MDD of `tuples`, which are sorted and deduplicated first.
    fn mdd_of(tuples: &[Vec<isize>], arity: usize) -> TableMdd {
        let mut tuples = tuples.to_vec();
        tuples.sort();
        tuples.dedup();
        TableMdd::build(&tuples, arity)
    }

    /// The number of nodes of every layer of the table MDD.
    fn widths(mdd: &TableMdd) -> Vec<usize> {
        mdd.level_start
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .collect()
    }

    /// The number of edges of the table MDD.
    fn arcs(mdd: &TableMdd) -> usize {
        mdd.children.len()
    }

    /// Every path of the table MDD, from the root to the sink.
    fn paths(mdd: &TableMdd) -> Vec<Vec<isize>> {
        fn walk(mdd: &TableMdd, node: u32, prefix: &mut Vec<isize>, out: &mut Vec<Vec<isize>>) {
            if node == mdd.sink() {
                out.push(prefix.clone());
                return;
            }
            let (from, to) = (
                mdd.child_range[node as usize] as usize,
                mdd.child_range[node as usize + 1] as usize,
            );
            for &(value, child) in &mdd.children[from..to] {
                prefix.push(value);
                walk(mdd, child, prefix, out);
                prefix.pop();
            }
        }
        let mut out = vec![];
        walk(mdd, mdd.root(), &mut vec![], &mut out);
        out
    }

    /// `count` distinct tuples of `arity` values below `values`.
    fn random_tuples(rng: &mut Lcg, arity: usize, values: u64, count: usize) -> Vec<Vec<isize>> {
        let mut tuples: Vec<Vec<isize>> = vec![];
        for _ in 0..count {
            let tuple = (0..arity).map(|_| rng.below(values) as isize).collect();
            if !tuples.contains(&tuple) {
                tuples.push(tuple);
            }
        }
        tuples
    }

    /// All tuples of the cartesian product of the domains, sorted.
    fn product(domains: &[Vec<isize>]) -> Vec<Vec<isize>> {
        let mut tuples: Vec<Vec<isize>> = vec![vec![]];
        for domain in domains {
            let mut next = vec![];
            for prefix in &tuples {
                for &value in domain {
                    let mut tuple = prefix.clone();
                    tuple.push(value);
                    next.push(tuple);
                }
            }
            tuples = next;
        }
        tuples
    }

    fn nodes_of(property: &dyn ConstraintProperty) -> Vec<u32> {
        property
            .as_any()
            .downcast_ref::<TableProperty>()
            .unwrap()
            .nodes
            .clone()
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    /// The compilation itself, before the diagram is taken out of the compiler.
    fn compile_to_compiler(problem: Problem, order: Vec<usize>, width: usize) -> Compiler {
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::Custom(order),
                merge: MergeHeuristic::StateSimilarity,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        if width > 1 {
            compiler.refine(width, usize::MAX);
        }
        compiler
    }

    /// The exact compilation, with the propagation run again until nothing changes (see the
    /// same helper in the tests of `Regular`).
    fn settled(problem: Problem, order: Vec<usize>) -> Mdd {
        let mut compiler = compile_to_compiler(problem, order, usize::MAX);
        for _ in 0..10 {
            if compiler.mdd().is_unsat() {
                break;
            }
            compiler.propagate_constraints();
        }
        compiler.into_mdd()
    }

    fn accepted(mdd: &Mdd) -> Vec<Vec<isize>> {
        if mdd.is_unsat() {
            return vec![];
        }
        let mut solutions = get_all_solutions(mdd);
        solutions.sort();
        solutions
    }

    // ----------------------------------------------------------------------------------------
    // The reduced MDD of a table
    // ----------------------------------------------------------------------------------------

    #[test]
    fn a_table_with_shared_prefixes_and_suffixes_is_folded() {
        // The first two columns say nothing about the last one except for (1, 1).
        let tuples = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 0, 0], vec![1, 1, 1]];
        let mdd = mdd_of(&tuples, 3);
        // root | the node after x0 = 0 and the one after x0 = 1 | two nodes before the sink | sink
        assert_eq!(widths(&mdd), vec![1, 2, 2, 1]);
        assert_eq!(arcs(&mdd), 2 + 2 + 2 + 1 + 1);
    }

    #[test]
    fn a_full_product_is_a_chain_of_single_nodes() {
        let tuples = product(&[dom(3), dom(3), dom(3)]);
        let mdd = mdd_of(&tuples, 3);
        assert_eq!(widths(&mdd), vec![1, 1, 1, 1]);
        assert_eq!(arcs(&mdd), 9);
    }

    #[test]
    fn a_diagonal_table_has_one_node_per_value_and_layer() {
        let tuples: Vec<Vec<isize>> = (0..4).map(|v| vec![v; 3]).collect();
        let mdd = mdd_of(&tuples, 3);
        assert_eq!(widths(&mdd), vec![1, 4, 4, 1]);
    }

    #[test]
    fn the_paths_of_the_table_mdd_are_the_tuples() {
        let mut rng = Lcg(21);
        for _ in 0..60 {
            let arity = 1 + rng.below(5) as usize;
            let count = rng.below(30) as usize;
            let mut tuples = random_tuples(&mut rng, arity, 3, count);
            // Duplicates are folded by `Table::new`, not by the builder.
            tuples.sort();
            let mdd = mdd_of(&tuples, arity);
            assert_eq!(paths(&mdd), tuples);
        }
    }

    #[test]
    fn the_table_mdd_is_trimmed_reduced_and_consistent() {
        let mut rng = Lcg(22);
        for _ in 0..60 {
            let arity = 1 + rng.below(5) as usize;
            let count = 1 + rng.below(30) as usize;
            let tuples = random_tuples(&mut rng, arity, 3, count);
            let mdd = mdd_of(&tuples, arity);
            let nodes = *mdd.level_start.last().unwrap() as usize;
            for node in 0..nodes as u32 {
                let (from, to) = (
                    mdd.child_range[node as usize] as usize,
                    mdd.child_range[node as usize + 1] as usize,
                );
                // Every node but the sink has a way out, and every node but the root a way in.
                assert_eq!(from == to, node == mdd.sink(), "node {node}");
                let (pfrom, pto) = (
                    mdd.parent_range[node as usize] as usize,
                    mdd.parent_range[node as usize + 1] as usize,
                );
                assert_eq!(pfrom == pto, node == mdd.root(), "node {node}");
                // Both views of the edges agree.
                for &(value, child) in &mdd.children[from..to] {
                    assert_eq!(mdd.child(node, value), Some(child));
                    assert!(
                        mdd.parents_with(child, value)
                            .iter()
                            .any(|&(_, parent)| parent == node)
                    );
                }
                for &(value, parent) in &mdd.parents[pfrom..pto] {
                    assert_eq!(mdd.child(parent, value), Some(node));
                }
            }
            assert_eq!(arcs(&mdd), mdd.parents.len());
            // Reduced: two nodes of a layer never have the same edges.
            for layer in mdd.level_start.windows(2) {
                let mut edges: Vec<&[(isize, u32)]> = (layer[0]..layer[1])
                    .map(|n| {
                        &mdd.children[mdd.child_range[n as usize] as usize
                            ..mdd.child_range[n as usize + 1] as usize]
                    })
                    .collect();
                let before = edges.len();
                edges.sort();
                edges.dedup();
                assert_eq!(edges.len(), before);
            }
        }
    }

    #[test]
    fn a_missing_edge_has_no_child_and_no_parent() {
        let mdd = mdd_of(&[vec![0, 1], vec![2, 1]], 2);
        assert_eq!(mdd.child(mdd.root(), 1), None);
        assert_eq!(mdd.child(mdd.root(), -5), None);
        let after = mdd.child(mdd.root(), 0).unwrap();
        assert!(mdd.parents_with(after, 1).is_empty());
        assert_eq!(mdd.parents_with(after, 0).len(), 1);
    }

    #[test]
    fn an_empty_table_has_a_root_and_a_sink_and_no_edge() {
        let mdd = mdd_of(&[], 3);
        assert_eq!(widths(&mdd), vec![1, 0, 0, 1]);
        assert_eq!(arcs(&mdd), 0);
        assert!(paths(&mdd).is_empty());
        assert_ne!(mdd.root(), mdd.sink());
    }

    #[test]
    fn a_table_without_columns_is_a_single_node() {
        let mdd = mdd_of(&[vec![]], 0);
        assert_eq!(widths(&mdd), vec![1]);
        assert_eq!(mdd.root(), mdd.sink());
        assert_eq!(paths(&mdd), vec![Vec::<isize>::new()]);
    }

    // ----------------------------------------------------------------------------------------
    // The constraint
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_tuples_are_sorted_and_deduplicated() {
        let (_, vars) = problem_of(2, 3);
        let table = Table::new(vars, vec![vec![2, 1], vec![0, 0], vec![2, 1], vec![0, 2]]);
        assert_eq!(table.tuples.len(), 3);
        assert!(table.tuples.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn is_satisfied_reads_the_scope_in_its_declared_order() {
        let (_, vars) = problem_of(4, 3);
        // x3 then x1: the tuple (2, 0) means x3 = 2 and x1 = 0.
        let table = Table::new(vec![vars[3], vars[1]], vec![vec![2, 0], vec![1, 1]]);
        assert!(table.is_satisfied(&[9, 0, 9, 2]));
        assert!(table.is_satisfied(&[0, 1, 0, 1]));
        assert!(!table.is_satisfied(&[0, 2, 0, 0]));
        assert!(!table.is_satisfied(&[2, 0, 0, 0]));
    }

    #[test]
    fn negative_values_are_ordinary_values() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![-2, -1, 0, 3]);
        let mut table = Table::new(vars.clone(), vec![vec![-2, 3], vec![-1, -1], vec![0, 0]]);
        table.update_variable_ordering(&vars);
        assert!(table.is_satisfied(&[-2, 3]));
        assert!(!table.is_satisfied(&[3, -2]));
        // `-1` after `-1` is accepted, and the order of the sorted tuples does not matter.
        let top = table.empty_property();
        let mut one = table.identity_property();
        one.update(&*top, -1, true);
        let mut end = table.identity_property();
        end.update(&*one, -1, true);
        assert_eq!(nodes_of(&*end), vec![table.mdd().sink()]);
    }

    #[test]
    fn an_empty_table_is_never_satisfied() {
        let (_, vars) = problem_of(2, 3);
        let table = Table::new(vars, vec![]);
        for t in product(&[dom(3), dom(3)]) {
            assert!(!table.is_satisfied(&t));
        }
    }

    #[test]
    fn with_an_empty_scope_the_table_holds_when_it_has_the_empty_tuple() {
        assert!(Table::new(vec![], vec![vec![]]).is_satisfied(&[]));
        assert!(!Table::new(vec![], vec![]).is_satisfied(&[]));
    }

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_panics() {
        let (_, vars) = problem_of(2, 3);
        Table::new(vec![vars[0], vars[1], vars[0]], vec![]);
    }

    #[test]
    #[should_panic(expected = "tuple 1 has 2 values for a scope of 3 variables")]
    fn a_tuple_of_the_wrong_length_panics() {
        let (_, vars) = problem_of(3, 3);
        Table::new(vars, vec![vec![0, 0, 0], vec![0, 0]]);
    }

    #[test]
    #[should_panic(expected = "not in the variable ordering")]
    fn an_ordering_without_a_variable_of_the_scope_panics() {
        let (_, vars) = problem_of(3, 3);
        let mut table = Table::new(vars.clone(), vec![vec![0, 0, 0]]);
        table.update_variable_ordering(&vars[..2]);
    }

    #[test]
    #[should_panic(expected = "update_variable_ordering has not been called")]
    fn a_property_needs_the_ordering() {
        let (_, vars) = problem_of(2, 3);
        Table::new(vars, vec![vec![0, 0]]).identity_property();
    }

    #[test]
    fn a_new_ordering_rebuilds_the_table_mdd() {
        let (_, vars) = problem_of(3, 3);
        // The first two values are equal and the last is 0: the diagram is wide where the two
        // equal values are read, and narrow before and after.
        let tuples: Vec<Vec<isize>> = (0..3).map(|a| vec![a, a, 0]).collect();
        let mut table = Table::new(vars.clone(), tuples);
        table.update_variable_ordering(&vars);
        assert_eq!(widths(&table.mdd()), vec![1, 3, 1, 1]);
        table.update_variable_ordering(&[vars[2], vars[1], vars[0]]);
        assert_eq!(widths(&table.mdd()), vec![1, 1, 3, 1]);
        assert_eq!(table.columns, vec![2, 1, 0]);
    }

    #[test]
    fn the_key_lists_the_tuples_in_the_order_of_the_layers() {
        let (problem, vars) = problem_of(3, 3);
        let tuples = vec![vec![0, 1, 2], vec![2, 1, 0], vec![1, 0, 0]];
        let mut table = Table::new(vars.clone(), tuples);
        let declared = table.structural_key(&problem);
        table.update_variable_ordering(&vars);
        assert_eq!(table.structural_key(&problem), declared);
        table.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        let ConstraintShapeKey::Table { arity, tuples } = table.structural_key(&problem) else {
            panic!("not a table key");
        };
        assert_eq!(arity, 3);
        let expected: Vec<Box<[isize]>> = vec![
            vec![0, 1, 0].into(),
            vec![0, 2, 1].into(),
            vec![2, 0, 1].into(),
        ];
        assert_eq!(*tuples, expected);
    }

    #[test]
    fn a_scope_in_another_order_has_the_same_key_once_ordered() {
        let (problem, vars) = problem_of(2, 3);
        let mut a = Table::new(vec![vars[0], vars[1]], vec![vec![0, 1], vec![1, 2]]);
        let mut b = Table::new(vec![vars[1], vars[0]], vec![vec![1, 0], vec![2, 1]]);
        let order = [vars[0], vars[1]];
        a.update_variable_ordering(&order);
        b.update_variable_ordering(&order);
        assert_eq!(a.structural_key(&problem), b.structural_key(&problem));
    }

    #[test]
    fn a_table_does_not_share_a_key_with_a_different_table() {
        let (problem, vars) = problem_of(2, 3);
        let a = Table::new(vars.clone(), vec![vec![0, 1]]);
        let b = Table::new(vars, vec![vec![0, 2]]);
        assert_ne!(a.structural_key(&problem), b.structural_key(&problem));
    }

    // ----------------------------------------------------------------------------------------
    // The properties
    // ----------------------------------------------------------------------------------------

    /// The five-tuple table used by the tests of the properties, over three variables.
    fn small() -> Table {
        let (_, vars) = problem_of(3, 3);
        let mut table = Table::new(
            vars.clone(),
            vec![
                vec![0, 0, 1],
                vec![0, 1, 1],
                vec![1, 0, 1],
                vec![1, 1, 2],
                vec![2, 2, 0],
            ],
        );
        table.update_variable_ordering(&vars);
        table
    }

    #[test]
    fn the_root_holds_the_root_and_the_sink_holds_the_sink() {
        let table = small();
        let mdd = table.mdd();
        assert_eq!(nodes_of(&*table.empty_property()), vec![mdd.root()]);
        assert_eq!(
            nodes_of(&*table.empty_property_backward()),
            vec![mdd.sink()]
        );
        assert!(nodes_of(&*table.identity_property()).is_empty());
    }

    #[test]
    fn folding_a_prefix_walks_the_table_mdd() {
        let table = small();
        let mdd = table.mdd();
        let mut property = table.empty_property();
        for &value in &[1, 1] {
            let mut next = table.identity_property();
            next.update(&*property, value, true);
            property = next;
        }
        // (1, 1) can only be followed by 2.
        let nodes = nodes_of(&*property);
        assert_eq!(nodes.len(), 1);
        assert_eq!(mdd.child(nodes[0], 2), Some(mdd.sink()));
        assert_eq!(mdd.child(nodes[0], 1), None);
        // (1, 2) is not a prefix of a tuple.
        let mut dead = table.identity_property();
        dead.update(&*table.empty_property(), 1, true);
        let mut dead2 = table.identity_property();
        dead2.update(&*dead, 2, true);
        assert!(nodes_of(&*dead2).is_empty());
    }

    #[test]
    fn folding_a_suffix_walks_the_table_mdd_backwards() {
        let table = small();
        let mdd = table.mdd();
        // The suffix (1) leaves the nodes reached by (0, *) and (1, 0): they are one node, the
        // one that holds the edge 1 to the sink; (2, 2) does not leave through 1.
        let mut property = table.identity_property();
        property.update_backward(&*table.empty_property_backward(), 1, true);
        let nodes = nodes_of(&*property);
        assert_eq!(nodes.len(), 1);
        assert_eq!(mdd.child(nodes[0], 1), Some(mdd.sink()));
        // (0, 1): the first value that leads to that node.
        let mut up = table.identity_property();
        up.update_backward(&*property, 1, true);
        let first = nodes_of(&*up);
        assert!(first.iter().all(|&n| mdd.child(n, 1) == Some(nodes[0])));
    }

    #[test]
    fn out_of_scope_layers_copy_the_set() {
        let table = small();
        let mut property = table.identity_property();
        property.update(&*table.empty_property(), 7, false);
        assert!(same(&*property, &*table.empty_property()));
        let mut property = table.identity_property();
        property.update_backward(&*table.empty_property_backward(), 7, false);
        assert!(same(&*property, &*table.empty_property_backward()));
    }

    #[test]
    fn merging_unites_the_sets_and_keeps_them_sorted() {
        let table = small();
        let a = TableProperty::new(table.mdd(), vec![1, 4]);
        let b = TableProperty::new(table.mdd(), vec![0, 4, 5]);
        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(merged.nodes, vec![0, 1, 4, 5]);
        assert!(!merged.is_single_state());
        let mut merged_again = merged.clone();
        merged_again.merge(&a);
        assert!(same(&merged, &merged_again));
    }

    #[test]
    fn the_single_state_is_at_most_one_node() {
        let table = small();
        assert!(table.identity_property().is_single_state());
        assert!(table.empty_property().is_single_state());
        assert!(TableProperty::new(table.mdd(), vec![3]).is_single_state());
        assert!(!TableProperty::new(table.mdd(), vec![3, 4]).is_single_state());
    }

    #[test]
    fn equal_sets_are_equal_and_hash_alike() {
        let table = small();
        let a = TableProperty::new(table.mdd(), vec![1, 3]);
        let b = TableProperty::new(table.mdd(), vec![1, 3]);
        let c = TableProperty::new(table.mdd(), vec![1, 4]);
        assert!(same(&a, &b));
        assert_eq!(hash_of(&a), hash_of(&b));
        assert!(!same(&a, &c));
        assert!(!same(&a, &*table.identity_property()));
    }

    #[test]
    fn the_order_key_has_three_finite_numbers() {
        let table = small();
        assert_eq!(table.identity_property().order_key(), vec![-1.0, -1.0, 0.0]);
        assert_eq!(
            TableProperty::new(table.mdd(), vec![2, 3, 6]).order_key(),
            vec![2.0, 6.0, 3.0]
        );
    }

    #[test]
    #[should_panic(expected = "instead of TableProperty")]
    fn a_property_of_another_constraint_is_refused() {
        let table = small();
        let (problem, vars) = problem_of(2, 3);
        let foreign = AllDifferent::new(vars, &problem).identity_property();
        table.is_assignment_invalid(&*table.empty_property(), &*foreign, 0, 0);
    }

    #[test]
    fn an_edge_is_removed_only_when_no_tuple_goes_through_it() {
        let table = small();
        let mdd = table.mdd();
        // After x0 = 1 the node holds the edges 0, to the node of the tuples ending in 1, and 1,
        // to the node of the tuple ending in 2. The child is the second one.
        let after_one = mdd.child(mdd.root(), 1).unwrap();
        let parent = TableProperty::new(table.mdd(), vec![after_one]);
        let child = TableProperty::new(table.mdd(), vec![mdd.child(after_one, 1).unwrap()]);
        assert!(!table.is_assignment_invalid(&parent, &child, 1, 1));
        assert!(table.is_assignment_invalid(&parent, &child, 1, 0));
        assert!(table.is_assignment_invalid(&parent, &child, 1, 2));
        // A dead parent or a dead child removes every edge.
        let dead = TableProperty::new(table.mdd(), vec![]);
        for value in 0..3 {
            assert!(table.is_assignment_invalid(&dead, &child, 1, value));
            assert!(table.is_assignment_invalid(&parent, &dead, 1, value));
        }
    }

    #[test]
    fn the_exact_sets_of_a_single_tuple_give_the_same_edges_as_the_tuples() {
        // For every prefix and suffix of the same tuple and every value, an edge is kept iff the
        // three parts form a tuple. This is what makes the filtering of an exact MDD complete.
        let table = small();
        let tuples: Vec<Vec<isize>> = table.tuples.iter().map(|t| t.to_vec()).collect();
        for left in &tuples {
            for right in &tuples {
                for layer in 0..3 {
                    let mut parent = table.empty_property();
                    for &value in &left[..layer] {
                        let mut next = table.identity_property();
                        next.update(&*parent, value, true);
                        parent = next;
                    }
                    let mut child = table.empty_property_backward();
                    for &value in right[layer + 1..].iter().rev() {
                        let mut next = table.identity_property();
                        next.update_backward(&*child, value, true);
                        child = next;
                    }
                    for value in 0..3 {
                        let mut glued = left[..layer].to_vec();
                        glued.push(value);
                        glued.extend_from_slice(&right[layer + 1..]);
                        assert_eq!(
                            table.is_assignment_invalid(&*parent, &*child, layer, value),
                            !table.is_satisfied(&glued),
                            "{left:?} | {value} | {right:?} at layer {layer}"
                        );
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Compilation
    // ----------------------------------------------------------------------------------------

    /// The solutions that a table over the variables `scope` of `domains` leaves, by brute force.
    fn expected(domains: &[Vec<isize>], scope: &[usize], tuples: &[Vec<isize>]) -> Vec<Vec<isize>> {
        let mut all = product(domains);
        all.retain(|a| {
            let projected: Vec<isize> = scope.iter().map(|&i| a[i]).collect();
            tuples.contains(&projected)
        });
        all.sort();
        all
    }

    #[test]
    fn an_exact_compilation_is_the_table() {
        let mut rng = Lcg(31);
        for round in 0..30 {
            let n = 2 + rng.below(4) as usize;
            let domains: Vec<Vec<isize>> = (0..n).map(|_| dom(2 + rng.below(2) as isize)).collect();
            // The variables in a random order: the order in which the tuples read them.
            let mut scope: Vec<usize> = (0..n).collect();
            for i in (1..scope.len()).rev() {
                scope.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let count = rng.below(12) as usize;
            let tuples = random_tuples(&mut rng, scope.len(), 3, count);
            // A random order of the layers.
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }

            let mut problem = Problem::default();
            let vars: Vec<VariableIndex> = domains
                .iter()
                .map(|d| problem.add_variable(d.clone()))
                .collect();
            table(
                &mut problem,
                scope.iter().map(|&i| vars[i]).collect(),
                tuples.clone(),
            );
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &scope, &tuples),
                "round {round}: scope {scope:?} order {order:?} tuples {tuples:?}"
            );
        }
    }

    #[test]
    fn the_exact_diagram_of_a_table_over_all_variables_has_the_widths_of_the_table_mdd() {
        let mut rng = Lcg(32);
        for _ in 0..20 {
            let n = 3 + rng.below(3) as usize;
            let count = 3 + rng.below(15) as usize;
            let tuples = random_tuples(&mut rng, n, 3, count);
            let (mut problem, vars) = problem_of(n, 3);
            table(&mut problem, vars.clone(), tuples.clone());
            let mdd = settled(problem, (0..n).collect());
            let reference = mdd_of(&tuples, n);
            let layers: Vec<usize> = (0..=n).map(|l| mdd.number_nodes_in_layer(l)).collect();
            assert_eq!(layers, widths(&reference), "tuples {tuples:?}");
            assert_eq!(
                mdd.count_solutions(),
                num_bigint::BigUint::from(tuples.len())
            );
        }
    }

    #[test]
    fn an_empty_table_makes_the_problem_unsatisfiable() {
        let (mut problem, vars) = problem_of(3, 3);
        table(&mut problem, vars, vec![]);
        assert!(settled(problem, vec![0, 1, 2]).is_unsat());
    }

    #[test]
    fn a_table_over_no_tuple_of_the_domain_makes_the_problem_unsatisfiable() {
        let (mut problem, vars) = problem_of(2, 3);
        table(&mut problem, vars, vec![vec![5, 5], vec![-1, 0]]);
        assert!(settled(problem, vec![0, 1]).is_unsat());
    }

    #[test]
    fn a_relaxed_diagram_keeps_every_tuple_at_every_width() {
        let mut rng = Lcg(33);
        for _ in 0..10 {
            let n = 4;
            let count = 8 + rng.below(20) as usize;
            let tuples = random_tuples(&mut rng, n, 3, count);
            for width in [1, 2, 3, 4, 8] {
                let (mut problem, vars) = problem_of(n, 3);
                table(&mut problem, vars, tuples.clone());
                let mdd = compile_to_compiler(problem, (0..n).collect(), width).into_mdd();
                let solutions = accepted(&mdd);
                for tuple in &tuples {
                    assert!(
                        solutions.contains(tuple),
                        "width {width} lost {tuple:?} of {tuples:?}"
                    );
                }
                assert!(
                    (0..=n).all(|l| mdd.number_nodes_in_layer(l) <= width.max(1)),
                    "width {width}"
                );
            }
        }
    }

    #[test]
    fn a_table_next_to_other_constraints_filters_them_both_ways() {
        // The table forces x0 + x1 = 2 on its own, the all-different forces x1 != x2: the
        // exact diagram holds the tuples of both.
        let (mut problem, vars) = problem_of(3, 3);
        let tuples = vec![vec![0, 2], vec![1, 1], vec![2, 0]];
        table(&mut problem, vec![vars[0], vars[1]], tuples);
        crate::constraints::not_equals(&mut problem, vars[1], vars[2]);
        let mdd = settled(problem, vec![2, 1, 0]);
        let mut expected: Vec<Vec<isize>> = product(&[dom(3), dom(3), dom(3)]);
        expected.retain(|a| a[0] + a[1] == 2 && a[1] != a[2]);
        expected.sort();
        assert_eq!(accepted(&mdd), expected);
    }
}
