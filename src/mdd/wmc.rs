//! Weighted model counting on an MDD: the forward and backward passes, the weighted count of
//! solutions and its gradient.
//!
//! # Purpose
//!
//! The weighted model count of a solution space given a probability distribution is the weighted
//! sum of each solution's probability.
//! Currently, we assume that the probability distribution on the solution space (i.e., an MDD) is
//! factorised as the product of each variable's own probability distribution over its domain.
//! Hence, the weighted model count can be computed in a single forward-pass on the MDD.
//! Similarly, the gradient (when probabilities come from a neural-network) can be computed with,
//! in addition, a single backward pass.
//!
//! # How it works
//!
//! Two dynamic programs run over the MDD structure in $\mathcal{O}(|N| + |E|)$ with $N$ the set of
//! nodes and $E$ the set of edges of the MDD.
//! The weight of a path is the product of the weights of its edges. We define by $\alpha(u)$ the
//! sum of the weights of the paths from the root to $u$ and by $\beta(u)$ the sum of the weights
//! of the paths from $u$ to the sink.
//! It follows that $\alpha(s)$, with $s$ the sink, gives the weighted model count of the MDD,
//! since each path is a full assignment encoded in the MDD.
//! Given that an MDD is a directed acyclic graph, these two values are easily computed by a
//! layer-wise forward/backward pass, starting from $\alpha(\text{root}) = 1$ and
//! $\beta(s) = 1$ (the empty path has weight $1$).
//!
//! Now, let us compute the gradient of the weighted model count with respect to the parameters
//! (i.e., the edges' weights).
//! Let $u$ be a node of layer $\ell$ and $E_u$ its outgoing edges.
//! For an edge $e$, we denote $w_e$ its weight, $\mathrm{from}(e)$ its source node, and
//! $\mathrm{to}(e)$ its destination node. The weight of the paths that leave $u$ can be computed as
//!
//! $$\beta(u) = \sum_{e \in E_u} w_e \times \beta(\mathrm{to}(e)).$$
//!
//! Since every path in the MDD crosses exactly one node in layer $\ell$, then the weighted
//! model count can be computed as
//!
//! $$\mathrm{WMC} = \sum_{u \in \ell} \alpha(u) \times \beta(u) = \sum_{u \in \ell} \sum_{e \in
//! E_u} \alpha(u) \times w_e \times \beta(\mathrm{to}(e)).$$
//!
//! Notice that $\alpha(u)$ only depends on the weights of the layers before $\ell$ and
//! $\beta(\mathrm{to}(e))$ on those after it: neither involves $w_e$. Hence, the gradient of the
//! $\mathrm{WMC}$ w.r.t. the parameter $w_e$ of edge $e \in E_u$ is given by
//!
//! $$ \frac{\partial\ \mathrm{WMC}}{\partial w_e} = \alpha(u) \times \beta(\mathrm{to}(e)).$$
//!
//! More generally, a parameter $w$ can be shared among a set of edges $E_w$ (here, the edges of
//! layer $\ell$ with the same value $v$ share the weight $w_{\ell,v}$). By the chain rule, the
//! gradient is given by
//!
//! $$ \frac{\partial\ \mathrm{WMC}}{\partial w} = \sum_{e \in E_w}
//! \alpha(\mathrm{from}(e)) \times \beta(\mathrm{to}(e)).$$
//!
//! ## Scaling
//!
//! The products of many weights underflow: a path of 400 weights of $10^{-3}$ is worth
//! $10^{-1200}$, which `f64` rounds to zero. The passes therefore rescale layer by layer, as is
//! done to decode long sequences with hidden Markov models. Let $c_\ell$ be a scale factor per
//! layer, with $c_0 = 1$, and let
//!
//! $$A_\ell = \prod_{k \le \ell} c_k, \qquad B_\ell = \prod_{k > \ell} c_k, \qquad
//! \hat\alpha(u) = \frac{\alpha(u)}{A_\ell}, \qquad \hat\beta(u) = \frac{\beta(u)}{B_\ell}
//! \quad (u \in \ell).$$
//!
//! *Forward pass.* Base case: the root is alone in layer $0$, $\alpha(\text{root}) = 1$ and
//! $A_0 = 1$, so $\hat\alpha(\text{root}) = 1$. For a layer $\ell + 1$, compute for every node $v$
//! of that layer
//!
//! $$\tilde\alpha(v) = \sum_{e : \mathrm{to}(e) = v} \hat\alpha(\mathrm{from}(e)) \times w_e,
//! \qquad c_{\ell+1} = \sum_{v \in \ell+1} \tilde\alpha(v), \qquad
//! \hat\alpha(v) = \frac{\tilde\alpha(v)}{c_{\ell+1}}.$$
//!
//! Then $\alpha(v) = A_\ell \times \tilde\alpha(v) = A_{\ell+1} \times \hat\alpha(v)$, so
//! $\hat\alpha$ is indeed the rescaled $\alpha$, and every layer sums to one. The sink $s$ is
//! alone in layer $L$: $\hat\alpha(s) = 1$ and
//!
//! $$\mathrm{WMC} = \alpha(s) = A_L = \prod_{\ell=1}^{L} c_\ell, \qquad
//! \ln \mathrm{WMC} = \sum_{\ell=1}^{L} \ln c_\ell.$$
//!
//! *Backward pass, with the same $c_\ell$.* Base case: $\beta(s) = 1$ and $B_L = 1$ (an empty
//! product), so $\hat\beta(s) = 1$. For a node $u$ of a layer $\ell < L$, since
//! $B_\ell = c_{\ell+1} B_{\ell+1}$,
//!
//! $$\hat\beta(u) = \frac{1}{c_{\ell+1}} \sum_{e \in E_u} w_e \times \hat\beta(\mathrm{to}(e)).$$
//!
//! The two passes meet: $A_\ell B_\ell = \prod_{k=0}^{L} c_k = \mathrm{WMC}$ for every layer,
//! hence $\sum_{u \in \ell} \hat\alpha(u) \times \hat\beta(u) = 1$. At layer $L$ this is
//! $\hat\alpha(s)\hat\beta(s) = 1$, and at layer $0$ it says that the backward pass ends with
//! $\hat\beta(\text{root}) = 1$.
//!
//! *Gradient.* With $\alpha(\mathrm{from}(e)) = A_\ell \times \hat\alpha(\mathrm{from}(e))$ and
//! $\beta(\mathrm{to}(e)) = B_{\ell+1} \times \hat\beta(\mathrm{to}(e))$, and $A_\ell B_\ell =
//! \mathrm{WMC}$,
//!
//! $$\frac{\partial \ln \mathrm{WMC}}{\partial w} = \frac{1}{\mathrm{WMC}}
//! \sum_{e \in E_w} \alpha(\mathrm{from}(e)) \times \beta(\mathrm{to}(e))
//! = \frac{1}{c_{\ell+1}} \sum_{e \in E_w} \hat\alpha(\mathrm{from}(e)) \times
//! \hat\beta(\mathrm{to}(e)).$$
//!
//! In these formulas no number leaves the range of ordinary floating point values.
//!
//! *Degenerate case.* If no weight reaches layer $\ell + 1$ then $c_{\ell+1} = 0$: $\hat\alpha$
//! and $\hat\beta$ are taken to be $0$ from there on (to avoid a division by zero),
//! $\mathrm{WMC} = 0$, $\ln \mathrm{WMC} = -\infty$ and the gradient is zero.
//!
//! # Design notes
//!
//! - **The functions work on logarithms.** [`log_wmc`] and [`log_wmc_and_gradient`] return
//!   $\ln \mathrm{WMC}$ and the gradient of $\ln \mathrm{WMC}$, which is what a loss of the form
//!   $-\ln \mathrm{WMC}$ needs; [`wmc`] exponentiates for the callers that want the count itself
//!   and accept that it may underflow. A diagram without any weighted path has $\ln \mathrm{WMC}
//!   = -\infty$ and a zero gradient.
//! - **Weights are indexed by layer** of the diagram (`weights[layer][value]`, one row per layer
//!   but the sink's), so the functions do not need to know which variable a layer stands for.
//! - **Generic over the representation.** Every function takes an [`MddView`], so it runs on a
//!   compiled [`Mdd`](crate::mdd::Mdd) as well as on a structure shared through the arena.
//! - **The diagram must be well formed.** The passes assume what `Mdd::new`, `Mdd::refine` and the
//!   structures made from them guarantee: a single root and a single sink, which are node `0` of
//!   their layer, and only active edges, each going from a layer to the next. Debug builds check
//!   it on every call.
//! - **Weights are plain numbers.** They need not be normalised or even positive.
//!
//! # Example
//!
//! The probability that two variables with uniform distributions over `{0, 1, 2}` take
//! different values is `6 / 9`, and the gradient says how much each weight contributes to it.
//!
//! ```
//! use aicad::constraints::not_equals;
//! use aicad::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
//! use aicad::mdd::wmc::{log_wmc_and_gradient, wmc};
//! use aicad::mdd::Mdd;
//! use aicad::modelling::Problem;
//! use std::sync::Arc;
//!
//! let mut problem = Problem::default();
//! let x = problem.add_variable(vec![0, 1, 2], None);
//! let y = problem.add_variable(vec![0, 1, 2], None);
//! let constraint = not_equals(&mut problem, x, y);
//!
//! let mut mdd = Mdd::new(
//!     Arc::new(problem),
//!     OrderingHeuristic::MinDomMaxLinked,
//!     MergeHeuristic::LessRelaxed,
//!     SelectHeuristic::Greedy,
//!     &[constraint],
//! );
//! mdd.refine(usize::MAX);
//!
//! // One row of weights per layer: here the first layer is `x`, the second `y`.
//! let uniform = vec![vec![1.0 / 3.0; 3]; 2];
//! assert!((wmc(&mdd, &uniform) - 6.0 / 9.0).abs() < 1e-12);
//!
//! // d(ln WMC) / d(weight of x = 1) = (weight of the values of `y` other than 1) / WMC = 1.
//! let (ln_count, gradient) = log_wmc_and_gradient(&mdd, &uniform);
//! assert!((ln_count.exp() - 6.0 / 9.0).abs() < 1e-12);
//! assert!((gradient[0][1] - 1.0).abs() < 1e-12);
//! ```

use super::NodeIndex;
use super::view::MddView;

/// What the passes of this module compute: the rescaled `alpha` and `beta` of the module
/// documentation and the constants that rescaled them.
struct Scaled {
    /// `alpha[layer][node]` is $\hat\alpha$; every layer sums to one, or to zero if no weight
    /// reaches it.
    alpha: Vec<Vec<f64>>,
    /// `beta[layer][node]` is $\hat\beta$.
    beta: Vec<Vec<f64>>,
    /// `scale[layer]` is $c_{layer}$, the factor of the forward pass in that layer (`1.0` for the
    /// root's layer). One of them is zero exactly when the diagram has no weighted path.
    scale: Vec<f64>,
}

impl Scaled {
    /// $\ln \mathrm{WMC}$, `-inf` if the diagram has no weighted path.
    fn log_wmc(&self) -> f64 {
        self.scale.iter().map(|c| c.ln()).sum()
    }
}

/// Panics, with a message naming the problem, if the diagram does not have the shape that the
/// passes rely on (see the module documentation). Walks the whole diagram, so it only runs in
/// debug builds.
fn assert_well_formed<T: MddView>(mdd: &T) {
    let last_layer = mdd.sink().0;
    assert_eq!(
        last_layer + 1,
        mdd.number_layers(),
        "the sink is in layer {last_layer} but the diagram has {} layers: the sink must be in the \
         last one",
        mdd.number_layers()
    );
    assert_eq!(
        mdd.number_nodes_in_layer(0),
        1,
        "the root layer must have exactly one node"
    );
    assert_eq!(
        mdd.number_nodes_in_layer(last_layer),
        1,
        "the sink layer must have exactly one node"
    );
    assert_eq!(
        mdd.root(),
        NodeIndex(0, 0),
        "the root must be node 0 of layer 0"
    );
    assert_eq!(
        mdd.sink(),
        NodeIndex(last_layer, 0),
        "the sink must be node 0 of the last layer"
    );
    for layer in 0..last_layer {
        for node in mdd.nodes_in_layer(layer) {
            for edge in mdd[node].iter_children() {
                assert!(
                    mdd[edge].is_active(),
                    "node {node:?} lists the deactivated edge {edge:?}: the diagram was not cleaned"
                );
                assert_eq!(
                    mdd[edge].to().0,
                    layer + 1,
                    "edge {edge:?} leaves layer {layer} but does not enter layer {}",
                    layer + 1
                );
            }
        }
    }
}

/// Runs both passes. Panics if `weights` does not have one row per layer but the sink's.
fn scaled_forward_backward<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> Scaled {
    if cfg!(debug_assertions) {
        assert_well_formed(mdd);
    }
    let last_layer = mdd.sink().0;
    if last_layer != weights.len() {
        panic!(
            "Calling the passes on an MDD structure with {} layers and {} layer's weights",
            last_layer,
            weights.len()
        );
    }

    // Forward. The root is the only node of layer 0 and carries all the weight, 1.
    let mut alpha: Vec<Vec<f64>> = Vec::with_capacity(last_layer + 1);
    let mut scale = vec![1.0];
    alpha.push(vec![1.0]);
    for layer in 0..last_layer {
        let mut next_alpha = vec![0.0; mdd.number_nodes_in_layer(layer + 1)];
        for node in mdd.nodes_in_layer(layer) {
            let mass = alpha[layer][node.1];
            // No mass can go through this node.
            if mass == 0.0 {
                continue;
            }
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let weight = weights[layer][value.0];
                next_alpha[mdd[edge].to().1] += mass * weight;
            }
        }
        let total: f64 = next_alpha.iter().sum();
        if total > 0.0 {
            for mass in &mut next_alpha {
                *mass /= total;
            }
        }
        scale.push(total);
        alpha.push(next_alpha);
    }

    // Backward, with the constants of the forward pass.
    let mut beta: Vec<Vec<f64>> = vec![Vec::new(); last_layer + 1];
    beta[last_layer] = vec![1.0];
    for layer in (0..last_layer).rev() {
        let mut layer_beta = vec![0.0; mdd.number_nodes_in_layer(layer)];
        if scale[layer + 1] > 0.0 {
            for node in mdd.nodes_in_layer(layer) {
                let mut mass = 0.0;
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let weight = weights[layer][value.0];
                    mass += weight * beta[layer + 1][mdd[edge].to().1];
                }
                layer_beta[node.1] = mass / scale[layer + 1];
            }
        }
        beta[layer] = layer_beta;
    }

    Scaled { alpha, beta, scale }
}

/// The logarithm of the weighted model count of `mdd`: the total weight of its root-to-sink
/// paths, where an edge of value `v` in `layer` weighs `weights[layer][v]`. It is `-inf` if
/// no path has a positive weight.
///
/// `weights` is indexed by layer of `mdd`, one row per layer but the sink's (see the
/// [module documentation](self)). The row of a layer needs an entry for every value of the
/// variable it branches on.
///
/// # Panics
///
/// Panics if `weights` does not have one row per layer but the sink's, and, when running, if a
/// row is shorter than the largest value of its layer. Debug builds also panic if the diagram is
/// not well formed.
pub fn log_wmc<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> f64 {
    scaled_forward_backward(mdd, weights).log_wmc()
}

/// The weighted model count of `mdd`, see [`log_wmc`]. It underflows to zero when the count is
/// below the smallest `f64`: prefer [`log_wmc`] for long diagrams.
pub fn wmc<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> f64 {
    log_wmc(mdd, weights).exp()
}

/// The logarithm of the weighted model count and its gradient: `gradient[layer][value]` is
/// $\partial \ln \mathrm{WMC} / \partial w_{layer,value}$, the derivative of the logarithm
/// of the count with respect to `weights[layer][value]`. See the [module documentation](self) for
/// the derivation.
///
/// Multiplied by the weight, the gradient is the probability that the variable takes the value
/// given that the constraint holds. Divided by its sum over the values of the layer it is the
/// belief of the constraint, which does not depend on the weights of the layer itself.
///
/// The gradient has the shape of `weights`. If no path has a positive weight the count is
/// `-inf` and the gradient is zero.
///
/// # Panics
///
/// Under the same conditions as [`log_wmc`].
pub fn log_wmc_and_gradient<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> (f64, Vec<Vec<f64>>) {
    let scaled = scaled_forward_backward(mdd, weights);
    let log_count = scaled.log_wmc();
    let last_layer = mdd.sink().0;
    let mut gradient: Vec<Vec<f64>> = weights.iter().map(|row| vec![0.0; row.len()]).collect();
    if log_count == f64::NEG_INFINITY {
        return (log_count, gradient);
    }

    for (layer, row) in gradient.iter_mut().enumerate().take(last_layer) {
        let factor = 1.0 / scaled.scale[layer + 1];
        for node in mdd.nodes_in_layer(layer) {
            let alpha = scaled.alpha[layer][node.1];
            if alpha == 0.0 {
                continue;
            }
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                row[value.0] += factor * alpha * scaled.beta[layer + 1][mdd[edge].to().1];
            }
        }
    }
    (log_count, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, not_equals, sum};
    use crate::mdd::Mdd;
    use crate::mdd::arena::{MddArena, compile_constraint, nominal_domain};
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::mdd::{Edge, EdgeIndex, Node};
    use crate::modelling::{ConstraintIndex, Problem, ValueIndex};
    use std::sync::Arc;

    /// The exact diagram of `constraints`.
    fn build_mdd(problem: Arc<Problem>, constraints: &[ConstraintIndex]) -> Mdd {
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            constraints,
        );
        mdd.refine(usize::MAX);
        mdd
    }

    /// The weighted count by enumeration: the sum, over the assignments of the problem (whose
    /// domains are `0..d`) that satisfy every constraint and agree with `decided` on the values
    /// of `assignment`, of the product of the weights of their values. `weights` is indexed by
    /// variable.
    fn brute_force(
        problem: &Problem,
        weights: &[Vec<f64>],
        decided: &[bool],
        assignment: &[ValueIndex],
    ) -> f64 {
        let mut total = 0.0;
        let mut tuple = vec![0isize; problem.number_variables()];
        loop {
            let agrees =
                (0..tuple.len()).all(|v| !decided[v] || assignment[v].0 as isize == tuple[v]);
            if agrees && problem.is_solution(&tuple) {
                total += (0..tuple.len())
                    .map(|v| weights[v][tuple[v] as usize])
                    .product::<f64>();
            }
            // Next tuple, odometer style.
            let mut position = 0;
            loop {
                if position == tuple.len() {
                    return total;
                }
                tuple[position] += 1;
                if (tuple[position] as usize) < weights[position].len() {
                    break;
                }
                tuple[position] = 0;
                position += 1;
            }
        }
    }

    /// The weights of the layers of `mdd`, from weights indexed by variable.
    fn by_layer(mdd: &Mdd, by_variable: &[Vec<f64>]) -> Vec<Vec<f64>> {
        (0..mdd.number_layers() - 1)
            .map(|layer| by_variable[mdd.decision_at_layer(layer).0].clone())
            .collect()
    }

    /// Arbitrary, distinct weights for `variables` variables over `domain` values.
    fn some_weights(variables: usize, domain: usize) -> Vec<Vec<f64>> {
        (0..variables)
            .map(|v| {
                (0..domain)
                    .map(|d| 0.1 + 0.07 * ((3 * v + 5 * d) % 11) as f64)
                    .collect()
            })
            .collect()
    }

    fn brute_force_not_equals(probs: &[f64], domain_size: usize) -> f64 {
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                if a != b {
                    brute += probs[a] * probs[domain_size + b];
                }
            }
        }
        brute
    }

    /// A diagram built by hand, to feed the passes diagrams that the compiler never produces.
    struct RawDiagram {
        nodes: Vec<Vec<Node>>,
        edges: Vec<Vec<Edge>>,
    }

    impl RawDiagram {
        /// `layers` layers of one node each, plus the sink's, with an edge for every value of
        /// `0..values` between two consecutive nodes.
        fn chain(layers: usize, values: usize) -> Self {
            let mut diagram = RawDiagram {
                nodes: (0..=layers)
                    .map(|layer| vec![Node::new(layer, 0, false)])
                    .collect(),
                edges: vec![Vec::new(); layers],
            };
            for layer in 0..layers {
                for value in 0..values {
                    diagram.add_edge(layer, NodeIndex(layer, 0), NodeIndex(layer + 1, 0), value);
                }
            }
            diagram
        }

        /// Adds an edge leaving `layer` and lists it among the children of `from`.
        fn add_edge(&mut self, layer: usize, from: NodeIndex, to: NodeIndex, value: usize) {
            let index = EdgeIndex(layer, self.edges[layer].len());
            self.edges[layer].push(Edge::new(from, to, ValueIndex(value)));
            self.nodes[from.0][from.1].add_child_edge(index);
        }

        /// A second node in `layer`.
        fn add_node(&mut self, layer: usize) {
            let index = self.nodes[layer].len();
            self.nodes[layer].push(Node::new(layer, index, false));
        }
    }

    impl std::ops::Index<NodeIndex> for RawDiagram {
        type Output = Node;
        fn index(&self, index: NodeIndex) -> &Node {
            &self.nodes[index.0][index.1]
        }
    }

    impl std::ops::Index<EdgeIndex> for RawDiagram {
        type Output = Edge;
        fn index(&self, index: EdgeIndex) -> &Edge {
            &self.edges[index.0][index.1]
        }
    }

    impl MddView for RawDiagram {
        fn root(&self) -> NodeIndex {
            NodeIndex(0, 0)
        }
        fn sink(&self) -> NodeIndex {
            NodeIndex(self.nodes.len() - 1, 0)
        }
        fn number_layers(&self) -> usize {
            self.nodes.len()
        }
        fn number_nodes_in_layer(&self, layer: usize) -> usize {
            self.nodes[layer].len()
        }
        fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = NodeIndex> + '_ {
            (0..self.nodes[layer].len()).map(move |index| NodeIndex(layer, index))
        }
        fn is_unsat(&self) -> bool {
            false
        }
    }

    #[test]
    fn wmc_matches_brute_force_for_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);

        let weights = vec![vec![0.2, 0.5, 0.3], vec![0.1, 0.3, 0.6]];
        let expected = brute_force_not_equals(&weights.concat(), 3);
        assert!((wmc(&mdd, &weights) - expected).abs() < 1e-9);
        assert!((log_wmc(&mdd, &weights) - expected.ln()).abs() < 1e-9);
    }

    #[test]
    fn unit_weights_count_the_solutions() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2, 3], None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);

        assert!((wmc(&mdd, &vec![vec![1.0; 4]; 4]) - 24.0).abs() < 1e-9);
        assert_eq!(mdd.count_solutions().to_string(), "24");
    }

    #[test]
    fn wmc_matches_brute_force_on_several_constraints() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        all_different(&mut problem, vars[..3].to_vec());
        not_equals(&mut problem, vars[2], vars[3]);
        sum(&mut problem, vars.clone(), 4);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(Arc::clone(&problem), &constraints);

        let weights = some_weights(4, 3);
        let expected = brute_force(&problem, &weights, &[false; 4], &[ValueIndex(0); 4]);
        assert!(expected > 0.0);
        assert!((wmc(&mdd, &by_layer(&mdd, &weights)) - expected).abs() < 1e-9);
    }

    #[test]
    fn an_unsatisfiable_diagram_has_no_weight_and_no_gradient() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);
        assert!(mdd.is_unsat());

        let weights = vec![vec![0.5, 0.5]; 3];
        let (log_count, gradient) = log_wmc_and_gradient(&mdd, &weights);
        assert_eq!(log_count, f64::NEG_INFINITY);
        assert_eq!(wmc(&mdd, &weights), 0.0);
        assert_eq!(gradient, vec![vec![0.0; 2]; 3]);
    }

    #[test]
    fn weights_that_forbid_every_solution_give_a_zero_count() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1], None);
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(Arc::clone(&problem), &constraints);

        // Both variables are forced to 0, which `not_equals` rules out.
        let by_variable = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let weights = by_layer(&mdd, &by_variable);
        let (log_count, gradient) = log_wmc_and_gradient(&mdd, &weights);
        assert_eq!(log_count, f64::NEG_INFINITY);
        assert_eq!(gradient, vec![vec![0.0; 2]; 2]);
    }

    #[test]
    fn the_scaled_passes_meet_at_every_layer() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        all_different(&mut problem, vars[..3].to_vec());
        sum(&mut problem, vars.clone(), 4);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);

        let weights = by_layer(&mdd, &some_weights(4, 3));
        let scaled = scaled_forward_backward(&mdd, &weights);
        assert!(scaled.log_wmc() > f64::NEG_INFINITY);
        // Every path crosses each layer exactly once: the paths through the nodes of a layer add
        // up to the whole count, which is 1 once both passes are scaled.
        for layer in 0..mdd.number_layers() {
            let through: f64 = scaled.alpha[layer]
                .iter()
                .zip(&scaled.beta[layer])
                .map(|(a, b)| a * b)
                .sum();
            assert!((through - 1.0).abs() < 1e-9, "layer {layer}: {through}");
            assert!((scaled.alpha[layer].iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn gradient_is_the_derivative_of_the_log_count() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);

        let weights = vec![
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.3, 0.6],
            vec![0.4, 0.4, 0.2],
        ];
        let (log_count, grad) = log_wmc_and_gradient(&mdd, &weights);
        assert!((log_wmc(&mdd, &weights) - log_count).abs() < 1e-12);
        assert_eq!(grad.len(), weights.len());

        let eps = 1e-6;
        for layer in 0..weights.len() {
            for value in 0..weights[layer].len() {
                let mut bumped = weights.clone();
                bumped[layer][value] += eps;
                let finite_diff = (log_wmc(&mdd, &bumped) - log_count) / eps;
                assert!(
                    (finite_diff - grad[layer][value]).abs() < 1e-4,
                    "layer={layer} value={value} analytic={} finite_diff={finite_diff}",
                    grad[layer][value],
                );
            }
        }
    }

    #[test]
    fn gradient_times_the_weight_is_the_probability_of_the_value() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(Arc::clone(&problem), &constraints);

        let by_variable = some_weights(3, 3);
        let weights = by_layer(&mdd, &by_variable);
        let (_, grad) = log_wmc_and_gradient(&mdd, &weights);
        let total = brute_force(&problem, &by_variable, &[false; 3], &[ValueIndex(0); 3]);
        for layer in 0..3 {
            let variable = mdd.decision_at_layer(layer).0;
            let mut layer_sum = 0.0;
            for value in 0..3 {
                let mut decided = [false; 3];
                decided[variable] = true;
                let mut assignment = [ValueIndex(0); 3];
                assignment[variable] = ValueIndex(value);
                let joint = brute_force(&problem, &by_variable, &decided, &assignment);
                let probability = weights[layer][value] * grad[layer][value];
                assert!((probability - joint / total).abs() < 1e-9);
                layer_sum += probability;
            }
            assert!((layer_sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_structure_shared_through_the_arena_counts_with_its_own_order() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        // The scope is given in a different order than the one the diagram will branch in.
        let constraint = sum(&mut problem, vec![vars[3], vars[1], vars[0], vars[2]], 5);
        let problem = Arc::new(problem);
        let compiled = compile_constraint(
            &MddArena::new(),
            &problem,
            constraint,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );

        let weights = some_weights(4, 3);
        let expected = brute_force(&problem, &weights, &[false; 4], &[ValueIndex(0); 4]);
        assert!(expected > 0.0);
        let by_layer: Vec<Vec<f64>> = (0..compiled.order.len())
            .map(|layer| weights[compiled.order[layer].0].clone())
            .collect();
        assert!((wmc(&compiled, &by_layer) - expected).abs() < 1e-9);
    }

    #[test]
    fn a_count_below_the_smallest_float_is_still_known_through_its_logarithm() {
        // 400 layers whose two values weigh 0.001 each: WMC = 0.002^400 = 10^-1079.
        let diagram = RawDiagram::chain(400, 2);
        let weights = vec![vec![1e-3, 1e-3]; 400];

        let (log_count, gradient) = log_wmc_and_gradient(&diagram, &weights);
        assert!((log_count - 400.0 * 0.002f64.ln()).abs() < 1e-9);
        assert_eq!(wmc(&diagram, &weights), 0.0);
        // d(ln WMC) / d(weight) = 1 / (sum of the two weights of the layer).
        for row in &gradient {
            for &g in row {
                assert!((g - 500.0).abs() < 1e-6, "{g}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "layer's weights")]
    fn too_few_rows_of_weights_are_reported() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);
        wmc(&mdd, &[vec![1.0; 3], vec![1.0; 3]]);
    }

    #[test]
    fn a_hand_built_chain_is_well_formed() {
        let diagram = RawDiagram::chain(3, 2);
        assert_well_formed(&diagram);
        assert!((wmc(&diagram, &vec![vec![1.0, 1.0]; 3]) - 8.0).abs() < 1e-9);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "root layer must have exactly one node")]
    fn a_second_root_is_reported() {
        let mut diagram = RawDiagram::chain(2, 2);
        diagram.add_node(0);
        assert_well_formed(&diagram);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "sink layer must have exactly one node")]
    fn a_second_sink_is_reported() {
        let mut diagram = RawDiagram::chain(2, 2);
        diagram.add_node(2);
        assert_well_formed(&diagram);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "deactivated edge")]
    fn a_listed_deactivated_edge_is_reported() {
        let mut diagram = RawDiagram::chain(2, 2);
        diagram.edges[0][1].deactivate();
        assert_well_formed(&diagram);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "does not enter layer")]
    fn an_edge_that_skips_a_layer_is_reported() {
        let mut diagram = RawDiagram::chain(3, 1);
        diagram.add_edge(0, NodeIndex(0, 0), NodeIndex(2, 0), 1);
        assert_well_formed(&diagram);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "root layer must have exactly one node")]
    fn the_passes_check_the_diagram_in_debug_builds() {
        let mut diagram = RawDiagram::chain(2, 2);
        diagram.add_node(0);
        wmc(&diagram, &vec![vec![1.0, 1.0]; 2]);
    }
}
