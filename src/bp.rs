//! Loopy belief propagation over compiled MDDs: refines the per-variable distributions of a
//! network with what the constraints say about them.
//!
//! # Purpose
//!
//! A network gives every variable $x$ a distribution $q_x$ over its domain (`probs`, the softmax
//! of its scores). Sampling each variable from $q_x$ alone ignores the constraints, and the values
//! picked together often violate them. This module computes, for every variable, an approximation
//! of the distribution of its value *given that the constraints hold*, starting from $q_x$.
//! Some variables can be `decided`: they are hard evidence (for instance the part of a board that
//! a destroy/repair step does not resample) and the other variables are refined around them.
//!
//! The scheme is the CP-BP framework \[1\], on a factor graph whose factors are the MDDs of the
//! constraints. Using a network's distribution as a permanent unary factor follows GeAI-BLAnC \[2\],
//! where it is the `oracle` constraint.
//!
//! # How it works
//!
//! ## Messages
//!
//! Let $c$ be a constraint, compiled to an MDD, with scope $S_c$. Two kinds of messages circulate,
//! both distributions over the domain of a variable (normalised, which the proportionality signs
//! below leave implicit):
//!
//! - $\mu_{x \to c}$, from a variable to a constraint containing it in its scope: what the rest of
//!   the problem believes about $x$. It labels the edges of the layer of $x$ in the MDD of $c$: an
//!   edge with value $v$ gets the weight $\mu_{x \to c}(v)$.
//! - $\mu_{c \to x}$, from a constraint to a variable of its scope: what $c$ alone says about
//!   $x$, given what the other variables of its scope say.
//!
//! ## The message of a constraint, and how it is computed
//!
//! The message of $c$ to $x$ is a weighted count of the solutions of $c$ that give $x$ the value
//! $v$, each solution being weighted by the messages of the *other* variables (Eq. 2 \[1\]):
//!
//! $$\mu_{c \to x}(v) \propto \sum_{\tau \models c,\ \tau(x) = v} \ \prod_{y \in S_c \setminus
//! x} \mu_{y \to c}(\tau(y)).$$
//!
//! The outer sum is over all models of the constraint assigning $v$ to $x$. This corresponds
//! exactly to the paths of an MDD following the edges labelled with $v$ at the layer of $x$.
//! The inner product multiplies the messages of all the variables in the constraint's scope,
//! except $x$. Since the messages from the variables to the constraints are used to weight the
//! edges of the MDD, this corresponds to considering all the paths passing through edges labelled
//! $v$ and summing their weight, excluding $x$'s contribution. This is exactly how the
//! [WMC](crate::mdd::wmc) is computed, see its documentation for the notations.
//!
//! Let us consider the message $\mu_{c \to x}(v)$, with $E_{x,v}$ the set of edges labelled with
//! $v$ in the layer of $x$. Let $\alpha(u)$ be the sum of the weights of the paths from the root
//! to a node $u$ and $\beta(u)$ the sum of those from $u$ to the sink, with the current labels.
//! Then, before normalisation,
//!
//! $$\mu_{c \to x}(v) = \sum_{e \in E_{x,v}} \alpha(\mathrm{from}(e))\beta(\mathrm{to}(e)).$$
//!
//! $x$'s contribution is excluded without any extra work: $\alpha$ only covers the layers above
//! the layer of $x$ and $\beta$ only the layers below it, so the labels of that layer never enter.
//! For the same reason this is the gradient of the WMC with respect to the labels of the layer
//! of $x$, and a single forward and a single backward pass give the messages to *all* the
//! variables of the scope (`mdd_local_beliefs`, which normalises the gradient returned by
//! [`log_wmc_and_gradient`]).
//!
//! The message is a likelihood. If the variables of the scope are independent with the
//! distributions $\mu_{y \to c}$, then $\mu_{c \to x}(v) \propto \Pr[c \text{ holds} \mid x = v]$:
//! how much of the remaining scope can still complete a solution if $x = v$. It is only the
//! evidence brought by $c$, the label $\mu_{x \to c}(v)$ (what the variable already believes
//! about itself) being left out of it. Where the support propagation of classical CP only keeps
//! whether this is positive, BP keeps how large it is.
//!
//! ## One round
//!
//! The edges of every MDD start labelled with the priors, $\mu_{x \to c} = q_x$. A round has
//! three steps:
//!
//! 1. **Compute the messages from the constraints to the variables.** Each MDD runs its
//!    forward and backward passes with its current labels and reads $\mu_{c \to x}$ for every
//!    variable of its scope (`mdd_local_beliefs`).
//! 2. **Update the marginals.** A variable multiplies its prior by the messages of all the
//!    constraints that contain it, in log-space:
//!    $$m_x(v) \propto q_x(v) \prod_{c \ni x} \mu_{c \to x}(v).$$
//!    The prior is the evidence from outside the factor graph and the messages are the evidence
//!    of the constraints, each of which is counted once.
//! 3. **Relabel the edges of the MDDs.** The variable tells each of its constraints what the
//!    others think, with that constraint's own message divided back out (Eq. 1, the *cavity*
//!    message, \[1\]):
//!    $$\mu_{x \to c}(v) \propto \frac{m_x(v)}{\mu_{c \to x}(v)} = q_x(v) \prod_{c^\prime \neq
//!    c} \mu_{c^\prime \to x}(v).$$
//!    The product is over the other constraints $c^\prime \neq c$ that contain $x$. Without the
//!    division, $c$ would receive its own opinion back as new evidence and count it twice. The new
//!    messages label the layer of $x$ in the MDD of $c$ for the next round.
//!
//! After `iterations` rounds the marginals $m_x$ are returned. With no round they are the priors.
//!
//! ## Decided variables
//!
//! A `decided` variable has the one-hot prior $\mathbb{1}[v = a_x]$ at the value $a_x$ of
//! `assignment`. It is never updated and always sends that one-hot message, so in every MDD the
//! paths that give it another value have weight zero: the evidence restricts what the other
//! variables of the constraint can say.
//!
//! # Design notes
//!
//! - **The prior is permanent.** Pesant's baseline resets every marginal to uniform at each round,
//!   which is the same as having no prior. We reset to the prior instead, as a unary factor that
//!   sends the same message every round. Without it, the network's distribution would wash out
//!   after a couple of rounds and BP would reduce to counting.
//! - **No domain filtering.** The CP-BP framework interleaves BP with constraint propagation,
//!   which removes the values that cannot be part of a solution. Here nothing is removed and
//!   nothing is propagated between rounds: a value that a constraint rules out gets a zero
//!   message and its marginal becomes zero, but it stays in the domain. The only hard evidence is
//!   the `decided` variables.
//! - **Every constraint counts equally**, as in the paper: there is no weight per MDD, nor on the
//!   prior.
//! - **Exact for one constraint, approximate for several.** With a single constraint, one round
//!   gives the marginal $\Pr[x = v \mid c]$ under the priors. With several, the product in step 2
//!   treats the constraints as independent evidence, which they are not when their scopes
//!   overlap, hence an approximation. There is no convergence check: a handful of rounds is meant
//!   to run inside the search, and Pesant observes that a few rounds capture most of the benefit,
//!   the distance to the true marginals sometimes growing again with more.
//! - **Numerics.** Products are accumulated as sums of logarithms, with `safe_ln` mapping a zero
//!   to a very small number. A constraint with no solution of positive weight under the current
//!   labels has no evidence to give and sends a uniform message (`normalize_or_uniform`).
//! - **Indexing.** The messages of an MDD are indexed by its layers, not by the global index of
//!   the variable: a variable owns a different layer in each of its constraints, which is what
//!   `build_var_to_mdds` records.
//!
//! # Example
//!
//! Take $x \ne y$ over $\lbrace 0, 1 \rbrace$ with $q_x = q_y = (0.9, 0.1)$. The MDD has two
//! paths, $x = 0, y = 1$ and $x = 1, y = 0$, and all the edges are labelled with the priors.
//!
//! - **Messages.** Reading the edges of $x$'s layer, $\alpha$ is 1 at the root and $\beta$ is the
//!   label of the $y$ edge below, so $\mu_{c \to x} \propto (q_y(1), q_y(0)) = (0.1, 0.9)$, and
//!   the same for $y$. The constraint pushes each variable away from the value that both favour.
//! - **Marginals.** $m_x \propto (0.9 \cdot 0.1, 0.1 \cdot 0.9) = (0.09, 0.09)$, that is
//!   $(0.5, 0.5)$, and the same for $y$.
//! - **Relabelling.** $m_x / \mu_{c \to x} = (0.5 / 0.1, 0.5 / 0.9) \propto (0.9, 0.1) = q_x$: with
//!   one constraint, the cavity message is the prior, so further rounds change nothing.
//!
//! The test `single_not_equals_mdd_with_symmetric_conflicting_prior_becomes_uniform_after_one_iteration`
//! checks these marginals.
//!
//! # References
//!
//! \[1\] "From Support Propagation to Belief Propagation in Constraint Programming" - G. Pesant,
//! JAIR 2019
//!
//! \[2\] "Constrained Sequential Inference in Machine Learning using Constraint Programming" - V.
//! Manibod et al., IJCAI 2025

use crate::mdd::wmc::log_wmc_and_gradient;
use crate::mdd::{CompiledConstraint, Mdd};
use crate::modelling::ValueIndex;

const LOG_ZERO: f64 = -745.0;

fn safe_ln(p: f64) -> f64 {
    if p > 0.0 { p.ln() } else { LOG_ZERO }
}

/// Normalises a categorical probability distribution if at least one element has non-zero weight,
/// otherwise returns a uniform distribution
fn normalize_or_uniform(mut weights: Vec<f64>, domain_size: usize) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / domain_size as f64; domain_size];
    }
    for w in &mut weights {
        *w /= total;
    }
    weights
}

/// Panics, with a message naming what is wrong, if the arguments of `belief_propagation` do not
/// fit together. Without it the same mistakes surface as out-of-bounds indexing deep in the
/// passes, which does not say which argument was wrong.
fn check_inputs(
    mdds: &[CompiledConstraint],
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
) {
    let num_vars = probs.len();
    assert_eq!(
        assignment.len(),
        num_vars,
        "belief_propagation: `assignment` has {} entries for {num_vars} variables",
        assignment.len()
    );
    assert_eq!(
        decided.len(),
        num_vars,
        "belief_propagation: `decided` has {} entries for {num_vars} variables",
        decided.len()
    );
    for (var, row) in probs.iter().enumerate() {
        assert!(
            !row.is_empty(),
            "belief_propagation: the prior of variable {var} is empty"
        );
        assert!(
            row.iter().all(|p| p.is_finite() && *p >= 0.0),
            "belief_propagation: the prior of variable {var} has a negative or non-finite \
             weight: {row:?}"
        );
        assert!(
            !decided[var] || assignment[var].0 < row.len(),
            "belief_propagation: variable {var} is decided at value {} but its prior has only {} \
             values",
            assignment[var].0,
            row.len()
        );
    }
    for (mdd_index, mdd) in mdds.iter().enumerate() {
        for layer in 0..mdd.number_decisions() {
            let var = mdd.decision_at_layer(layer).0;
            assert!(
                var < num_vars,
                "belief_propagation: MDD {mdd_index} branches on variable {var} but only \
                 {num_vars} priors were given"
            );
        }
    }
}

/// For each global variable, every `(mdd_index, layer)` pair where that MDD has this variable in
/// its scope, at that layer. Used by `belief_propagation` (multi-round marginal aggregation)
/// to know "which MDDs does this variable belong to, and at what layer in each".
///
/// `num_vars` is passed explicitly rather than read off a problem since a `CompiledConstraint`
/// shares its diagram with other constraints, and the diagram's problem is the template's (see the
/// arena module doc).
fn build_var_to_mdds(mdds: &[CompiledConstraint], num_vars: usize) -> Vec<Vec<(usize, usize)>> {
    let mut var_to_mdds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_vars];
    for (mdd_index, mdd) in mdds.iter().enumerate() {
        for layer in 0..mdd.number_decisions() {
            let variable = mdd.decision_at_layer(layer);
            var_to_mdds[variable.0].push((mdd_index, layer));
        }
    }
    var_to_mdds
}

fn log_combine_and_normalize(log_combined: Vec<f64>) -> Vec<f64> {
    let max_log = log_combined
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut combined: Vec<f64> = log_combined.iter().map(|&l| (l - max_log).exp()).collect();
    let total: f64 = combined.iter().sum();
    for c in &mut combined {
        *c /= total;
    }
    combined
}

/// A one-hot distribution: all mass on `value`. Used as the permanent "message"/prior for a
/// `decided` variable.
fn one_hot(value: ValueIndex, domain_size: usize) -> Vec<f64> {
    let mut weights = vec![0.0; domain_size];
    weights[value.0] = 1.0;
    weights
}

/// One MDD's local belief for every variable in its scope.
/// Returned as one probability vector per layer, indexed the same way as `mdd.decision_at_layer` .
fn mdd_local_beliefs(mdd: &Mdd, messages: &[Vec<f64>]) -> Vec<Vec<f64>> {
    log_wmc_and_gradient(mdd, messages)
        .1
        .into_iter()
        .zip(messages)
        .map(|(weights, m)| normalize_or_uniform(weights, m.len()))
        .collect()
}

/// Cavity message sent from the variables to the constraints. This is the marginals of the
/// variables with the constraint's own contribution divided back out. This division is made in
/// log-space via `safe_ln` so local belief that are zero don't lead to infinities.
fn cavity_message(marginal: &[f64], local_belief: &[f64]) -> Vec<f64> {
    let weights: Vec<f64> = marginal
        .iter()
        .zip(local_belief)
        .map(|(&m, &l)| (safe_ln(m) - safe_ln(l)).exp())
        .collect();
    normalize_or_uniform(weights, marginal.len())
}

/// Runs the belief propagation over the `mdds` and the priors `probs` for `iterations` steps.
/// The current assignment and the decided values are given as parameter to generate one-hot
/// distributions for these variables.
///
/// The MDDs are expected to be satisfiable. An unsatisfiable one, like a constraint that the
/// decided variables make infeasible, has no solution to count: it sends a uniform message, that
/// is no information, and does not make the call fail.
///
/// # Panics
///
/// - If `assignment` or `decided` does not have one entry per row of `probs`.
/// - If a row of `probs` is empty or holds a negative or non-finite weight.
/// - If a `decided` variable has a value outside its row of `probs`.
/// - If an MDD branches on a variable that has no row in `probs`.
/// - If a row of `probs` is shorter than the largest value its variable takes in an MDD (see
///   `crate::mdd::wmc::log_wmc_and_gradient`).
pub(crate) fn belief_propagation(
    mdds: &[CompiledConstraint],
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
    iterations: usize,
) -> Vec<Vec<f64>> {
    let num_vars = probs.len();
    check_inputs(mdds, probs, assignment, decided);
    let var_to_mdds = build_var_to_mdds(mdds, num_vars);

    // The permanent prior for each variable: the prior belief for one that's still free,
    // or a one-hot at its fixed value for one already `decided`.
    let priors: Vec<Vec<f64>> = (0..num_vars)
        .map(|var| {
            if decided[var] {
                one_hot(assignment[var], probs[var].len())
            } else {
                probs[var].clone()
            }
        })
        .collect();

    // messages[mdd_index][layer]: the cavity message that layer's variable currently sends this
    // MDD. Seeded from `priors` -- Algorithm 5's "reset to 1" becomes "reset to the prior"
    // throughout, since the prior is permanent here rather than a one-time initialization.
    let mut messages: Vec<Vec<Vec<f64>>> = mdds
        .iter()
        .map(|mdd| {
            (0..mdd.number_decisions())
                .map(|layer| priors[mdd.decision_at_layer(layer).0].clone())
                .collect()
        })
        .collect();

    // local_beliefs[mdd_index][layer], carried across iterations so next round's cavity message
    // can divide this round's local belief back out.
    let mut local_beliefs: Vec<Vec<Vec<f64>>> = mdds
        .iter()
        .map(|mdd| {
            (0..mdd.number_decisions())
                .map(|layer| vec![1.0; probs[mdd.decision_at_layer(layer).0].len()])
                .collect()
        })
        .collect();

    let mut marginal: Vec<Vec<f64>> = priors.clone();

    for _ in 0..iterations {
        // Every constraint computes its local beliefs for every variable in its scope from this
        // round's messages, in one combined forward-backward pass each.
        for (mdd_index, mdd) in mdds.iter().enumerate() {
            local_beliefs[mdd_index] = mdd_local_beliefs(mdd, &messages[mdd_index]);
        }

        // Reset every variable's marginal to its prior, then accumulate every MDD's local belief.
        let mut log_marginal: Vec<Vec<f64>> = priors
            .iter()
            .map(|p| p.iter().map(|&v| safe_ln(v)).collect())
            .collect();
        for var in 0..num_vars {
            if decided[var] {
                continue;
            }
            for &(mdd_index, layer) in &var_to_mdds[var] {
                let belief = &local_beliefs[mdd_index][layer];
                for (d, log_d) in log_marginal[var].iter_mut().enumerate() {
                    *log_d += safe_ln(belief[d]);
                }
            }
        }
        // Updates the marginals for the variables.
        marginal = log_marginal
            .into_iter()
            .map(log_combine_and_normalize)
            .collect();

        // Re-weight the MDDs' edges with the new marginals.
        for var in 0..num_vars {
            for &(mdd_index, layer) in &var_to_mdds[var] {
                messages[mdd_index][layer] = if decided[var] {
                    priors[var].clone()
                } else {
                    cavity_message(&marginal[var], &local_beliefs[mdd_index][layer])
                };
            }
        }
    }

    marginal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, gcc, not_equals, sum};
    use crate::mdd::Compiler;
    use crate::mdd::heuristics::{
        CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
    };
    use crate::modelling::{ConstraintIndex, Problem};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::sync::Arc;

    /// Same helper as the NLS decode tests -- duplicated locally rather than exposed across
    /// the two test modules, to keep each self-contained.
    fn build_mdd(problem: Arc<Problem>, constraints: &[ConstraintIndex]) -> CompiledConstraint {
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            constraints,
        );
        compiler.refine(usize::MAX);
        CompiledConstraint::from_mdd(compiler.into_mdd())
    }

    #[test]
    fn belief_propagation_indexes_messages_by_mdd_layer_not_by_global_variable_id() {
        // Regression test for a real out-of-bounds panic hit on Sudoku: `w` sits *before* `x` and
        // `y` in the problem's global variable numbering (mirrors
        // `crate::local_search::decode::tests::destroyed_variables_with_deterministic_rule_destroys_exactly_the_violated_mdds`'s
        // `_w` pattern, just placed first here instead of last), so x and y's global indices (1, 2)
        // no longer line up with their layer indices in the *2*-layer MDD covering just {x, y}
        // (layers 0, 1). `unclamped_alpha`/`unclamped_beta`/`mdd_local_beliefs` all index the
        // per-MDD `messages` array -- length 2 here, one entry per layer -- so indexing it by
        // `variable.0` (1 and 2) instead of by layer (0 and 1) either silently reads the wrong
        // variable's message or, as happened on a 9-wide Sudoku group, panics with an
        // out-of-bounds index once a variable's global id exceeds the MDD's own layer count.
        // Otherwise identical to `single_not_equals_mdd_with_symmetric_conflicting_prior_becomes_uniform_after_one_iteration`,
        // so the expected result is the same hand-derived [0.5, 0.5] collapse for both x and y.
        let mut problem = Problem::default();
        let _w = problem.add_variable(vec![0, 1, 2, 3, 4], None);
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let mdds = vec![mdd];

        let probs = vec![vec![0.2; 5], vec![0.9, 0.1], vec![0.9, 0.1]];
        let assignment = vec![ValueIndex(0); 3];
        let decided = vec![false; 3];
        let marginals = belief_propagation(&mdds, &probs, &assignment, &decided, 1);

        assert!((marginals[x.0][0] - 0.5).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[x.0][1] - 0.5).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[y.0][0] - 0.5).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[y.0][1] - 0.5).abs() < 1e-9, "{marginals:?}");
    }

    #[test]
    fn zero_iterations_returns_probs_unchanged() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let mdds = vec![mdd];

        let probs = vec![vec![0.9, 0.1], vec![0.3, 0.7]];
        let assignment = vec![ValueIndex(0); 2];
        let decided = vec![false; 2];
        let marginals = belief_propagation(&mdds, &probs, &assignment, &decided, 0);
        assert_eq!(marginals, probs);
    }

    #[test]
    fn single_not_equals_mdd_with_symmetric_conflicting_prior_becomes_uniform_after_one_iteration()
    {
        // x != y over {0,1} x {0,1}, both variables' priors favouring the *same* value (0) equally
        // strongly. Hand-derivable: with domain size 2, the MDD's local belief for each variable is
        // exactly the *other* variable's message reversed (the only way to satisfy != is to take
        // the complementary value), so local_belief_x = reverse(probs_y) = [0.1, 0.9] and
        // local_belief_y = reverse(probs_x) = [0.1, 0.9]. Combined with the identical-but-reversed
        // prior, `normalize(prior * local_belief)` collapses to exactly [0.5, 0.5] for both --
        // the constraint's pull away from the shared favourite exactly cancels the prior's pull
        // toward it.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let mdds = vec![mdd];

        let probs = vec![vec![0.9, 0.1], vec![0.9, 0.1]];
        let assignment = vec![ValueIndex(0); 2];
        let decided = vec![false; 2];
        let marginals = belief_propagation(&mdds, &probs, &assignment, &decided, 1);

        assert_eq!(marginals.len(), 2);
        for marginal in &marginals {
            assert!((marginal[0] - 0.5).abs() < 1e-9, "{marginals:?}");
            assert!((marginal[1] - 0.5).abs() < 1e-9, "{marginals:?}");
        }
    }

    #[test]
    fn deciding_one_variable_clamps_it_and_propagates_as_hard_evidence_to_the_other() {
        // Same x != y MDD, but now x is `decided` (fixed at 0, the way a destroy/repair loop would
        // clamp every variable it isn't resampling this round) while y is still free. Hand-derivable
        // exactly like the symmetric-prior test: x's message to the MDD is now the one-hot [1, 0]
        // (not its raw prior), which forces the MDD's local belief for y to collapse to [0, 1] --
        // and since y is undecided, its returned marginal combines that with its own prior [0.9,
        // 0.1] to land squarely on [0, 1] (the near-zero mass at y=0 in the prior is wiped out by
        // the local belief's *exact* zero there). x, being `decided`, must come back unchanged at
        // its fixed one-hot regardless of what the MDD believes about it.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let mdds = vec![mdd];

        let probs = vec![vec![0.9, 0.1], vec![0.9, 0.1]];
        let mut assignment = vec![ValueIndex(0); 2];
        assignment[x.0] = ValueIndex(0);
        let mut decided = vec![false; 2];
        decided[x.0] = true;

        let marginals = belief_propagation(&mdds, &probs, &assignment, &decided, 1);

        assert!((marginals[x.0][0] - 1.0).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[x.0][1] - 0.0).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[y.0][0] - 0.0).abs() < 1e-9, "{marginals:?}");
        assert!((marginals[y.0][1] - 1.0).abs() < 1e-9, "{marginals:?}");
    }

    #[test]
    fn uninformative_mdds_leave_the_marginal_close_to_the_prior_after_any_number_of_iterations() {
        // An unbound gcc over a single variable is a real MDD (real scope) with no actual
        // restriction -- every value is always reachable, so its local belief is uniform every
        // round and should never sharpen or distort the prior, for however many MDDs share the
        // variable or however many rounds run (mirrors
        // `combined_partial_conditional_does_not_multiply_the_networks_own_belief_once_per_mdd_membership`
        // for the single-shot combiner).
        let probs = vec![vec![0.7, 0.2, 0.1]];

        for count in [1, 5] {
            let mdds: Vec<CompiledConstraint> = (0..count)
                .map(|_| {
                    let mut problem = Problem::default();
                    let x = problem.add_variable(vec![0, 1, 2], None);
                    gcc(&mut problem, vec![x], vec![]);
                    let problem = Arc::new(problem);
                    let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
                    build_mdd(problem, &constraints)
                })
                .collect();

            for iterations in [1, 3] {
                let assignment = vec![ValueIndex(0)];
                let decided = vec![false];
                let marginals =
                    belief_propagation(&mdds, &probs, &assignment, &decided, iterations);
                assert_eq!(marginals.len(), 1);
                for d in 0..3 {
                    assert!(
                        (marginals[0][d] - probs[0][d]).abs() < 1e-6,
                        "count={count}, iterations={iterations}, d={d}, marginals={marginals:?}"
                    );
                }
            }
        }
    }

    // ---- Input checks ----

    /// `x != y` over `{0, 1}` and the arguments of a valid call on it, for the tests of the checks.
    fn not_equals_inputs() -> (Vec<CompiledConstraint>, Vec<Vec<f64>>) {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        (
            vec![build_mdd(problem, &constraints)],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        )
    }

    #[test]
    #[should_panic(expected = "`assignment` has 1 entries for 2 variables")]
    fn an_assignment_of_the_wrong_length_panics() {
        let (mdds, probs) = not_equals_inputs();
        belief_propagation(&mdds, &probs, &[ValueIndex(0)], &[false; 2], 1);
    }

    #[test]
    #[should_panic(expected = "`decided` has 3 entries for 2 variables")]
    fn a_decided_mask_of_the_wrong_length_panics() {
        let (mdds, probs) = not_equals_inputs();
        belief_propagation(&mdds, &probs, &[ValueIndex(0); 2], &[false; 3], 1);
    }

    #[test]
    #[should_panic(expected = "the prior of variable 1 is empty")]
    fn an_empty_prior_panics() {
        let (mdds, mut probs) = not_equals_inputs();
        probs[1] = vec![];
        belief_propagation(&mdds, &probs, &[ValueIndex(0); 2], &[false; 2], 1);
    }

    #[test]
    #[should_panic(expected = "the prior of variable 0 has a negative or non-finite weight")]
    fn a_negative_prior_weight_panics() {
        let (mdds, mut probs) = not_equals_inputs();
        probs[0] = vec![1.5, -0.5];
        belief_propagation(&mdds, &probs, &[ValueIndex(0); 2], &[false; 2], 1);
    }

    #[test]
    #[should_panic(expected = "the prior of variable 0 has a negative or non-finite weight")]
    fn a_nan_prior_weight_panics() {
        let (mdds, mut probs) = not_equals_inputs();
        probs[0] = vec![f64::NAN, 0.5];
        belief_propagation(&mdds, &probs, &[ValueIndex(0); 2], &[false; 2], 1);
    }

    #[test]
    #[should_panic(expected = "variable 1 is decided at value 2 but its prior has only 2 values")]
    fn a_decided_value_outside_its_prior_panics() {
        let (mdds, probs) = not_equals_inputs();
        belief_propagation(
            &mdds,
            &probs,
            &[ValueIndex(0), ValueIndex(2)],
            &[false, true],
            1,
        );
    }

    #[test]
    #[should_panic(expected = "MDD 0 branches on variable 1 but only 1 priors were given")]
    fn an_mdd_over_a_variable_without_prior_panics() {
        let (mdds, probs) = not_equals_inputs();
        belief_propagation(&mdds, &probs[..1], &[ValueIndex(0)], &[false], 1);
    }

    #[test]
    fn an_unsatisfiable_mdd_sends_no_information() {
        // x != y with both variables decided at 0: the evidence makes the constraint infeasible,
        // so every local belief is the uniform fallback, the call neither fails nor produces a NaN,
        // and the decided variables stay put.
        let (mdds, probs) = not_equals_inputs();
        let marginals = belief_propagation(&mdds, &probs, &[ValueIndex(0); 2], &[true, true], 2);
        for marginal in &marginals {
            assert!((marginal[0] - 1.0).abs() < 1e-12, "{marginals:?}");
            assert!(marginal[1] < 1e-12, "{marginals:?}");
        }
    }

    // ---- Exactness for a single constraint ----

    /// The distribution of every variable given that `problem` holds, with independent variables
    /// distributed as `priors`, by enumerating all the assignments. The reference for the
    /// exactness of belief propagation on a single constraint.
    fn exact_marginals(problem: &Problem, priors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let sizes: Vec<usize> = priors.iter().map(Vec::len).collect();
        let mut marginals: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
        let mut assignment = vec![0isize; sizes.len()];
        loop {
            if problem.is_solution(&assignment) {
                let weight: f64 = assignment
                    .iter()
                    .enumerate()
                    .map(|(var, &value)| priors[var][value as usize])
                    .product();
                for (var, &value) in assignment.iter().enumerate() {
                    marginals[var][value as usize] += weight;
                }
            }
            // Next assignment, like an odometer.
            let mut var = 0;
            while var < sizes.len() {
                assignment[var] += 1;
                if (assignment[var] as usize) < sizes[var] {
                    break;
                }
                assignment[var] = 0;
                var += 1;
            }
            if var == sizes.len() {
                break;
            }
        }
        marginals
            .into_iter()
            .map(|row| {
                let total: f64 = row.iter().sum();
                row.into_iter().map(|w| w / total).collect()
            })
            .collect()
    }

    #[test]
    fn with_a_single_constraint_one_round_gives_the_exact_marginals() {
        // With one constraint there is no loop in the factor graph, so the marginals are the
        // exact distribution of each variable given that the constraint holds. Checked against
        // enumeration, on random priors, with and without a decided variable, and on a variable
        // outside the scope, whose marginal is its prior. Further rounds change nothing: the
        // cavity message of a lone constraint is the prior.
        type Build = fn(&mut Problem) -> Vec<crate::modelling::VariableIndex>;
        let cases: [(&str, Build); 3] = [
            ("all_different", |problem| {
                let vars = problem.add_variables(4, vec![0, 1, 2], None);
                all_different(problem, vars[..3].to_vec());
                vars
            }),
            ("sum", |problem| {
                let vars = problem.add_variables(4, vec![0, 1, 2], None);
                sum(problem, vars[..3].to_vec(), 3);
                vars
            }),
            ("gcc", |problem| {
                let vars = problem.add_variables(4, vec![0, 1, 2], None);
                gcc(problem, vars.clone(), vec![(0, 1, 2), (1, 0, 1), (2, 1, 3)]);
                vars
            }),
        ];

        let mut rng = StdRng::seed_from_u64(7);
        for (name, build) in cases {
            let mut problem = Problem::default();
            build(&mut problem);
            let problem = Arc::new(problem);
            let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            let mdds = vec![build_mdd(problem.clone(), &constraints)];
            let n = problem.number_variables();

            for decide_first in [false, true] {
                for _ in 0..10 {
                    let probs: Vec<Vec<f64>> = (0..n)
                        .map(|_| {
                            let weights: Vec<f64> =
                                (0..3).map(|_| rng.random_range(0.05..1.0)).collect();
                            let total: f64 = weights.iter().sum();
                            weights.into_iter().map(|w| w / total).collect()
                        })
                        .collect();
                    // Variable 0 decided at 1, which every case can complete to a solution.
                    let mut assignment = vec![ValueIndex(0); n];
                    let mut decided = vec![false; n];
                    let mut reference_priors = probs.clone();
                    if decide_first {
                        assignment[0] = ValueIndex(1);
                        decided[0] = true;
                        reference_priors[0] = one_hot(ValueIndex(1), 3);
                    }
                    let expected = exact_marginals(&problem, &reference_priors);

                    for iterations in [1, 3] {
                        let marginals =
                            belief_propagation(&mdds, &probs, &assignment, &decided, iterations);
                        for var in 0..n {
                            for value in 0..3 {
                                assert!(
                                    (marginals[var][value] - expected[var][value]).abs() < 1e-9,
                                    "{name}, decided={decide_first}, iterations={iterations}, \
                                     variable {var}, value {value}: got {marginals:?}, \
                                     expected {expected:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
