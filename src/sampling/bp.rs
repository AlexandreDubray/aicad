//! Loopy belief propagation over a set of compiled MDDs, following Gilles Pesant's "From Support
//! Propagation to Belief Propagation in Constraint Programming" (JAIR 66, 2019). Each MDD is one
//! factor/constraint in a factor graph; `probs` (the network's output) plays the role of a
//! permanent unary prior factor on every variable, present every round. The paper's own baseline
//! has no such prior -- Algorithm 5 resets every variable's marginal to 1 each round -- we reset to
//! `probs` instead, since without it the network's beliefs would just wash out after a couple of
//! rounds and BP would degrade to plain (unweighted) counting-based propagation.
//!
//! Notation follows the paper directly:
//! - `mu_{x -> c}(v)`, Eq. 1: the *cavity* message a variable sends a constraint -- its current
//!   marginal with that same constraint's own previous contribution divided back out, so a
//!   constraint never receives its own opinion back as evidence about itself.
//! - `mu_{c -> x}(v)`, Eq. 2: the message a constraint sends a variable -- weighted counting over
//!   every tuple consistent with `x=v`, weighted by every *other* scope variable's incoming
//!   message. `x`'s own message is explicitly excluded from that product.
//!
//! For a compiled MDD, "weighted counting consistent with x=v" is exactly a forward/backward (WMC)
//! pass: `alpha[layer]` accumulates path mass up to `layer` and `beta[layer]` accumulates it from
//! `layer` to the sink, both weighted edge-by-edge by the current message of whichever variable
//! owns that layer. Reading off `alpha[layer(x)][node] * beta[layer(x)+1][child]` for every edge
//! labelled `v` gives `mu_{c -> x}(v)` directly -- *without* multiplying in `x`'s own message,
//! since layer `layer(x)` is exactly the one being read off, not one folded into alpha or beta.
//!
//! `partial_alpha_at`/`partial_beta_at` (see `super`) already compute a clamped version of this,
//! one target layer at a time, which costs O(scope_size * edges) across a full variable-by-variable
//! decode. Belief propagation instead needs *every* variable's local belief from the *same* MDD,
//! every round, which is naturally an O(edges) computation instead: one full forward pass and one
//! full backward pass per MDD per iteration, with every layer's local belief read off the same pair
//! of arrays in a single combined pass (`mdd_local_beliefs`).

use crate::mdd::Mdd;
use crate::modelling::ValueIndex;

use super::{build_var_to_mdds, log_combine_and_normalize, normalize_or_uniform, safe_ln};

/// A one-hot distribution: all mass on `value`. Used as the permanent "message"/prior for a
/// `decided` variable -- hard evidence from outside this belief-propagation call (e.g. the rest of
/// a destroy/repair board that isn't being resampled this round), rather than something for the
/// MDDs to keep refining.
fn one_hot(value: ValueIndex, domain_size: usize) -> Vec<f64> {
    let mut weights = vec![0.0; domain_size];
    weights[value.0] = 1.0;
    weights
}

/// Full, unclamped forward (WMC) pass over `mdd`: `alpha[layer][node]` is the total mass reaching
/// `node` (indexed within its layer) from the root, with each layer's edges weighted by
/// `messages[layer][value]` -- `messages` is indexed *by layer within this MDD* (length
/// `mdd.number_layers() - 1`), not by global variable id, since it's built per-MDD by
/// `belief_propagation` from that MDD's own scope. Unlike `partial_alpha_at`, every layer sums over
/// every outgoing edge -- there's no `decided`/clamped layer here -- and the *entire* array of
/// per-layer vectors is kept (one entry per layer, root through sink) rather than only the value at
/// one target layer, since belief propagation needs every layer's local belief out of one pass.
fn unclamped_alpha(mdd: &Mdd, messages: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let last_layer = mdd.sink().0;
    let mut alphas: Vec<Vec<f64>> = Vec::with_capacity(last_layer + 1);
    alphas.push(vec![1.0; mdd.number_nodes_in_layer(0)]);

    for layer in 0..last_layer {
        let mut next_alpha = vec![0.0; mdd.number_nodes_in_layer(layer + 1)];
        for node in mdd.nodes_in_layer(layer) {
            let mass = alphas[layer][node.1];
            if mass == 0.0 {
                continue;
            }
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let weight = messages[layer][value.0];
                next_alpha[mdd[edge].to().1] += mass * weight;
            }
        }
        alphas.push(next_alpha);
    }

    alphas
}

/// The backward counterpart of `unclamped_alpha`: `beta[layer][node]` is the total mass from
/// `node` to the sink, weighted the same way (`messages` indexed by layer, same convention).
fn unclamped_beta(mdd: &Mdd, messages: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let last_layer = mdd.sink().0;
    let mut betas: Vec<Vec<f64>> = vec![Vec::new(); last_layer + 1];
    betas[last_layer] = vec![1.0; mdd.number_nodes_in_layer(last_layer)];

    for layer in (0..last_layer).rev() {
        let mut prev_beta = vec![0.0; mdd.number_nodes_in_layer(layer)];
        for node in mdd.nodes_in_layer(layer) {
            let mut mass = 0.0;
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let weight = messages[layer][value.0];
                mass += weight * betas[layer + 1][mdd[edge].to().1];
            }
            prev_beta[node.1] = mass;
        }
        betas[layer] = prev_beta;
    }

    betas
}

/// One MDD's local belief for every variable in its scope, computed in a single combined
/// forward-backward pass -- Eq. 2, with `x`'s own message excluded from the product (its own
/// layer's edges carry only structural existence; everything else is already folded into
/// `alpha`/`beta` via every *other* layer's weighting). Returned as one probability vector per
/// layer, indexed the same way as `mdd.decision_at_layer` (and the same way `messages` itself is
/// indexed -- by layer within this MDD, not by global variable id).
fn mdd_local_beliefs(mdd: &Mdd, messages: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let alpha = unclamped_alpha(mdd, messages);
    let beta = unclamped_beta(mdd, messages);
    let last_layer = mdd.sink().0;

    (0..last_layer)
        .map(|layer| {
            let domain_size = messages[layer].len();
            let mut weights = vec![0.0; domain_size];
            for node in mdd.nodes_in_layer(layer) {
                let mass = alpha[layer][node.1];
                if mass == 0.0 {
                    continue;
                }
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    weights[value.0] += mass * beta[layer + 1][mdd[edge].to().1];
                }
            }
            normalize_or_uniform(weights, domain_size)
        })
        .collect()
}

/// Eq. 1's cavity message: `marginal` with `local_belief` (that same constraint's own previous
/// contribution) divided back out, in log-space via `safe_ln` so a genuinely zero local belief
/// doesn't produce infinities -- it just leaves that value's message at (numerically) zero, same
/// convention as the rest of this module's WMC machinery.
fn cavity_message(marginal: &[f64], local_belief: &[f64]) -> Vec<f64> {
    let weights: Vec<f64> = marginal
        .iter()
        .zip(local_belief)
        .map(|(&m, &l)| (safe_ln(m) - safe_ln(l)).exp())
        .collect();
    normalize_or_uniform(weights, marginal.len())
}

/// Loopy belief propagation over `mdds`, seeded from and permanently anchored to `probs` (see the
/// module doc). Runs a fixed `iterations` rounds, synchronous/flooding-style -- every constraint's
/// local beliefs this round come from messages computed last round, matching Pesant's Algorithm 5 --
/// with no convergence check: this is meant to run a handful of iterations inside search, and
/// Pesant's own empirical results (Section 6) show a few iterations already capture most of the
/// benefit, with KL-divergence to the true marginals sometimes *increasing* again after many more.
///
/// No per-MDD weights: every constraint counts equally, matching the paper's plain (unweighted)
/// formulation -- `MddSampler::set_weight` is a separate, destroy-heuristic-specific knob and isn't
/// reused here.
///
/// `assignment`/`decided` supply hard evidence exactly the way `partial_alpha_at`/`partial_beta_at`
/// do: a `decided` variable is clamped to `assignment`'s value for the *entire* run -- its outgoing
/// message to every MDD is a permanent one-hot rather than something derived from a cavity message,
/// and it never receives an updated marginal from the MDDs' local beliefs about it (its returned
/// marginal is just that same one-hot). This is what lets belief propagation be dropped into a
/// destroy/repair loop: the variables *not* being resampled this round are folded in as fixed
/// context for the ones that are, instead of every call re-solving the whole board from scratch as
/// if nothing were already decided. An all-`false` `decided` recovers the plain, evidence-free
/// marginal (nothing clamped, same as this function's first version).
///
/// Returns one combined marginal per variable (`probs[var]`'s one-hot at `assignment[var]` for a
/// `decided` variable; `probs[var]` itself, unchanged, for an undecided one that belongs to no MDD
/// or when `iterations == 0`).
pub fn belief_propagation(
    mdds: &[Mdd],
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
    iterations: usize,
) -> Vec<Vec<f64>> {
    let var_to_mdds = build_var_to_mdds(mdds);
    let num_vars = var_to_mdds.len();

    // The permanent prior for each variable: the network's own belief for one that's still free,
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
            (0..mdd.number_layers() - 1)
                .map(|layer| priors[mdd.decision_at_layer(layer).0].clone())
                .collect()
        })
        .collect();

    // local_beliefs[mdd_index][layer], carried across iterations so next round's cavity message
    // can divide this round's local belief back out.
    let mut local_beliefs: Vec<Vec<Vec<f64>>> = mdds
        .iter()
        .map(|mdd| {
            (0..mdd.number_layers() - 1)
                .map(|layer| vec![1.0; probs[mdd.decision_at_layer(layer).0].len()])
                .collect()
        })
        .collect();

    let mut marginal: Vec<Vec<f64>> = priors.clone();

    for _ in 0..iterations {
        // Every constraint computes its local beliefs for every variable in its scope from this
        // round's messages, in one combined forward-backward pass each (Algorithm 4's
        // `updateBelief`).
        for (mdd_index, mdd) in mdds.iter().enumerate() {
            local_beliefs[mdd_index] = mdd_local_beliefs(mdd, &messages[mdd_index]);
        }

        // Reset every variable's marginal to its prior, then accumulate every MDD's local belief
        // about it in log-space (Algorithm 1's `receiveMessage`, applied once per MDD membership)
        // -- except for a `decided` variable, which stays exactly at its one-hot prior: it's fixed
        // evidence, not something for the MDDs to keep refining.
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
        marginal = log_marginal
            .into_iter()
            .map(log_combine_and_normalize)
            .collect();

        // Every variable sends each of its constraints a fresh cavity message for next round
        // (Eq. 1), built from this round's marginal and this round's local belief -- except a
        // `decided` variable, whose outgoing message is always its fixed one-hot prior rather than
        // a cavity message: dividing a one-hot marginal by a constraint's local belief could hit a
        // genuine zero there (evidence that's locally infeasible under this MDD given some *other*
        // decided variable) and blow up `cavity_message`'s `safe_ln` division for no benefit -- a
        // decided variable's outgoing message doesn't depend on what any MDD currently believes
        // about it.
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
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::modelling::{ConstraintIndex, Problem, gcc, not_equals};
    use std::sync::Arc;

    /// Same helper as `super::tests::build_mdd` -- duplicated locally rather than exposed across
    /// the two test modules, to keep each self-contained.
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

    #[test]
    fn belief_propagation_indexes_messages_by_mdd_layer_not_by_global_variable_id() {
        // Regression test for a real out-of-bounds panic hit on Sudoku: `w` sits *before* `x` and
        // `y` in the problem's global variable numbering (mirrors
        // `super::super::tests::destroyed_variables_with_deterministic_rule_destroys_exactly_the_violated_mdds`'s
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
    fn single_not_equals_mdd_with_symmetric_conflicting_prior_becomes_uniform_after_one_iteration() {
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
            let mdds: Vec<Mdd> = (0..count)
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
                let marginals = belief_propagation(&mdds, &probs, &assignment, &decided, iterations);
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
}
