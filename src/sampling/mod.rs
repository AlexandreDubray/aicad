use crate::mdd::Mdd;
use crate::modelling::{ValueIndex, VariableIndex};

use rand::seq::SliceRandom;
use rand::RngExt;

/// Floor applied to the probabilities to avoid undefined behavior
const MIN_PROB: f64 = 1e-9;

/// How `GibbsSampler::sweep` turns a variable's combined conditional distribution into a value.
#[derive(Clone, Copy, Debug)]
pub enum DecodeMode {
    /// Draw a value proportionally to the combined distribution
    Sample,
    /// Take the most likely value
    Greedy,
}

/// Runs the forward (WMC computation) pass over the MDD until a target layer.
/// The forward pass must be consistent with the given assignment; computing the WMC flow only
/// through the edges associated with the assignment.
/// This correspond to a path from the root to the target layer because each node has exactly one
/// outgoing edge per label. Hence, only one node in target_layer has a non-zero forward
/// probability mass.
///
/// Returns: A vector of the same size as the target layer. Each entry correspond to the
///          probability of reaching the associated node from the root, consistently with the assignment.
fn clamped_alpha_at(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
) -> Vec<f64> {
    let n = mdd.number_nodes_in_layer(target_layer);
    let mut alpha: Vec<f64> = vec![0.0; n];
    let mut path_probability = 1.0;
    let mut current_node = mdd.root();

    for layer in 0..target_layer {
        let variable = mdd.decision_at_layer(layer);
        let clamp_value = assignment[variable.0];

        for edge in mdd[current_node].iter_children() {
            let value = mdd[edge].assignment();
            if value == clamp_value {
                // We found the edge, we follow it
                path_probability = path_probability * probs[variable.0][value.0].max(MIN_PROB);
                current_node = mdd[edge].to();
                break;
            }
        }
        // If the current_node has not been updated it is still on the current_layer. That means
        // that no paths exist consistent with the current assignment, forward pass is 0.0
        if current_node.0 == layer {
            path_probability = 0.0;
            break;
        }
    }
    if current_node.0 == target_layer {
        alpha[current_node.1] = path_probability;
    }

    alpha
}

/// Runs the backward (WMC computation, in reverse order) pass over the MDD until a target layer.
/// The backward pass must be consistent with the given assignment; computing the WMC flow only
/// through the edges associated with the assignment.
///
/// Returns: A vector of the same size as the target layer. Each entry correspond to the
///          probability of reaching the associated node from the sink, consistently with the assignment.
fn clamped_beta_at(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
) -> Vec<f64> {
    // Sink node has always a probability of 1.0 being reached
    let mut beta: Vec<f64> = vec![1.0];
    let last_layer = mdd.sink().0;

    // Iterates from the sink layer up to the target layer
    for layer in ((target_layer + 1)..last_layer + 1).rev() {
        // Compute the backward WMC from the prev_layer by iterating on the children
        let prev_layer = layer - 1;
        let variable = mdd.decision_at_layer(prev_layer);
        let clamp_value = assignment[variable.0];

        let mut prev_beta = vec![0.0; mdd.number_nodes_in_layer(prev_layer)];
        for node in mdd.nodes_in_layer(prev_layer) {
            let mut mass = 0.0;
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                // Skip edges inconsistent with the assignment
                if value != clamp_value {
                    continue;
                }
                let prob = probs[variable.0][value.0].max(MIN_PROB);
                // Multiply the edge probability with the backward WMC of the child
                mass += prob * beta[mdd[edge].to().1];
            }
            prev_beta[node.1] = mass;
        }
        beta = prev_beta;
    }
    beta
}

/// Computes the distribution of the variable at target_layer conditionned on the MDD structure and
/// the assignment.
pub fn clamped_conditional(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
) -> Vec<f64> {
    let variable = mdd.decision_at_layer(target_layer);
    let domain_size = probs[variable.0].len();

    // Computes both forward and backward probability mass of the nodes in the layer.
    // The alpha vector gives for each node its probability of reaching it from the source
    // The beta vector gives for each node its probability of reaching it from the sink
    let alpha = clamped_alpha_at(mdd, target_layer, probs, assignment);
    let beta = clamped_beta_at(mdd, target_layer + 1, probs, assignment);

    // Probability distribution over the domain
    let mut weights = vec![0.0; domain_size];
    for node in mdd.nodes_in_layer(target_layer) {
        let mass = alpha[node.1];
        if mass == 0.0 {
            continue;
        }
        for edge in mdd[node].iter_children() {
            let value = mdd[edge].assignment();
            let prob = probs[variable.0][value.0].max(MIN_PROB);
            // We accumulate the probability of reaching the node, selecting the edge, and
            // extending the partial assignment (with the edge) into a full solution
            weights[value.0] += mass * prob * beta[mdd[edge].to().1];
        }
    }

    normalize_or_uniform(weights, domain_size)
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

/// Unconditional (i.e. not clamped to any assignment) forward/backward WMC pass over `mdd`, given
/// only `probs`. `alpha[layer][node]`/`beta[layer][node]` are indexed the same way
/// `mdd.nodes_in_layer(layer)` enumerates that layer's nodes.
fn unconditional_alpha_beta(mdd: &Mdd, probs: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let last_layer = mdd.sink().0;

    let mut alpha: Vec<Vec<f64>> = Vec::with_capacity(last_layer + 1);
    alpha.push(vec![1.0; mdd.number_nodes_in_layer(0)]);
    for layer in 0..last_layer {
        let variable = mdd.decision_at_layer(layer);
        let mut next_alpha = vec![0.0; mdd.number_nodes_in_layer(layer + 1)];
        for node in mdd.nodes_in_layer(layer) {
            let mass = alpha[layer][node.1];
            if mass == 0.0 {
                continue;
            }
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let prob = probs[variable.0][value.0].max(MIN_PROB);
                next_alpha[mdd[edge].to().1] += mass * prob;
            }
        }
        alpha.push(next_alpha);
    }

    let mut beta: Vec<Vec<f64>> = vec![Vec::new(); last_layer + 1];
    beta[last_layer] = vec![1.0; mdd.number_nodes_in_layer(last_layer)];
    for layer in (0..last_layer).rev() {
        let variable = mdd.decision_at_layer(layer);
        let mut prev_beta = vec![0.0; mdd.number_nodes_in_layer(layer)];
        for node in mdd.nodes_in_layer(layer) {
            let mut mass = 0.0;
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let prob = probs[variable.0][value.0].max(MIN_PROB);
                mass += prob * beta[layer + 1][mdd[edge].to().1];
            }
            prev_beta[node.1] = mass;
        }
        beta[layer] = prev_beta;
    }

    (alpha, beta)
}

/// Unconditional per-value marginal of every decision layer of `mdd`, given only `probs` -- no
/// assignment is clamped anywhere, so this can't be blinded by an inconsistency elsewhere in the
/// MDD's scope the way `clamped_conditional` can. Returned in layer order, `result[layer]` has
/// length `probs[mdd.decision_at_layer(layer).0].len()`.
fn mdd_marginals(mdd: &Mdd, probs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let last_layer = mdd.sink().0;
    let (alpha, beta) = unconditional_alpha_beta(mdd, probs);

    (0..last_layer)
        .map(|layer| {
            let variable = mdd.decision_at_layer(layer);
            let domain_size = probs[variable.0].len();
            let mut weights = vec![0.0; domain_size];
            for node in mdd.nodes_in_layer(layer) {
                let mass = alpha[layer][node.1];
                if mass == 0.0 {
                    continue;
                }
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let prob = probs[variable.0][value.0].max(MIN_PROB);
                    weights[value.0] += mass * prob * beta[layer + 1][mdd[edge].to().1];
                }
            }
            normalize_or_uniform(weights, domain_size)
        })
        .collect()
}

/// Sampler used for generating a new assignment given a set of MDDs; aggregate multiple MDDs
/// opinion about variables update
pub struct GibbsSampler<'a> {
    mdds: &'a [Mdd],
    /// Scaling factor for each MDD choice, default to 1.0 (each MDD has equal voting power)
    weights: Vec<f64>,
    /// For each global variable, every `(mdd_index, layer)` pair where that MDD has this variable
    /// in its scope, at that layer.
    var_to_mdds: Vec<Vec<(usize, usize)>>,
}

impl<'a> GibbsSampler<'a> {
    pub fn new(mdds: &'a [Mdd]) -> Self {
        let num_vars = mdds
            .first()
            .map(|mdd| mdd.problem().number_variables())
            .unwrap_or(0);

        let mut var_to_mdds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_vars];
        for (mdd_index, mdd) in mdds.iter().enumerate() {
            for layer in 0..mdd.number_layers() - 1 {
                let variable = mdd.decision_at_layer(layer);
                var_to_mdds[variable.0].push((mdd_index, layer));
            }
        }

        Self {
            mdds,
            weights: vec![1.0; mdds.len()],
            var_to_mdds,
        }
    }

    /// Sets the voting weight of the MDD at mdd_index
    pub fn set_weight(&mut self, mdd_index: usize, weight: f64) {
        self.weights[mdd_index] = weight;
    }

    pub fn members_of(&self, var: VariableIndex) -> &[(usize, usize)] {
        let members = &self.var_to_mdds[var.0];
        assert!(
            !members.is_empty(),
            "variable {} is not in the scope of any MDD -- every variable must appear in at \
             least one for GibbsSampler to have anything to decode it from",
            var.0
        );
        members
    }

    /// Combines the all probability distribution for a given variable. These probability
    /// distribution are conditionned on the MDDs and the current assignment. The combination is a
    /// weighted sum of each MDD conditional distribution, renormalised to sum to 1.0
    pub fn combined_conditional(
        &self,
        var: VariableIndex,
        probs: &[Vec<f64>],
        assignment: &[ValueIndex],
    ) -> Vec<f64> {
        let members = self.members_of(var);
        let domain_size = probs[var.0].len();
        let mut log_combined = vec![0.0; domain_size];
        for &(mdd_index, layer) in members {
            let weight = self.weights[mdd_index];
            let conditional = clamped_conditional(&self.mdds[mdd_index], layer, probs, assignment);
            for (d, log_d) in log_combined.iter_mut().enumerate() {
                *log_d += weight * conditional[d].max(MIN_PROB).ln();
            }
        }
        log_combine_and_normalize(log_combined)
    }

    /// Same combination as `combined_conditional`, but every MDD's contribution is its
    /// unconditional marginal (`mdd_marginals`) rather than a conditional clamped to `assignment`.
    pub fn combined_marginal(
        &self,
        var: VariableIndex,
        probs: &[Vec<f64>],
        marginals: &[Vec<Vec<f64>>],
    ) -> Vec<f64> {
        let members = self.members_of(var);
        let domain_size = probs[var.0].len();
        let mut log_combined = vec![0.0; domain_size];
        for &(mdd_index, layer) in members {
            let weight = self.weights[mdd_index];
            let marginal = &marginals[mdd_index][layer];
            for (d, log_d) in log_combined.iter_mut().enumerate() {
                *log_d += weight * marginal[d].max(MIN_PROB).ln();
            }
        }
        log_combine_and_normalize(log_combined)
    }

    /// `mdd_marginals` for every MDD this sampler holds, given `probs`. Computed once per call
    /// (each MDD is a single forward+backward pass over every layer at once) rather than per
    /// queried variable, since -- unlike `clamped_conditional` -- the result doesn't depend on any
    /// assignment.
    pub fn unconditional_marginals(&self, probs: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        self.mdds
            .iter()
            .map(|mdd| mdd_marginals(mdd, probs))
            .collect()
    }

    /// Perform a number of gibbs sampling given the probability and assignment. From an initial
    /// assignment and a number of variable to update, perform `rounds` sampling steps. Each
    /// sampling step, variables are considered in a random order. Each variable is then sampled
    /// from its combined probability distribution conditioned on the current assignment (including
    /// replaced variables) and the MDD it appears in.
    pub fn sweep(
        &self,
        probs: &[Vec<f64>],
        assignment: &mut [ValueIndex],
        order: &[VariableIndex],
        rounds: usize,
        mode: DecodeMode,
    ) {
        let mut round_order: Vec<VariableIndex> = order.to_vec();
        for _ in 0..rounds {
            crate::utils::with_rng(|rng| round_order.shuffle(rng));
            for &var in &round_order {
                let combined = self.combined_conditional(var, probs, assignment);
                assignment[var.0] = ValueIndex(match mode {
                    DecodeMode::Greedy => argmax(&combined),
                    DecodeMode::Sample => sample_categorical(&combined),
                });
            }
        }
    }

    /// Resamples `block` (typically the scope of a small set of destroyed constraints) in two
    /// stages: every variable in `block` is first drawn independently from `combined_marginal`
    /// (unconditional, so a stale/inconsistent value elsewhere in `assignment` can't blind it),
    /// then -- if `cleanup_rounds > 0` -- `sweep` runs `cleanup_rounds` clamped rounds restricted
    /// to `block`, seeded from that draw, to resolve any joint inconsistency the independent draw
    /// left behind (two block variables of the same MDD can still collide, since marginals don't
    /// carry the joint's correlations the way a clamped conditional does).
    pub fn resample_block(
        &self,
        probs: &[Vec<f64>],
        assignment: &mut [ValueIndex],
        block: &[VariableIndex],
        mode: DecodeMode,
        cleanup_rounds: usize,
    ) {
        let marginals = self.unconditional_marginals(probs);
        for &var in block {
            let combined = self.combined_marginal(var, probs, &marginals);
            assignment[var.0] = ValueIndex(match mode {
                DecodeMode::Greedy => argmax(&combined),
                DecodeMode::Sample => sample_categorical(&combined),
            });
        }

        if cleanup_rounds > 0 {
            self.sweep(probs, assignment, block, cleanup_rounds, mode);
        }
    }
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

fn argmax(weights: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("weights should never be NaN"))
        .map(|(index, _)| index)
        .expect("weights should not be empty")
}

fn sample_categorical(weights: &[f64]) -> usize {
    crate::utils::with_rng(|rng| {
        let total: f64 = weights.iter().sum();
        let mut target = rng.random_range(0.0..total);
        for (index, &w) in weights.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                return index;
            }
        }
        weights.len() - 1
    })
}

/// Shannon entropy of a discrete distribution.
pub fn entropy(dist: &[f64]) -> f64 {
    dist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::modelling::{gcc, not_equals, ConstraintIndex, Problem};
    use std::sync::Arc;

    /// Builds an exact MDD for `constraints` over `problem`. `Mdd::new` alone only produces the
    /// initial (possibly relaxed -- nodes merged, so it can admit solutions that don't actually
    /// satisfy every constraint) MDD; `refine(usize::MAX)` -- the same call `mdd_refine` in
    /// `crate::mdd::mdd::test_mdd` makes, just with an unbounded width so it runs to full exactness
    /// rather than stopping at some node budget -- splits every relaxed node back apart until the
    /// MDD exactly represents the constraints' solution set. The sampling module's correctness
    /// tests need that exactness: they assert on precise conditional probabilities that only hold
    /// for the exact MDD.
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

    /// Uniform probability matrix: every value of every variable equally likely a priori. Useful
    /// as a baseline so a test's assertions are driven purely by the MDD's combinatorial structure
    /// (which paths exist), not by any particular network output.
    fn uniform_probs(problem: &Problem) -> Vec<Vec<f64>> {
        (0..problem.number_variables())
            .map(|v| {
                let domain_size = problem[VariableIndex(v)].domain_size();
                vec![1.0 / domain_size as f64; domain_size]
            })
            .collect()
    }

    #[test]
    fn clamped_conditional_matches_uniform_marginal_on_free_variable() {
        // x != y over {0,1} x {0,1}: only 2 of the 4 combinations are valid. Clamping y=0 should
        // put all conditional mass on x=1 (the only value making x != y hold), regardless of the
        // (uniform) input probabilities.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);

        // Find which layer decides y, clamp it to value 0, and query x's layer.
        let y_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == y)
            .expect("y must be in the MDD's scope");
        let x_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == x)
            .expect("x must be in the MDD's scope");

        let mut assignment = vec![ValueIndex(0); problem.number_variables()];
        assignment[y.0] = ValueIndex(0);

        let conditional = clamped_conditional(&mdd, x_layer, &probs, &assignment);
        assert_eq!(conditional.len(), 2);
        assert!((conditional[0] - 0.0).abs() < 1e-9, "{conditional:?}");
        assert!((conditional[1] - 1.0).abs() < 1e-9, "{conditional:?}");

        // And the reverse: clamping x should push all mass onto the layer deciding y.
        let mut assignment2 = vec![ValueIndex(0); problem.number_variables()];
        assignment2[x.0] = ValueIndex(1);
        let conditional_y = clamped_conditional(&mdd, y_layer, &probs, &assignment2);
        assert!((conditional_y[0] - 1.0).abs() < 1e-9, "{conditional_y:?}");
        assert!((conditional_y[1] - 0.0).abs() < 1e-9, "{conditional_y:?}");
    }

    #[test]
    fn clamped_conditional_abstains_uniformly_when_locally_infeasible() {
        // x != y, y != z, clamped so that x and y are pinned to the same (infeasible) value: no
        // path through the MDD survives, so the conditional over z's layer should come back
        // uniform rather than panicking or returning all-zero.
        //
        // All three variables share the same domain here (unlike the pairwise-not_equals trio in
        // `gibbs_sampler_single_mdd_greedy_sweep_is_deterministic_and_feasible`, which mirrors
        // `mdd::mdd::tests::mdd_refine`'s specific domain choice): `NotEquals`'s per-constraint
        // `val_to_bit` map only covers the union of *its own* two variables' domains, and
        // `Mdd::fold_property_over_parents` unconditionally looks up every layer's assigned value
        // in every constraint's map (even ones the layer's variable isn't part of, guarded only
        // *after* the lookup by `in_scope`) -- so a layer whose variable's domain isn't a subset
        // of every other constraint's own map panics on `.unwrap()` before `in_scope` ever gets
        // checked. That's a latent bug in `NotEqualsProperty::update`, not in this module; using
        // matching domains here sidesteps it rather than papering over it.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);

        let z_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == z)
            .expect("z must be in the MDD's scope");

        // x = y = 0 is infeasible under not_equals(x, y).
        let mut assignment = vec![ValueIndex(0); problem.number_variables()];
        assignment[x.0] = ValueIndex(0);
        assignment[y.0] = ValueIndex(0);
        assignment[z.0] = ValueIndex(0);

        let conditional = clamped_conditional(&mdd, z_layer, &probs, &assignment);
        assert_eq!(conditional.len(), 3);
        for p in &conditional {
            assert!((p - 1.0 / 3.0).abs() < 1e-9, "{conditional:?}");
        }
    }

    #[test]
    fn gibbs_sampler_single_mdd_greedy_sweep_is_feasible_regardless_of_round_count() {
        // A single MDD spanning all variables: greedy sweeps (with `sweep` free to shuffle the
        // visiting order every round) should always land on a feasible assignment (no constraint
        // violated), since only edges that exist in the MDD are ever chosen, regardless of how
        // many rounds run or what order each round visits variables in.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);
        let mdds = vec![mdd];

        let sampler = GibbsSampler::new(&mdds);
        let order: Vec<VariableIndex> = (0..mdds[0].number_layers() - 1)
            .map(|l| mdds[0].decision_at_layer(l))
            .collect();

        for rounds in [1, 4, 10] {
            let mut assignment = vec![ValueIndex(0); problem.number_variables()];
            sampler.sweep(&probs, &mut assignment, &order, rounds, DecodeMode::Greedy);

            // Feasibility check: every solution the MDD encodes satisfies all three not_equals
            // constraints by construction, and `sweep` only ever follows existing MDD edges, so
            // the resulting assignment must too, no matter how many (shuffled) rounds ran.
            let xv = problem[x].value(assignment[x.0]);
            let yv = problem[y].value(assignment[y.0]);
            let zv = problem[z].value(assignment[z.0]);
            assert_ne!(xv, yv, "rounds={rounds}");
            assert_ne!(yv, zv, "rounds={rounds}");
            assert_ne!(xv, zv, "rounds={rounds}");
        }
    }

    #[test]
    fn sweep_with_zero_rounds_is_a_no_op() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);
        let mdds = vec![mdd];
        let sampler = GibbsSampler::new(&mdds);

        let order = vec![x, y];
        let mut assignment = vec![ValueIndex(1); problem.number_variables()];
        let before = assignment.clone();
        sampler.sweep(&probs, &mut assignment, &order, 0, DecodeMode::Greedy);
        assert_eq!(assignment, before);
    }

    #[test]
    fn gibbs_sampler_combines_multiple_mdds_via_product_of_experts() {
        // Two single-constraint MDDs over the same 2 variables, deliberately with opposite
        // "orders" (each only has one decision layer, so order isn't the point here -- the point
        // is that neither MDD alone determines y, but their conjunction does): x != y and,
        // separately, y != 0 encoded as a gcc bound. Combined, the only valid combos are the ones
        // satisfying both, so a greedy sweep starting from x should pick a y consistent with both.
        // Two independently-built `Problem`s over the same variable layout (`Problem` isn't
        // `Clone`, and each `Mdd` owns its own `Arc<Problem>` anyway -- `GibbsSampler` only
        // requires that every MDD passed to it agree on what a `VariableIndex` means, not that
        // they share the same `Arc`).
        let mut problem_a = Problem::default();
        let x = problem_a.add_variable(vec![0, 1], None);
        let y = problem_a.add_variable(vec![0, 1], None);
        not_equals(&mut problem_a, x, y);
        let problem_a = Arc::new(problem_a);
        let constraints_a: Vec<ConstraintIndex> = problem_a.iter_constraints().collect();
        let mdd_a = build_mdd(problem_a.clone(), &constraints_a);

        // Unbound gcc over just y stands in for "no real extra restriction" (see mdd_creation's
        // comment) while still giving a second MDD whose scope includes y, so `var_to_mdds[y]`
        // legitimately has two entries to combine.
        let mut problem_b = Problem::default();
        let _x_b = problem_b.add_variable(vec![0, 1], None);
        let y_b = problem_b.add_variable(vec![0, 1], None);
        gcc(&mut problem_b, vec![y_b], vec![]);
        let problem_b = Arc::new(problem_b);
        let constraints_b: Vec<ConstraintIndex> = problem_b.iter_constraints().collect();
        let mdd_b = build_mdd(problem_b.clone(), &constraints_b);

        let probs = uniform_probs(&problem_a);
        let mdds = vec![mdd_a, mdd_b];
        let sampler = GibbsSampler::new(&mdds);

        let mut assignment = vec![ValueIndex(0); problem_a.number_variables()];
        assignment[x.0] = ValueIndex(1);
        let combined = sampler.combined_conditional(y, &probs, &assignment);
        // Only not_equals(x, y) actually constrains y here; with x=1, y=0 is the only option.
        // `mdd_b`'s unbound gcc contributes a uniform (non-opinionated) conditional, and
        // `MIN_PROB` keeps every probability this module works with strictly positive, so the
        // combined result approaches but never exactly reaches [1.0, 0.0] -- tolerance is
        // loosened accordingly (still tight enough to catch a real logic error).
        assert!((combined[0] - 1.0).abs() < 1e-6, "{combined:?}");
        assert!((combined[1] - 0.0).abs() < 1e-6, "{combined:?}");
    }

    #[test]
    fn mdd_marginals_matches_brute_force_marginal_for_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);

        let probs = vec![vec![0.9, 0.1], vec![0.3, 0.7]];
        let marginals = mdd_marginals(&mdd, &probs);

        let x_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == x)
            .unwrap();
        let y_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == y)
            .unwrap();

        // P(x=0) ~ probs_x[0]*probs_y[1], P(x=1) ~ probs_x[1]*probs_y[0], normalised; and
        // symmetrically for y. Hand-computed from probs = [[0.9, 0.1], [0.3, 0.7]].
        assert!(
            (marginals[x_layer][0] - 21.0 / 22.0).abs() < 1e-9,
            "{marginals:?}"
        );
        assert!(
            (marginals[x_layer][1] - 1.0 / 22.0).abs() < 1e-9,
            "{marginals:?}"
        );
        assert!(
            (marginals[y_layer][0] - 1.0 / 22.0).abs() < 1e-9,
            "{marginals:?}"
        );
        assert!(
            (marginals[y_layer][1] - 21.0 / 22.0).abs() < 1e-9,
            "{marginals:?}"
        );
    }

    #[test]
    fn resample_block_with_zero_cleanup_rounds_only_samples_from_marginals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);
        let mdds = vec![mdd];
        let sampler = GibbsSampler::new(&mdds);

        let block = vec![x, y];
        let marginals = sampler.unconditional_marginals(&probs);
        let expected_x = argmax(&sampler.combined_marginal(x, &probs, &marginals));
        let expected_y = argmax(&sampler.combined_marginal(y, &probs, &marginals));

        let mut assignment = vec![ValueIndex(0); problem.number_variables()];
        sampler.resample_block(&probs, &mut assignment, &block, DecodeMode::Greedy, 0);

        assert_eq!(assignment[x.0], ValueIndex(expected_x));
        assert_eq!(assignment[y.0], ValueIndex(expected_y));
    }

    #[test]
    fn resample_block_with_cleanup_rounds_is_feasible_from_a_colliding_start() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);
        let mdds = vec![mdd];
        let sampler = GibbsSampler::new(&mdds);

        let block: Vec<VariableIndex> = (0..mdds[0].number_layers() - 1)
            .map(|l| mdds[0].decision_at_layer(l))
            .collect();

        for cleanup_rounds in [1, 4] {
            let mut assignment = vec![ValueIndex(0); problem.number_variables()];
            sampler.resample_block(
                &probs,
                &mut assignment,
                &block,
                DecodeMode::Greedy,
                cleanup_rounds,
            );

            let xv = problem[x].value(assignment[x.0]);
            let yv = problem[y].value(assignment[y.0]);
            let zv = problem[z].value(assignment[z.0]);
            assert_ne!(xv, yv, "cleanup_rounds={cleanup_rounds}");
            assert_ne!(yv, zv, "cleanup_rounds={cleanup_rounds}");
            assert_ne!(xv, zv, "cleanup_rounds={cleanup_rounds}");
        }
    }

    #[test]
    fn sample_categorical_respects_zero_weight_never_selected() {
        // Not a statistical test (no fixed seed guarantee here) -- just checks that a value with
        // zero weight is structurally unreachable given how `sample_categorical` walks the
        // cumulative distribution, run enough times to be confident about it.
        let weights = vec![0.0, 1.0, 0.0];
        for _ in 0..200 {
            assert_eq!(sample_categorical(&weights), 1);
        }
    }

    #[test]
    fn argmax_picks_the_largest_weight() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.9, 0.05, 0.05]), 0);
    }

    #[test]
    fn entropy_of_a_deterministic_distribution_is_zero() {
        assert_eq!(entropy(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn entropy_of_a_uniform_distribution_matches_ln_n() {
        let dist = vec![0.25; 4];
        assert!((entropy(&dist) - 4.0f64.ln()).abs() < 1e-12);
    }
}
