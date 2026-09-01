pub mod bp;
pub mod solve;

use crate::mdd::Mdd;
use crate::modelling::{ValueIndex, VariableIndex};

use rand::RngExt;

const LOG_ZERO: f64 = -745.0;

fn safe_ln(p: f64) -> f64 {
    if p > 0.0 {
        p.ln()
    } else {
        LOG_ZERO
    }
}

/// How a variable's combined distribution turns into a value: `sample_categorical` or `argmax`.
#[derive(Clone, Copy, Debug)]
pub enum DecodeMode {
    /// Draw a value proportionally to the combined distribution
    Sample,
    /// Take the most likely value
    Greedy,
}

/// Generalises the forward (WMC) pass over `mdd` up to `target_layer`: at a layer whose variable
/// is `decided`, follows only the edge matching `assignment`'s current value for it. At a layer
/// whose variable is not yet `decided`, sums over every outgoing edge instead.
fn partial_alpha_at(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
) -> Vec<f64> {
    let mut alpha: Vec<f64> = vec![1.0];

    for layer in 0..target_layer {
        let variable = mdd.decision_at_layer(layer);
        let mut next_alpha = vec![0.0; mdd.number_nodes_in_layer(layer + 1)];
        for node in mdd.nodes_in_layer(layer) {
            let mass = alpha[node.1];
            if mass == 0.0 {
                continue;
            }
            if decided[variable.0] {
                let clamp_value = assignment[variable.0];
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    if value == clamp_value {
                        let prob = probs[variable.0][value.0];
                        next_alpha[mdd[edge].to().1] += mass * prob;
                        break;
                    }
                }
            } else {
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let prob = probs[variable.0][value.0];
                    next_alpha[mdd[edge].to().1] += mass * prob;
                }
            }
        }
        alpha = next_alpha;
    }

    alpha
}

/// The backward counterpart of `partial_alpha_at`: generalises the backward (WMC) pass over `mdd`
/// down to `target_layer`, clamping a `decided` layer's variable to its assigned value and summing
/// over every edge at an undecided one.
fn partial_beta_at(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
) -> Vec<f64> {
    let last_layer = mdd.sink().0;
    let mut beta: Vec<f64> = vec![1.0; mdd.number_nodes_in_layer(last_layer)];

    for layer in (target_layer..last_layer).rev() {
        let variable = mdd.decision_at_layer(layer);
        let mut prev_beta = vec![0.0; mdd.number_nodes_in_layer(layer)];
        for node in mdd.nodes_in_layer(layer) {
            let mut mass = 0.0;
            if decided[variable.0] {
                let clamp_value = assignment[variable.0];
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    if value == clamp_value {
                        let prob = probs[variable.0][value.0];
                        mass += prob * beta[mdd[edge].to().1];
                        break;
                    }
                }
            } else {
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let prob = probs[variable.0][value.0];
                    mass += prob * beta[mdd[edge].to().1];
                }
            }
            prev_beta[node.1] = mass;
        }
        beta = prev_beta;
    }

    beta
}

/// Computes the distribution of the variable at `target_layer`, conditioned on the MDD structure
/// and whichever evidence `decided` supplies from `assignment` -- an undecided variable elsewhere
/// in the MDD's scope is marginalised out.
fn partial_conditional(
    mdd: &Mdd,
    target_layer: usize,
    probs: &[Vec<f64>],
    assignment: &[ValueIndex],
    decided: &[bool],
) -> Vec<f64> {
    let variable = mdd.decision_at_layer(target_layer);
    let domain_size = probs[variable.0].len();

    let alpha = partial_alpha_at(mdd, target_layer, probs, assignment, decided);
    let beta = partial_beta_at(mdd, target_layer + 1, probs, assignment, decided);

    let mut weights = vec![0.0; domain_size];
    for node in mdd.nodes_in_layer(target_layer) {
        let mass = alpha[node.1];
        if mass == 0.0 {
            continue;
        }
        for edge in mdd[node].iter_children() {
            let value = mdd[edge].assignment();
            let prob = probs[variable.0][value.0];
            weights[value.0] += mass * prob * beta[mdd[edge].to().1];
        }
    }

    normalize_or_uniform(weights, domain_size)
}

/// Walks the single path `assignment` traces through `mdd` (root to sink, following `assignment`'s
/// value at each layer), returning the `(variable, value)` decisions actually taken if that path
/// exists, or `None` the moment some layer has no edge matching `assignment`'s value there -- i.e.
/// `assignment` violates the constraint(s) `mdd` represents. `Mdd::refine` builds an *exact* MDD
/// (see `compile_problem_mdds`), so this agrees exactly with checking every one of
/// `mdd.iter_constraints()` against the raw assignment; it's simply cheaper, walking the compiled
/// structure directly instead of needing `assignment` converted back to raw domain values first.
/// Shared by `mdd_accepts` (does the path exist at all) and `mdd_satisfaction_probability` (how
/// much does the network endorse it), so the walk itself isn't duplicated between them.
fn mdd_walk(mdd: &Mdd, assignment: &[ValueIndex]) -> Option<Vec<(VariableIndex, ValueIndex)>> {
    let mut node = mdd.root();
    let mut path = Vec::with_capacity(mdd.sink().0);

    for layer in 0..mdd.sink().0 {
        let variable = mdd.decision_at_layer(layer);
        let clamp_value = assignment[variable.0];
        let edge = mdd[node]
            .iter_children()
            .find(|&edge| mdd[edge].assignment() == clamp_value)?;
        path.push((variable, clamp_value));
        node = mdd[edge].to();
    }

    Some(path)
}

/// Whether `assignment` satisfies the constraint(s) `mdd` represents -- see `mdd_walk`. Used by
/// `DestroyRule::Deterministic`.
fn mdd_accepts(mdd: &Mdd, assignment: &[ValueIndex]) -> bool {
    mdd_walk(mdd, assignment).is_some()
}

/// How much the network's own beliefs endorse `assignment`'s current values for `mdd`'s scope:
/// `mdd_walk`'s path, if it exists, weighted by the *geometric* mean of `probs[var][value]` over
/// the (variable, value) pairs on it -- 1.0 only if the network is fully confident in every one of
/// those specific choices, pulled down by any variable whose chosen value it isn't sure about, even
/// though the assignment is satisfying overall. Exactly 0 if the path doesn't exist at all, i.e.
/// `assignment` violates the constraint(s) `mdd` represents. Used by `DestroyRule::Probabilistic`.
///
/// Geometric mean, not the raw product (equivalent to `partial_alpha_at`'s `alpha[sink]` with
/// `decided` all-true, which is what an earlier version of this function returned): the *raw,
/// unconditioned* `probs` this runs on come from a single network pass with nothing decided yet
/// (`probs_for` always feeds an all-ones mask), so no individual variable's marginal is likely to be
/// sharply peaked on its own -- that only happens once evidence from decided neighbours narrows
/// things down, which `combined_partial_conditional` (used during the actual resampling walk) has
/// and this doesn't. A plain product multiplies that per-variable uncertainty together once per
/// scope variable, so it collapses toward 0 purely from scope width -- a 9-variable Sudoku
/// row/column/box needs each variable's raw marginal to average above roughly 0.36 confidence just
/// to clear 0.0001, which single-pass raw marginals routinely don't -- regardless of whether
/// `assignment` actually satisfies the constraint. That was an earlier failure mode observed in
/// practice: every MDD read back ~0 every step no matter how many constraints the current
/// assignment actually satisfied. The geometric mean is scale-invariant with scope width -- it
/// reflects the *average* per-variable confidence along the accepted path rather than their
/// compounded joint probability -- while still returning exactly 0 whenever the path doesn't exist.
fn mdd_satisfaction_probability(mdd: &Mdd, probs: &[Vec<f64>], assignment: &[ValueIndex]) -> f64 {
    let Some(path) = mdd_walk(mdd, assignment) else {
        return 0.0;
    };
    if path.is_empty() {
        return 1.0;
    }
    let log_sum: f64 = path
        .iter()
        .map(|&(var, val)| safe_ln(probs[var.0][val.0]))
        .sum();
    (log_sum / path.len() as f64).exp()
}

/// Extra softening applied to `DestroyRule::Probabilistic`'s WMC before turning it into a destroy
/// probability -- matches `ConsFormerConfig::tau`, the logit-scaling temperature already used when
/// training this network (see `learning::consformer::architecture::ConsFormer::forward`), on the
/// theory that whatever softening scale the model was trained around is a reasonable default here
/// too, rather than introducing a fresh, unrelated number. Fixed, not exposed as a knob -- see
/// `DestroyRule::Probabilistic`'s doc for why plain `wmc` alone isn't enough.
const DESTROY_TEMPERATURE: f64 = 5.0;

/// Which rule decides whether an MDD's scope gets resampled this step -- see
/// `MddSampler::destroyed_variables`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestroyRule {
    /// Destroy exactly the MDDs `assignment` currently violates (`mdd_accepts`), deterministically
    /// -- no randomness, no partial credit for a "confident but violated" or "unsure but satisfied"
    /// constraint. Cheap (no network probabilities needed at all) and, in practice, close to what
    /// `Probabilistic` degenerates to whenever the network's raw per-variable beliefs are already
    /// close to one-hot -- which they usually are (see `Probabilistic`'s doc).
    Deterministic,
    /// Destroy each MDD independently with probability `1 - wmc.powf(1.0 / DESTROY_TEMPERATURE)`,
    /// `wmc` being `mdd_satisfaction_probability`'s geometric-mean confidence in `assignment`'s
    /// current values. A violated MDD is still destroyed unconditionally -- WMC 0 stays 0 whatever
    /// the exponent. For a satisfied one, this softens `wmc` toward 1 before turning it into a
    /// destroy probability, which matters because the plain `wmc` on its own tends to already sit
    /// very close to 0 or very close to 1: the network's raw, single-pass marginals are usually
    /// sharp rather than smoothly uncertain, so without this softening `Probabilistic` mostly just
    /// reduces to `Deterministic` anyway. `DESTROY_TEMPERATURE` restores genuine gradation between
    /// "clearly fine" and "technically fine but the network isn't very sure" MDDs.
    Probabilistic,
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

/// For each global variable, every `(mdd_index, layer)` pair where that MDD has this variable in
/// its scope, at that layer. Shared by `MddSampler::new` (per-step destroy/resample decoding) and
/// `bp::belief_propagation` (multi-round marginal aggregation) -- both need the same "which MDDs
/// does this variable belong to, and at what layer in each" index.
fn build_var_to_mdds(mdds: &[Mdd]) -> Vec<Vec<(usize, usize)>> {
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
    var_to_mdds
}

/// Sampler used for generating a new assignment given a set of MDDs; aggregate multiple MDDs
/// opinion about variables update
pub struct MddSampler<'a> {
    mdds: &'a [Mdd],
    /// Scaling factor for each MDD choice, default to 1.0 (each MDD has equal voting power)
    weights: Vec<f64>,
    /// For each global variable, every `(mdd_index, layer)` pair where that MDD has this variable
    /// in its scope, at that layer.
    var_to_mdds: Vec<Vec<(usize, usize)>>,
}

impl<'a> MddSampler<'a> {
    pub fn new(mdds: &'a [Mdd]) -> Self {
        Self {
            mdds,
            weights: vec![1.0; mdds.len()],
            var_to_mdds: build_var_to_mdds(mdds),
        }
    }

    /// Sets the voting weight of the MDD at mdd_index
    pub fn set_weight(&mut self, mdd_index: usize, weight: f64) {
        self.weights[mdd_index] = weight;
    }

    /// Number of variables this sampler knows about (the shared scope of its MDDs).
    pub fn number_variables(&self) -> usize {
        self.var_to_mdds.len()
    }

    pub fn members_of(&self, var: VariableIndex) -> &[(usize, usize)] {
        let members = &self.var_to_mdds[var.0];
        assert!(
            !members.is_empty(),
            "variable {} is not in the scope of any MDD -- every variable must appear in at \
             least one for MddSampler to have anything to decode it from",
            var.0
        );
        members
    }

    /// Combines every MDD's `partial_conditional` for `var`, given `decided` evidence from
    /// `assignment` -- a weighted product of experts in log-space, dividing out the network's own
    /// belief once per MDD so it isn't double-counted when several MDDs share `var`. An all-`false`
    /// `decided` gives `var`'s plain unconditional marginal under every MDD combined -- there's no
    /// separate "nothing decided yet" path; this handles it for free via `partial_conditional`.
    ///
    /// Per MDD, whether a `decided` variable other than `var` clamps that MDD's forward or
    /// backward pass depends purely on that MDD's own layer order relative to `var` -- there is no
    /// requirement that `decided`, or any other MDD, agree on an order at all. An undecided
    /// variable is marginalised out in every MDD, regardless of which side of `var`'s layer it
    /// sits on in that MDD.
    pub fn combined_partial_conditional(
        &self,
        var: VariableIndex,
        probs: &[Vec<f64>],
        assignment: &[ValueIndex],
        decided: &[bool],
    ) -> Vec<f64> {
        let members = self.members_of(var);
        let network_log: Vec<f64> = probs[var.0].iter().map(|&p| safe_ln(p)).collect();
        let mut log_combined = network_log.clone();
        for &(mdd_index, layer) in members {
            let weight = self.weights[mdd_index];
            let conditional =
                partial_conditional(&self.mdds[mdd_index], layer, probs, assignment, decided);
            for (d, log_d) in log_combined.iter_mut().enumerate() {
                // For each MDD, we divide (in log-space) by the network's own belief. Otherwise,
                // combining multiple MDDs over the same variable introduces the network belief
                // multiple times into the marginal, which is fine when there are few constraints,
                // but gives degenerate results when the number of MDDs increases (e.g., graph
                // colouring).
                *log_d += weight * (safe_ln(conditional[d]) - network_log[d]);
            }
        }
        log_combine_and_normalize(log_combined)
    }

    /// Picks which variables sequential imputation should resample this step, driven by `probs`
    /// (this step's network output), `assignment` (the current, pre-step assignment), and `rule`
    /// (see `DestroyRule`) -- no fraction. A destroyed MDD's whole scope gets resampled -- resampling
    /// only part of an MDD's variables would just leave the same unsatisfied structure half-fixed.
    /// The returned mask covers every variable belonging to at least one destroyed MDD.
    pub fn destroyed_variables(
        &self,
        probs: &[Vec<f64>],
        assignment: &[ValueIndex],
        rule: DestroyRule,
    ) -> Vec<bool> {
        let mut destroyed = vec![false; self.number_variables()];
        let mut destroyed_mdds = 0usize;
        // Only collected when debug logging is on and `rule` is `Probabilistic` -- cheap (the WMC
        // is already computed either way) but no reason to allocate/format it otherwise.
        let mut wmcs: Vec<f64> = Vec::new();
        let log_enabled = log::log_enabled!(log::Level::Debug);

        crate::utils::with_rng(|rng| {
            for mdd in self.mdds {
                let destroy = match rule {
                    DestroyRule::Deterministic => !mdd_accepts(mdd, assignment),
                    DestroyRule::Probabilistic => {
                        let wmc = mdd_satisfaction_probability(mdd, probs, assignment)
                            .powf(1.0 / DESTROY_TEMPERATURE);
                        if log_enabled {
                            wmcs.push(wmc);
                        }
                        rng.random_bool((1.0 - wmc).clamp(0.0, 1.0))
                    }
                };
                if destroy {
                    destroyed_mdds += 1;
                    for layer in 0..mdd.number_layers() - 1 {
                        let var = mdd.decision_at_layer(layer);
                        destroyed[var.0] = true;
                    }
                }
            }
        });

        if log_enabled {
            let destroyed_vars = destroyed.iter().filter(|&&d| d).count();
            // Scientific notation, not `{:.4}`: WMC legitimately spans many orders of magnitude
            // (an exactly-0.0 violated MDD next to a satisfied one at, say, 3e-5), and a fixed
            // 4-decimal format displays both ends of that range as an indistinguishable "0.0000" --
            // which is exactly what made an earlier, since-fixed bug (every MDD reading back ~0
            // regardless of whether the assignment satisfied it) look identical to this heuristic
            // actually working as intended, in the log alone, for a while.
            let wmc_summary = match rule {
                DestroyRule::Deterministic => String::new(),
                DestroyRule::Probabilistic => {
                    let (min_wmc, max_wmc, mean_wmc) = min_max_mean(&wmcs);
                    format!(
                        " -- WMC (post-temperature) min={min_wmc:.3e}, max={max_wmc:.3e}, \
                         mean={mean_wmc:.3e}"
                    )
                }
            };
            log::debug!(
                "destroyed_variables[{rule:?}]: {destroyed_mdds}/{} MDD(s)/constraint group(s) \
                 destroyed, covering {destroyed_vars}/{} variable(s){wmc_summary}",
                self.mdds.len(),
                destroyed.len()
            );
        }

        destroyed
    }
}

/// `(min, max, mean)` of `values`, or all-zero for an empty slice -- only ever called behind a
/// debug-logging check, so a degenerate empty input is just displayed as zeros rather than
/// meriting an `Option`.
fn min_max_mean(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (min, max, mean)
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

pub(crate) fn argmax(weights: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("weights should never be NaN"))
        .map(|(index, _)| index)
        .expect("weights should not be empty")
}

pub(crate) fn sample_categorical(weights: &[f64]) -> usize {
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
    fn partial_conditional_with_everything_decided_matches_uniform_marginal_on_free_variable() {
        // x != y over {0,1} x {0,1}: only 2 of the 4 combinations are valid. Clamping y=0 (fully
        // decided) should put all conditional mass on x=1 (the only value making x != y hold),
        // regardless of the (uniform) input probabilities -- same behaviour the old, always-fully-
        // clamped `clamped_conditional` had, since `decided = [true, true]` recovers it exactly.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);

        let x_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == x)
            .expect("x must be in the MDD's scope");

        let mut assignment = vec![ValueIndex(0); problem.number_variables()];
        assignment[y.0] = ValueIndex(0);
        let decided = vec![true; problem.number_variables()];

        let conditional = partial_conditional(&mdd, x_layer, &probs, &assignment, &decided);
        assert_eq!(conditional.len(), 2);
        assert!((conditional[0] - 0.0).abs() < 1e-9, "{conditional:?}");
        assert!((conditional[1] - 1.0).abs() < 1e-9, "{conditional:?}");
    }

    #[test]
    fn partial_conditional_with_nothing_decided_matches_brute_force_marginal_for_not_equals() {
        // decided = all-false should recover the plain unconditional marginal, since there's no
        // evidence left to clamp on anywhere in the MDD -- checked here against a hand-computed
        // value rather than a second implementation.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = vec![vec![0.9, 0.1], vec![0.3, 0.7]];

        let x_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == x)
            .unwrap();
        let y_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == y)
            .unwrap();

        let assignment = vec![ValueIndex(0); problem.number_variables()];
        let decided = vec![false; problem.number_variables()];
        let conditional_x = partial_conditional(&mdd, x_layer, &probs, &assignment, &decided);
        let conditional_y = partial_conditional(&mdd, y_layer, &probs, &assignment, &decided);

        // P(x=0) ~ probs_x[0]*probs_y[1], P(x=1) ~ probs_x[1]*probs_y[0], normalised; and
        // symmetrically for y. Hand-computed from probs = [[0.9, 0.1], [0.3, 0.7]].
        assert!(
            (conditional_x[0] - 21.0 / 22.0).abs() < 1e-9,
            "{conditional_x:?}"
        );
        assert!(
            (conditional_x[1] - 1.0 / 22.0).abs() < 1e-9,
            "{conditional_x:?}"
        );
        assert!(
            (conditional_y[0] - 1.0 / 22.0).abs() < 1e-9,
            "{conditional_y:?}"
        );
        assert!(
            (conditional_y[1] - 21.0 / 22.0).abs() < 1e-9,
            "{conditional_y:?}"
        );
    }

    #[test]
    fn partial_conditional_abstains_uniformly_when_locally_infeasible() {
        // x != y, y != z, both x and y decided and pinned to the same (infeasible) value: no path
        // through the MDD survives, so the conditional over z's layer should come back uniform
        // rather than panicking or returning all-zero.
        //
        // All three variables share the same domain here (unlike the pairwise-not_equals trio in
        // `sequential_walk_over_a_single_mdd_is_always_feasible`, which mirrors
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
        let mut decided = vec![false; problem.number_variables()];
        decided[x.0] = true;
        decided[y.0] = true;

        let conditional = partial_conditional(&mdd, z_layer, &probs, &assignment, &decided);
        assert_eq!(conditional.len(), 3);
        for p in &conditional {
            assert!((p - 1.0 / 3.0).abs() < 1e-9, "{conditional:?}");
        }
    }

    #[test]
    fn partial_conditional_marginalises_undecided_evidence_regardless_of_mdd_layer_order() {
        // x != y != z (all-different chain, all sharing domain {0,1,2}, uniform probs). Deciding
        // only y (leaving x and z undecided, whichever side of y's layer they happen to fall on in
        // this MDD) must sum x and z out rather than silently treating them as fixed at their
        // placeholder ValueIndex(0). Every value of y forbids exactly one value each for x and z
        // (2 choices left for each, independently, regardless of which value y takes), so the
        // correct marginal is exactly uniform -- hand-derivable, no second implementation needed as
        // an oracle.
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

        let y_layer = (0..mdd.number_layers() - 1)
            .find(|&l| mdd.decision_at_layer(l) == y)
            .unwrap();

        // Placeholder ValueIndex(0) for x and z everywhere -- decided says they must be ignored.
        let assignment = vec![ValueIndex(0); problem.number_variables()];
        let decided = vec![false; problem.number_variables()];

        let conditional = partial_conditional(&mdd, y_layer, &probs, &assignment, &decided);
        assert_eq!(conditional.len(), 3);
        for p in &conditional {
            assert!((p - 1.0 / 3.0).abs() < 1e-9, "{conditional:?}");
        }
    }

    #[test]
    fn sequential_walk_over_a_single_mdd_is_always_feasible() {
        // A single MDD spanning all variables: visiting every variable once, in any order, each
        // time sampling from `combined_partial_conditional` (greedy) given only the variables
        // already visited, should always land on a feasible assignment -- only edges that exist in
        // the MDD are ever chosen, regardless of what order the walk visits variables in.
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
        let sampler = MddSampler::new(&mdds);

        for order in [vec![x, y, z], vec![z, x, y], vec![y, z, x]] {
            let mut assignment = vec![ValueIndex(0); problem.number_variables()];
            let mut decided = vec![false; problem.number_variables()];

            for &var in &order {
                // decided starts all-false, so the first variable in the order is automatically
                // drawn from its plain marginal -- no separate "nothing decided yet" case needed.
                let combined =
                    sampler.combined_partial_conditional(var, &probs, &assignment, &decided);
                assignment[var.0] = ValueIndex(argmax(&combined));
                decided[var.0] = true;
            }

            let xv = problem[x].value(assignment[x.0]);
            let yv = problem[y].value(assignment[y.0]);
            let zv = problem[z].value(assignment[z.0]);
            assert_ne!(xv, yv, "order={order:?}");
            assert_ne!(yv, zv, "order={order:?}");
            assert_ne!(xv, zv, "order={order:?}");
        }
    }

    #[test]
    fn gibbs_sampler_combines_multiple_mdds_via_product_of_experts() {
        // Two single-constraint MDDs over the same 2 variables: x != y and, separately, y != 0
        // encoded as a gcc bound. Combined, the only valid combos are the ones satisfying both, so
        // deciding x should push y's partial conditional onto the value consistent with both.
        // Two independently-built `Problem`s over the same variable layout (`Problem` isn't
        // `Clone`, and each `Mdd` owns its own `Arc<Problem>` anyway -- `MddSampler` only
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
        let sampler = MddSampler::new(&mdds);

        let mut assignment = vec![ValueIndex(0); problem_a.number_variables()];
        assignment[x.0] = ValueIndex(1);
        let mut decided = vec![false; problem_a.number_variables()];
        decided[x.0] = true;
        let combined = sampler.combined_partial_conditional(y, &probs, &assignment, &decided);
        // Only not_equals(x, y) actually constrains y here; with x=1, y=0 is the only option.
        // `mdd_b`'s unbound gcc contributes a uniform (non-opinionated) conditional. `safe_ln`
        // means y=1's genuinely-zero conditional no longer gets floored up to a fixed MIN_PROB, so
        // the combined result can land exactly on [1.0, 0.0] -- the tolerance here is kept loose
        // regardless, since exactness isn't the point of this test.
        assert!((combined[0] - 1.0).abs() < 1e-6, "{combined:?}");
        assert!((combined[1] - 0.0).abs() < 1e-6, "{combined:?}");
    }

    /// Builds `count` independent, mutually uninformative MDDs over a single variable `x` (each
    /// one an unbound gcc, same trick as `gibbs_sampler_combines_multiple_mdds_via_product_of_experts`
    /// -- a real scope, but zero actual restriction). `x` belongs to all of them, so
    /// `var_to_mdds[x]` has `count` entries: exactly the fan-in situation `combined_partial_conditional`
    /// needs to not double-count the network's own belief about `x`.
    fn uninformative_mdds_over_one_variable(
        count: usize,
    ) -> (Arc<Problem>, VariableIndex, Vec<Mdd>) {
        let mut problems = Vec::with_capacity(count);
        for _ in 0..count {
            let mut problem = Problem::default();
            let x = problem.add_variable(vec![0, 1, 2], None);
            gcc(&mut problem, vec![x], vec![]);
            problems.push(Arc::new(problem));
        }
        let x = VariableIndex(0);
        let mdds: Vec<Mdd> = problems
            .iter()
            .map(|p| {
                let constraints: Vec<ConstraintIndex> = p.iter_constraints().collect();
                build_mdd(p.clone(), &constraints)
            })
            .collect();
        (problems[0].clone(), x, mdds)
    }

    #[test]
    fn combined_partial_conditional_does_not_multiply_the_networks_own_belief_once_per_mdd_membership(
    ) {
        // Regression test for the fan-in over-counting bug: previously, the marginal path summed
        // each member MDD's log-marginal as-is, and since every MDD's own marginal already bakes
        // in the network's belief about `x` (`probs[x]`), a variable belonging to `count`
        // uninformative MDDs would have that same belief effectively raised to the `count`-th
        // power -- sharpening the combined distribution even though none of the MDDs contribute
        // any actual constraint information. With the fix, the network's belief must be applied
        // exactly once, so the combined distribution should stay close to `probs[x]` itself
        // regardless of how many (uninformative) MDDs `x` is a member of. Exercised through
        // `combined_partial_conditional` with nothing decided, which is the "no evidence" case.
        let probs = vec![vec![0.7, 0.2, 0.1]];

        for count in [1, 5, 20] {
            let (problem, x, mdds) = uninformative_mdds_over_one_variable(count);
            let sampler = MddSampler::new(&mdds);
            let assignment = vec![ValueIndex(0); problem.number_variables()];
            let decided = vec![false; problem.number_variables()];
            let combined = sampler.combined_partial_conditional(x, &probs, &assignment, &decided);

            for d in 0..3 {
                assert!(
                    (combined[d] - probs[0][d]).abs() < 1e-6,
                    "count={count}, d={d}, combined={combined:?}, expected={:?}",
                    probs[0]
                );
            }
        }
    }

    #[test]
    fn combined_partial_conditional_preserves_network_confidence_far_below_the_old_min_prob_floor()
    {
        // Regression test for the entropy-floor bug: the old `MIN_PROB = 1e-9` clamp was applied
        // to `probs` itself before it ever reached the combination math, so no matter how sharp the
        // network's real output was, the combined distribution could never read as more confident
        // than that floor allowed -- observed as the combination's output plateauing at a fixed
        // entropy regardless of how much further the raw network kept sharpening, identically for
        // every problem type and grouping (even `size_bound = 0`, i.e. plain per-constraint MDDs
        // with no merging at all), since the floor sat upstream of any MDD-specific computation.
        // With `safe_ln`, a probability well below the old floor (here 1e-15, six orders of
        // magnitude past where `MIN_PROB` used to clamp) must survive combination with a set of
        // uninformative MDDs essentially unchanged, instead of being floored up to ~1e-9.
        // Exercised through `combined_partial_conditional` with nothing decided, which is the "no
        // evidence" case.
        let probs = vec![vec![1.0 - 2e-15, 1e-15, 1e-15]];
        let (problem, x, mdds) = uninformative_mdds_over_one_variable(5);
        let sampler = MddSampler::new(&mdds);
        let assignment = vec![ValueIndex(0); problem.number_variables()];
        let decided = vec![false; problem.number_variables()];
        let combined = sampler.combined_partial_conditional(x, &probs, &assignment, &decided);

        // The old MIN_PROB=1e-9 floor would have left combined[1]/combined[2] at roughly 1e-9;
        // asserting well below that (and consistent with the true ~1e-15 input) catches a
        // regression back to a fixed floor without demanding exact-to-the-bit reproduction of the
        // input (renormalization and floating-point summation can shift the last few digits).
        assert!(combined[1] < 1e-12, "combined={combined:?}");
        assert!(combined[2] < 1e-12, "combined={combined:?}");
        assert!((combined[0] - 1.0).abs() < 1e-9, "combined={combined:?}");
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

    #[test]
    fn mdd_satisfaction_probability_is_zero_for_a_violated_assignment_and_positive_for_a_satisfied_one(
    ) {
        // Regression test for the destroy-step bug: an earlier version of this function ignored
        // the current assignment entirely (marginalised over every possible one instead of
        // conditioning on this one), so it returned ~0 regardless of whether the assignment
        // actually satisfied the constraint -- destroying every MDD every step no matter how many
        // constraints were already satisfied. x != y over {0,1}: x=y=0 has no accepting path (the
        // constraint is violated), x=0,y=1 does.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = uniform_probs(&problem);

        let violated = vec![ValueIndex(0), ValueIndex(0)];
        let satisfied = vec![ValueIndex(0), ValueIndex(1)];

        assert_eq!(mdd_satisfaction_probability(&mdd, &probs, &violated), 0.0);
        assert!(mdd_satisfaction_probability(&mdd, &probs, &satisfied) > 0.0);
    }

    #[test]
    fn mdd_satisfaction_probability_is_the_geometric_mean_not_the_product_along_the_path() {
        // Regression test for a second, related bug: the first fix conditioned on the assignment
        // (previous test) but still returned the raw *product* of per-variable probabilities along
        // the path, same as WMC does. That's scale-sensitive with scope width for reasons that have
        // nothing to do with whether the assignment is actually satisfying -- moderately confident
        // per-variable beliefs still crush toward 0 once there are enough of them to multiply
        // together, which is exactly what was observed in practice (every MDD reading back ~0.0000
        // regardless of how many constraints were satisfied, because Sudoku's row/column/box scopes
        // are 9 variables wide). The geometric mean is scale-invariant instead: with both terms
        // equal (0.6 here), it should land exactly on that shared value, not on their product
        // (0.36).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);
        let probs = vec![vec![0.6, 0.4], vec![0.4, 0.6]];

        // x=0, y=1 satisfies not_equals, with probs[x][0] = 0.6 and probs[y][1] = 0.6.
        let satisfied = vec![ValueIndex(0), ValueIndex(1)];
        let wmc = mdd_satisfaction_probability(&mdd, &probs, &satisfied);
        assert!((wmc - 0.6).abs() < 1e-9, "wmc={wmc}, expected the geometric mean 0.6, not the product 0.36");
    }

    #[test]
    fn destroyed_variables_with_deterministic_rule_destroys_exactly_the_violated_mdds() {
        // One MDD, not_equals(x, y), over a 3-variable problem where w is free but appears in no
        // constraint at all -- so it belongs to no MDD's scope and `destroyed_variables` can never
        // touch it, regardless of what x and y are assigned. x=y=0 violates not_equals, so
        // `Deterministic` should destroy exactly x and y, leaving w alone.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let _w = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);

        let mdds = vec![mdd];
        let sampler = MddSampler::new(&mdds);
        let probs = uniform_probs(&problem);
        let assignment = vec![ValueIndex(0), ValueIndex(0), ValueIndex(0)];

        let destroyed =
            sampler.destroyed_variables(&probs, &assignment, DestroyRule::Deterministic);
        assert_eq!(destroyed, vec![true, true, false]);
    }
}
