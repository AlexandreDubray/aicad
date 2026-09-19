use super::view::{MddView, MddViewWithOrder};
#[cfg(test)]
use super::Mdd;
use crate::modelling::ValueIndex;

/// Full, unclamped forward (WMC) pass over `mdd`: `alpha[layer][node]` is the total mass reaching
/// `node` (indexed within its layer) from the root, with each layer's edges weighted by
/// `weights[layer][value]` -- `weights` is indexed *by layer within this MDD* (length
/// `mdd.number_layers() - 1`), not by global variable id.
///
/// Generic over `MddView` -- the purely structural read surface implemented by both the
/// legacy, self-contained `Mdd` and the arena-shared `MddStructure`/`CompiledConstraint` (see
/// `crate::mdd::arena`) -- so this and every other function in this module work unchanged
/// against either representation.
pub fn forward<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
                let weight = weights[layer][value.0];
                next_alpha[mdd[edge].to().1] += mass * weight;
            }
        }
        alphas.push(next_alpha);
    }

    alphas
}

/// The backward counterpart of `forward`: `beta[layer][node]` is the total mass from `node` to the
/// sink, weighted the same way (`weights` indexed by layer, same convention).
pub fn backward<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let last_layer = mdd.sink().0;
    let mut betas: Vec<Vec<f64>> = vec![Vec::new(); last_layer + 1];
    betas[last_layer] = vec![1.0; mdd.number_nodes_in_layer(last_layer)];

    for layer in (0..last_layer).rev() {
        let mut prev_beta = vec![0.0; mdd.number_nodes_in_layer(layer)];
        for node in mdd.nodes_in_layer(layer) {
            let mut mass = 0.0;
            for edge in mdd[node].iter_children() {
                let value = mdd[edge].assignment();
                let weight = weights[layer][value.0];
                mass += weight * betas[layer + 1][mdd[edge].to().1];
            }
            prev_beta[node.1] = mass;
        }
        betas[layer] = prev_beta;
    }

    betas
}

/// `WMC = alpha(sink)`, root and sink both always being node index 0 within their layer (`Mdd`'s
/// own convention).
pub fn wmc<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> f64 {
    forward(mdd, weights)[mdd.sink().0][0]
}

/// `gradient[layer][value] = d(WMC)/d(weights[layer][value])`, unnormalized -- from `alpha`/`beta`
/// already computed by the caller (see `wmc_and_gradient` to compute both from scratch in one
/// call).
fn gradient_from_forward_backward<T: MddView>(
    mdd: &T,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    domain_sizes: &[usize],
) -> Vec<Vec<f64>> {
    let last_layer = mdd.sink().0;
    (0..last_layer)
        .map(|layer| {
            let mut grad = vec![0.0; domain_sizes[layer]];
            for node in mdd.nodes_in_layer(layer) {
                let mass = alpha[layer][node.1];
                if mass == 0.0 {
                    continue;
                }
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    grad[value.0] += mass * beta[layer + 1][mdd[edge].to().1];
                }
            }
            grad
        })
        .collect()
}

pub fn gradient<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let alpha = forward(mdd, weights);
    let beta = backward(mdd, weights);
    let domain_sizes: Vec<usize> = weights.iter().map(|w| w.len()).collect();
    gradient_from_forward_backward(mdd, &alpha, &beta, &domain_sizes)
}

pub fn wmc_and_gradient<T: MddView>(mdd: &T, weights: &[Vec<f64>]) -> (f64, Vec<Vec<f64>>) {
    let alpha = forward(mdd, weights);
    let beta = backward(mdd, weights);
    let value = alpha[mdd.sink().0][0];
    let domain_sizes: Vec<usize> = weights.iter().map(|w| w.len()).collect();
    let grad = gradient_from_forward_backward(mdd, &alpha, &beta, &domain_sizes);
    (value, grad)
}

/// Generalises `forward` up to `target_layer`: at a layer whose variable is `decided`, follows only
/// the edge matching `assignment`'s current value for it. At a layer whose variable is not yet
/// `decided`, sums over every outgoing edge instead.
pub fn partial_forward<T: MddViewWithOrder>(
    mdd: &T,
    target_layer: usize,
    weights: &[Vec<f64>],
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
                        let weight = weights[variable.0][value.0];
                        next_alpha[mdd[edge].to().1] += mass * weight;
                        break;
                    }
                }
            } else {
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let weight = weights[variable.0][value.0];
                    next_alpha[mdd[edge].to().1] += mass * weight;
                }
            }
        }
        alpha = next_alpha;
    }

    alpha
}

/// The backward counterpart of `partial_forward`: generalises `backward` down to `target_layer`,
/// clamping a `decided` layer's variable to its assigned value and summing over every edge at an
/// undecided one.
pub fn partial_backward<T: MddViewWithOrder>(
    mdd: &T,
    target_layer: usize,
    weights: &[Vec<f64>],
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
                        let weight = weights[variable.0][value.0];
                        mass += weight * beta[mdd[edge].to().1];
                        break;
                    }
                }
            } else {
                for edge in mdd[node].iter_children() {
                    let value = mdd[edge].assignment();
                    let weight = weights[variable.0][value.0];
                    mass += weight * beta[mdd[edge].to().1];
                }
            }
            prev_beta[node.1] = mass;
        }
        beta = prev_beta;
    }

    beta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::modelling::{all_different, not_equals, ConstraintIndex, Problem};
    use std::sync::Arc;

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
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem, &constraints);

        let weights = vec![vec![0.2, 0.5, 0.3], vec![0.1, 0.3, 0.6], vec![0.4, 0.4, 0.2]];
        let (base_wmc, grad) = wmc_and_gradient(&mdd, &weights);
        assert!((wmc(&mdd, &weights) - base_wmc).abs() < 1e-12);

        let eps = 1e-6;
        for layer in 0..weights.len() {
            for value in 0..weights[layer].len() {
                let mut bumped = weights.clone();
                bumped[layer][value] += eps;
                let bumped_wmc = wmc(&mdd, &bumped);
                let finite_diff = (bumped_wmc - base_wmc) / eps;
                assert!(
                    (finite_diff - grad[layer][value]).abs() < 1e-4,
                    "layer={layer} value={value} analytic={} finite_diff={finite_diff}",
                    grad[layer][value],
                );
            }
        }
    }

    #[test]
    fn partial_forward_partial_backward_agree_with_full_forward_backward_when_nothing_is_decided()
    {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = build_mdd(problem.clone(), &constraints);

        let weights = vec![vec![0.2, 0.5, 0.3], vec![0.1, 0.3, 0.6], vec![0.4, 0.4, 0.2]];
        let alpha = forward(&mdd, &weights);
        let beta = backward(&mdd, &weights);

        let assignment = vec![ValueIndex(0); problem.number_variables()];
        let decided = vec![false; problem.number_variables()];
        let mut weights_by_variable = vec![Vec::new(); problem.number_variables()];
        for layer in 0..mdd.number_layers() - 1 {
            let variable = mdd.decision_at_layer(layer);
            weights_by_variable[variable.0] = weights[layer].clone();
        }

        for layer in 0..=mdd.sink().0 {
            let partial_alpha = partial_forward(&mdd, layer, &weights_by_variable, &assignment, &decided);
            assert_eq!(partial_alpha.len(), alpha[layer].len());
            for (a, b) in partial_alpha.iter().zip(&alpha[layer]) {
                assert!((a - b).abs() < 1e-9);
            }

            let partial_beta = partial_backward(&mdd, layer, &weights_by_variable, &assignment, &decided);
            assert_eq!(partial_beta.len(), beta[layer].len());
            for (a, b) in partial_beta.iter().zip(&beta[layer]) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }
}
