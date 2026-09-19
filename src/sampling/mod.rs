pub mod bp;

use crate::mdd::MddViewWithOrder;

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
/// its scope, at that layer. Used by `bp::belief_propagation` (multi-round marginal aggregation)
/// to know "which MDDs does this variable belong to, and at what layer in each".
///
/// Generic over `MddViewWithOrder` (implemented by both `Mdd` and the arena-shared
/// `CompiledConstraint`, see `crate::mdd::arena`), so this works for either representation; `num_vars`
/// is passed explicitly rather than read off `mdd.problem()` since a `CompiledConstraint` carries no
/// `Arc<Problem>` of its own (that's the whole point of the split -- see the arena module doc).
fn build_var_to_mdds<T: MddViewWithOrder>(
    mdds: &[T],
    num_vars: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut var_to_mdds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_vars];
    for (mdd_index, mdd) in mdds.iter().enumerate() {
        for layer in 0..mdd.number_layers() - 1 {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
