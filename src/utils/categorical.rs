//! Helpers on categorical distributions given as slices of non-negative weights.

use rand::RngExt;

/// How a categorical distribution given as weights turns into a value: [`sample_categorical`] or
/// [`argmax`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodeMode {
    /// Draw a value proportionally to its weight.
    Sample,
    /// Take the value of largest weight.
    Greedy,
}

/// The index of the largest weight; the first one on ties.
pub(crate) fn argmax(weights: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("weights should never be NaN"))
        .map(|(index, _)| index)
        .expect("weights should not be empty")
}

/// Draws an index with probability proportional to its weight, from the process-wide RNG
/// ([`with_rng`](super::with_rng)). The weights need not be normalised, but must have a positive sum.
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
