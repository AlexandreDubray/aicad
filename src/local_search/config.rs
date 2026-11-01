//! Configuration of a local search run.
//!
//! [`SolveConfig`] holds what every local search takes: the budget, the seed and the batching.
//! What a search is made of is the business of its architecture, and the settings of each live in
//! a [`SearchConfig`] variant.

use std::time::Duration;

use super::bp_search::BeliefPropagationSearchConfig;
use super::neural_prior::{NeuralPriorBpSearchConfig, NeuralPriorSearchConfig};

/// Stopping criterion for the search; whichever limit is first reached stops it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget {
    /// Limit on the runtime.
    pub time_limit: Duration,
    /// Limit on the number of iterations.
    pub iteration_limit: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            time_limit: Duration::MAX,
            iteration_limit: usize::MAX,
        }
    }
}

/// The architecture of the search and its settings: one variant per
/// [`SearchArchitecture`](super::SearchArchitecture).
#[derive(Debug, Clone)]
pub(crate) enum SearchConfig {
    /// A trained network, decoding its logits.
    NeuralPrior(NeuralPriorSearchConfig),
    /// A trained network whose probabilities are the prior of belief propagation on the MDDs.
    NeuralPriorBp(NeuralPriorBpSearchConfig),
    /// Belief propagation on the MDDs from uniform probabilities, on the CPU only.
    BeliefPropagation(BeliefPropagationSearchConfig),
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig::NeuralPrior(NeuralPriorSearchConfig::default())
    }
}

/// Every parameter shared across all possible local searches.
///
/// Built from the default, no limit and a single batch, with the `with_*` methods:
/// `SolveConfig::default().with_runtime_limit(100).with_iteration_limit(2000)`.
#[derive(Debug, Clone, Default)]
pub(crate) struct SolveConfig {
    /// When the search of a batch stops. It applies to each batch separately.
    pub budget: Budget,
    /// Seed of the search. `None` draws one from the process-wide generator, so it is
    /// reproducible when that generator was seeded.
    pub seed: Option<u64>,
    /// Number of problems stepped together; batches run one after the other. `None` puts every
    /// problem in a single batch. The architecture may restrict which problems can share a batch.
    pub batch_size: Option<usize>,
    /// The architecture of the search and its settings.
    pub search: SearchConfig,
}

impl SolveConfig {
    /// Stops the search of each batch after `seconds` seconds.
    pub(crate) fn with_runtime_limit(mut self, seconds: u64) -> Self {
        self.budget.time_limit = Duration::from_secs(seconds);
        self
    }

    /// Stops the search of each batch after `iterations` iterations.
    pub(crate) fn with_iteration_limit(mut self, iterations: usize) -> Self {
        self.budget.iteration_limit = iterations;
        self
    }

    /// Steps at most `batch_size` problems together.
    pub(crate) fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Seeds the search.
    pub(crate) fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the architecture of the search and its settings.
    pub(crate) fn with_search(mut self, search: SearchConfig) -> Self {
        self.search = search;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limits_are_set_one_by_one() {
        let config = SolveConfig::default()
            .with_runtime_limit(100)
            .with_iteration_limit(2000)
            .with_batch_size(400);
        assert_eq!(config.budget.time_limit, Duration::from_secs(100));
        assert_eq!(config.budget.iteration_limit, 2000);
        assert_eq!(config.batch_size, Some(400));
        assert_eq!(config.seed, None);
    }

    #[test]
    fn nothing_is_limited_by_default() {
        let config = SolveConfig::default();
        assert_eq!(config.budget.time_limit, Duration::MAX);
        assert_eq!(config.budget.iteration_limit, usize::MAX);
        assert_eq!(config.batch_size, None);
    }
}
