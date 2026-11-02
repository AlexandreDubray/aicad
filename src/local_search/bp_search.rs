//! The belief-propagation local search: it repairs on the CPU only, with belief propagation over
//! the decomposed MDDs of the constraints starting from uniform probabilities.
//!
//! # Meaning
//!
//! No network and no device are involved, and nothing restricts the problems of a batch: each row
//! is repaired on its own, so problems of different sizes can be searched together.
//!
//! # How it works
//!
//! Setup compiles the constraints of every problem. A step destroys part of each pending row, then
//! runs belief propagation on the row, with every in-domain value equally likely at the start,
//! and takes the value of largest marginal (or samples one) for each freed variable.

use std::sync::Arc;

use rand::rngs::StdRng;

use crate::destroy::{DestroyKind, DestroyOperator};
use crate::mdd::arena::MddCompilationConfig;
use crate::modelling::Problem;
use crate::sampling::DecodeMode;

use super::repair::{BatchRepair, MddRepair};
use super::{SearchArchitecture, SearchState, destroy_rows, row_seeds};

/// The settings of the belief-propagation local search.
#[derive(Debug, Clone)]
pub(crate) struct BeliefPropagationSearchConfig {
    /// Destroy operator choosing the variables to free at each iteration.
    pub destroy_kind: DestroyKind,
    /// Fraction of the variables the destroy operator frees at each iteration, in `[0, 1]`.
    pub destroy_fraction: f64,
    /// Sample values from the marginals instead of taking the most likely one.
    pub stochastic: bool,
    /// Number of belief-propagation rounds per iteration.
    pub bp_iterations: usize,
}

impl Default for BeliefPropagationSearchConfig {
    fn default() -> Self {
        Self {
            destroy_kind: DestroyKind::default(),
            destroy_fraction: 1.0,
            stochastic: false,
            bp_iterations: 1,
        }
    }
}

/// The belief-propagation local search.
pub(crate) struct BeliefPropagationSearch {
    /// Heuristic choosing the variables to free.
    destroy_op: Box<dyn DestroyOperator>,
    /// How a value is taken from a marginal.
    mode: DecodeMode,
    /// Rounds of belief propagation per iteration.
    bp_iterations: usize,
}

impl BeliefPropagationSearch {
    /// Builds the search from its settings.
    pub(crate) fn new(config: &BeliefPropagationSearchConfig) -> Self {
        Self {
            destroy_op: config.destroy_kind.build(config.destroy_fraction),
            mode: if config.stochastic {
                DecodeMode::Sample
            } else {
                DecodeMode::Greedy
            },
            bp_iterations: config.bp_iterations,
        }
    }
}

impl SearchArchitecture for BeliefPropagationSearch {
    fn setup<'a>(&'a self, problems: &[Arc<Problem>]) -> Result<Box<dyn SearchState + 'a>, String> {
        // The constraints are compiled over the alphabet that covers the values of the batch.
        let domain_size = MddRepair::domain_size_covering(problems);
        let repair = MddRepair::new(
            MddCompilationConfig::default(),
            domain_size,
            self.mode,
            self.bp_iterations,
        );
        Ok(Box::new(BeliefPropagationState {
            search: self,
            repair: BatchRepair::new(repair, problems),
            // Every value starts equally likely.
            uniform: 1.0 / domain_size as f64,
        }))
    }
}

/// The state of a belief-propagation search over a batch.
struct BeliefPropagationState<'a> {
    /// The components of the search.
    search: &'a BeliefPropagationSearch,
    /// Repairs the freed variables of the rows, with the alphabet of this batch.
    repair: BatchRepair,
    /// The probability of every value at the start of belief propagation.
    uniform: f64,
}

impl SearchState for BeliefPropagationState<'_> {
    fn step(
        &mut self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    ) {
        let destroyed = destroy_rows(
            self.search.destroy_op.as_ref(),
            problems,
            assignments,
            pending,
            rng,
        );
        let seeds = row_seeds(problems.len(), rng);
        let uniform = self.uniform;
        self.repair
            .repair_rows(problems, assignments, &destroyed, &seeds, |_, _, _| uniform);
    }

    fn solved(&mut self, solved: &[bool]) {
        self.repair.solved(solved);
    }

    fn retain(&mut self, keep: &[bool]) {
        self.repair.retain(keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use crate::local_search::{Budget, LocalSearch};

    fn clique(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect());
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    #[test]
    fn problems_of_different_sizes_are_solved_together() {
        // Different numbers of variables and of values: nothing here is rectangular.
        let problems = vec![clique(3, 3), clique(5, 6), clique(4, 4)];
        // Uniform marginals tie everywhere, so the values are sampled, not taken greedily.
        let search = LocalSearch::new(Box::new(BeliefPropagationSearch::new(
            &BeliefPropagationSearchConfig {
                destroy_fraction: 0.5,
                stochastic: true,
                bp_iterations: 3,
                ..BeliefPropagationSearchConfig::default()
            },
        )));
        let budget = Budget {
            iteration_limit: 2000,
            ..Budget::default()
        };
        let solutions = search.run(&problems, budget, 0, None).unwrap();
        for (problem, solution) in problems.iter().zip(&solutions) {
            let values = solution.assignment().expect("an easy problem is solved");
            assert!(problem.is_solution(values));
        }
    }
}
