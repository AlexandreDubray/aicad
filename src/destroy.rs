//! Destroy operators: heuristical way of destroying parts of a full assignment to variables.
//!
//! They are shared by the training of the learned models, which masks the variables it asks the
//! network to predict, and by the local searches, which free them to repair the assignment.

use std::collections::HashSet;

use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::modelling::Problem;

// Flooring factor for scoring to avoid division by zero
const SCORE_EPS: f64 = 1e-8;

/// Trait to implement for a new destroy operator.
pub(crate) trait DestroyOperator: Send + Sync {
    /// Returns the index of the variables in `problem` to destroy given the current assignment.
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;
}

/// The destroy operators one can choose from, as a plain value that builds the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum DestroyKind {
    /// [`RandomDestroy`].
    #[default]
    Random,
    /// [`WorstDestroy`].
    Worst,
    /// [`RelatedDestroy`].
    Related,
}

impl DestroyKind {
    /// Builds the operator of this kind that destroys (on average) `fraction` of the free
    /// variables.
    pub(crate) fn build(self, fraction: f64) -> Box<dyn DestroyOperator> {
        match self {
            DestroyKind::Random => Box::new(RandomDestroy { fraction }),
            DestroyKind::Worst => Box::new(WorstDestroy { fraction }),
            DestroyKind::Related => Box::new(RelatedDestroy { fraction }),
        }
    }
}

/// Uniformly destroys a `fraction` of the free (domain size > 1) variables.
pub(crate) struct RandomDestroy {
    pub fraction: f64,
}

impl DestroyOperator for RandomDestroy {
    fn destroy(&self, problem: &Problem, _assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let mut free: Vec<usize> = free_variables(problem);
        if free.is_empty() {
            return free;
        }
        free.shuffle(rng);
        let k = ((free.len() as f64 * self.fraction).round() as usize).clamp(1, free.len());
        free.truncate(k);
        free
    }
}

/// Stochastic worst removal: each free variable is destroyed with a probability that is the
/// sigmoid of its standardised count of currently-violated constraints, rescaled so the mean
/// probability matches `fraction`. Every free variable keeps a non-zero probability.
pub(crate) struct WorstDestroy {
    pub fraction: f64,
}

impl DestroyOperator for WorstDestroy {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let mut unsat_per_variable = vec![0f64; problem.number_variables()];
        for c in problem.iter_constraints() {
            if !problem[c].is_satisfied(assignment) {
                for v in problem[c].iter_scope() {
                    unsat_per_variable[v.0] += 1.0;
                }
            }
        }
        let free = free_variables(problem);
        if free.is_empty() {
            return free;
        }

        let n = free.len() as f64;
        let values: Vec<f64> = free.iter().map(|&i| unsat_per_variable[i]).collect();
        let mean = values.iter().sum::<f64>() / n;
        let std = if free.len() > 1 {
            (values.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let probs: Vec<f64> = values
            .iter()
            .map(|s| 1.0 / (1.0 + (-(s - mean) / (std + SCORE_EPS)).exp()))
            .collect();
        let scale = self.fraction / (probs.iter().sum::<f64>() / n + SCORE_EPS);

        free.into_iter()
            .zip(probs)
            .filter(|&(_, p)| rng.random_bool((p * scale).clamp(0.0, 1.0)))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Stochastic related removal (Shaw): draws a random subset of constraints
/// and destroys every variable in their scope. Constraints are drawn
/// independently with a probability rescaled so the expected number of
/// destroyed variables tracks `fraction * n`, regardless of how many
/// constraints the problem has relative to its variables
pub(crate) struct RelatedDestroy {
    pub fraction: f64,
}

impl DestroyOperator for RelatedDestroy {
    fn destroy(&self, problem: &Problem, _assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let n = problem.number_variables().max(1);
        let m = problem.number_constraints();
        if m == 0 {
            return Vec::new();
        }

        let total_scope: usize = problem
            .iter_constraints()
            .map(|c| problem[c].iter_scope().count())
            .sum();
        let avg_scope = (total_scope as f64 / m as f64).max(1.0);

        // E[# destroyed] ~= p_constraint * m * avg_scope (upper bound, ignores
        // overlap between scopes)
        let p_constraint = (self.fraction * n as f64 / (m as f64 * avg_scope)).clamp(0.0, 1.0);

        let mut destroyed = HashSet::new();
        for c in problem.iter_constraints() {
            if rng.random_bool(p_constraint) {
                for v in problem[c].iter_scope() {
                    if problem[v].domain_size() > 1 {
                        destroyed.insert(v.0);
                    }
                }
            }
        }
        destroyed.into_iter().collect()
    }
}

/// Returns the variables that can be destroyed in the problem; that is, the variables with domain
/// size greater than 1.
fn free_variables(problem: &Problem) -> Vec<usize> {
    problem
        .iter_variables()
        .filter(|&v| problem[v].domain_size() > 1)
        .map(|v| v.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::not_equals;
    use rand::SeedableRng;

    fn two_pairs() -> Problem {
        let mut problem = Problem::default();
        let vars: Vec<_> = (0..4)
            .map(|_| problem.add_variable(vec![0, 1, 2], None))
            .collect();
        not_equals(&mut problem, vars[0], vars[1]);
        not_equals(&mut problem, vars[2], vars[3]);
        problem
    }

    fn frequencies(
        problem: &Problem,
        assignment: &[isize],
        fraction: f64,
        draws: usize,
    ) -> Vec<f64> {
        let op = WorstDestroy { fraction };
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = vec![0usize; assignment.len()];
        for _ in 0..draws {
            for v in op.destroy(problem, assignment, &mut rng) {
                counts[v] += 1;
            }
        }
        counts
            .into_iter()
            .map(|c| c as f64 / draws as f64)
            .collect()
    }

    #[test]
    fn violated_variables_are_destroyed_more_often_but_not_exclusively() {
        let problem = two_pairs();
        let freq = frequencies(&problem, &[0, 0, 0, 1], 0.5, 20_000);
        assert!((freq[0] - 0.704).abs() < 0.02, "{freq:?}");
        assert!((freq[1] - 0.704).abs() < 0.02, "{freq:?}");
        assert!((freq[2] - 0.296).abs() < 0.02, "{freq:?}");
        assert!((freq[3] - 0.296).abs() < 0.02, "{freq:?}");
    }

    #[test]
    fn a_satisfied_assignment_is_destroyed_uniformly_at_the_requested_fraction() {
        let problem = two_pairs();
        let freq = frequencies(&problem, &[0, 1, 0, 1], 0.3, 20_000);
        assert!(freq.iter().all(|f| (f - 0.3).abs() < 0.02), "{freq:?}");
    }
}
