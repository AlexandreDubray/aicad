//! Destroy operators for neural local search.
//! We only implement stochastics versions of the operators

use std::collections::HashSet;
use std::sync::Arc;

use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::learning::consformer::TrainingDestroy;
use crate::modelling::Problem;

/// Selects, for a single assignment row, the indices of variables to
/// unassign ("destroy") this iteration.
pub trait DestroyOperator: Send + Sync {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;

    /// Called once by `NeuralLocalSearch::run`, before the destroy/repair loop starts, with the
    /// exact set of problems that run will search over. Stateless operators ignore it.
    fn on_run_start(&self, _problems: &[Arc<Problem>]) {}
}

impl<T: DestroyOperator> TrainingDestroy for T {
    fn select(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        self.destroy(problem, assignment, rng)
    }
}

/// Hard-violation info for the current assignment: how many currently
/// unsatisfied constraints each variable participates in, and which
/// constraints are violated.
struct ViolationInfo {
    per_variable: Vec<usize>,
}

impl ViolationInfo {
    fn compute(problem: &Problem, assignment: &[isize]) -> Self {
        let mut per_variable = vec![0usize; problem.number_variables()];
        for c in problem.iter_constraints() {
            if !problem[c].is_satisfied(assignment) {
                for v in problem[c].iter_scope() {
                    per_variable[v.0] += 1;
                }
            }
        }
        ViolationInfo { per_variable }
    }
}

/// Uniformly destroys a `fraction` of the free (domain size > 1) variables.
/// Corresponds to the original ConsFormer's random subset selection.
pub struct RandomDestroy {
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
pub struct WorstDestroy {
    pub fraction: f64,
}

impl DestroyOperator for WorstDestroy {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let info = ViolationInfo::compute(problem, assignment);
        sigmoid_select(problem, &info.per_variable, self.fraction, rng)
    }
}

/// Stochastic related removal (Shaw): draws a random subset of constraints
/// and destroys every variable in their scope. Constraints are drawn
/// independently with a probability rescaled so the expected number of
/// destroyed variables tracks `fraction * n`, regardless of how many
/// constraints the problem has relative to its variables
pub struct RelatedDestroy {
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
        // overlap between scopes) -- solve for p_constraint so this tracks
        // fraction * n.
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

fn free_variables(problem: &Problem) -> Vec<usize> {
    problem
        .iter_variables()
        .filter(|&v| problem[v].domain_size() > 1)
        .map(|v| v.0)
        .collect()
}

fn sigmoid_select(
    problem: &Problem,
    scores: &[usize],
    fraction: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    let free = free_variables(problem);
    if free.is_empty() {
        return free;
    }

    let n = free.len() as f64;
    let values: Vec<f64> = free.iter().map(|&i| scores[i] as f64).collect();
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
    let scale = fraction / (probs.iter().sum::<f64>() / n + SCORE_EPS);

    free.into_iter()
        .zip(probs)
        .filter(|&(_, p)| rng.random_bool((p * scale).clamp(0.0, 1.0)))
        .map(|(i, _)| i)
        .collect()
}

const SCORE_EPS: f64 = 1e-8;

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
