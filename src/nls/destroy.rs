//! Destroy operators for neural local search.
//! We only implement stochastics versions of the operators

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;

use crate::modelling::Problem;

/// Selects, for a single assignment row, the indices of variables to
/// unassign ("destroy") this iteration.
pub trait DestroyOperator: Send + Sync {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;
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

/// Stochastic worst removal: destroys variables with probability
/// proportional to how many currently-violated constraints they take part
/// in, normalised so the expected fraction destroyed matches `fraction`.
pub struct WorstDestroy {
    pub fraction: f64,
}

impl DestroyOperator for WorstDestroy {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let info = ViolationInfo::compute(problem, assignment);
        bernoulli_select(problem, &info.per_variable, self.fraction, rng)
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

/// Selects free variables independently with `pi_i ~= score_i / sum(score) * fraction * n_free`
/// , i.e. proportional to `score`, normalised so the mean
/// selection probability over free variables matches `fraction`. Falls back
/// to a uniform `fraction` when every score is 0 (e.g. a fully satisfied
/// assignment).
fn bernoulli_select(
    problem: &Problem,
    scores: &[usize],
    fraction: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    let free = free_variables(problem);
    if free.is_empty() {
        return free;
    }

    let total: f64 = free.iter().map(|&i| scores[i] as f64).sum();
    if total == 0.0 {
        let p = fraction.clamp(0.0, 1.0);
        return free.into_iter().filter(|_| rng.random_bool(p)).collect();
    }

    let n_free = free.len() as f64;
    free.into_iter()
        .filter(|&i| {
            let pi = (scores[i] as f64 / total) * fraction * n_free;
            rng.random_bool(pi.clamp(0.0, 1.0))
        })
        .collect()
}
