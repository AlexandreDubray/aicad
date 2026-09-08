//! Destroy operators for neural local search.
//! We only implement stochastics versions of the operators

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;

use crate::modelling::Problem;

/// Selects, for a single assignment row, the indices of variables to
/// unassign ("destroy") this iteration.
pub trait DestroyOperator: Send + Sync {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;

    /// Called once by `NeuralLocalSearch::run`, before the destroy/repair loop starts, with the
    /// exact set of problems that run will search over. Stateless operators ignore it.
    /// Operators that track per-problem state across iterations (`WeightedRelatedDestroy`)
    /// use it to (re)initialise that state.
    fn on_run_start(&self, _problems: &[Arc<Problem>]) {}
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

/// Stochastic related removal (Shaw) with adaptive per-constraint weights, in the spirit of
/// Guided Local Search's penalised "features" and SAT clause-weighting local search: rather than
/// firing every constraint with the same probability regardless of how hard it actually is to
/// satisfy (`RelatedDestroy`'s scheme), each constraint carries a weight that grows every time
/// it's still violated after the previous repair step, and relaxes back towards the 1.0 baseline
/// otherwise. A constraint that's chronically hard to satisfy accumulates weight and
/// gets destroyed often, while constraints that are already easily satisfied fall back towards
/// being fired at roughly the baseline `RelatedDestroy` rate.
///
/// Weights are tracked per problem (keyed by the `Problem`'s address, reset in `on_run_start`.
pub struct WeightedRelatedDestroy {
    pub fraction: f64,
    /// How much a still-violated constraint's weight grows each time `destroy` sees it violated.
    pub bump: f64,
    /// Per-call multiplicative pull of every *non*-violated constraint's weight back towards the
    /// 1.0 baseline, in `[0, 1]`. `1.0` disables decay (a constraint that was ever violated keeps
    /// its accumulated weight forever); lower values forget past violations faster.
    pub decay: f64,
    weights: Mutex<HashMap<usize, Vec<f64>>>,
}

impl WeightedRelatedDestroy {
    pub fn new(fraction: f64, bump: f64, decay: f64) -> Self {
        WeightedRelatedDestroy {
            fraction,
            bump,
            decay,
            weights: Mutex::new(HashMap::new()),
        }
    }

    /// Address-based identity for a `Problem`, shared between `on_run_start` (which is handed
    /// `Arc<Problem>`s) and `destroy` (which only sees a `&Problem` deref'd from one of them, but
    /// at the same address).
    fn key(problem: &Problem) -> usize {
        problem as *const Problem as usize
    }
}

impl DestroyOperator for WeightedRelatedDestroy {
    fn on_run_start(&self, problems: &[Arc<Problem>]) {
        let mut weights = self.weights.lock().expect("weights mutex poisoned");
        weights.clear();
        for problem in problems {
            weights.insert(Self::key(problem), vec![1.0; problem.number_constraints()]);
        }
    }

    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let n = problem.number_variables().max(1);
        let m = problem.number_constraints();
        if m == 0 {
            return Vec::new();
        }

        let mut weights = self.weights.lock().expect("weights mutex poisoned");
        let w = weights
            .entry(Self::key(problem))
            .or_insert_with(|| vec![1.0; m]);
        // Defensive: this should never trigger once `on_run_start` has been called with this
        // run's problems, but guards against a stale/mismatched entry (e.g. address reuse across
        // `run` calls without going through `on_run_start`) indexing out of bounds below.
        if w.len() != m {
            *w = vec![1.0; m];
        }

        for (i, c) in problem.iter_constraints().enumerate() {
            if !problem[c].is_satisfied(assignment) {
                w[i] += self.bump;
            } else {
                w[i] = 1.0 + (w[i] - 1.0) * self.decay;
            }
        }

        let total_scope: usize = problem
            .iter_constraints()
            .map(|c| problem[c].iter_scope().count())
            .sum();
        let avg_scope = (total_scope as f64 / m as f64).max(1.0);
        let total_weight: f64 = w.iter().sum();

        let mut destroyed = HashSet::new();
        if total_weight > 0.0 {
            // Same E[# destroyed] ~= fraction * n derivation as `RelatedDestroy`, but with
            // constraint c's firing probability scaled by its share of the total weight instead
            // of every constraint getting the same probability -- reduces to `RelatedDestroy`
            // exactly when every weight is 1.0.
            for (i, c) in problem.iter_constraints().enumerate() {
                let p_c =
                    (self.fraction * n as f64 * w[i] / (avg_scope * total_weight)).clamp(0.0, 1.0);
                if rng.random_bool(p_c) {
                    for v in problem[c].iter_scope() {
                        if problem[v].domain_size() > 1 {
                            destroyed.insert(v.0);
                        }
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

#[cfg(test)]
mod test_weighted_related_destroy {
    use super::*;
    use crate::modelling::not_equals;
    use rand::SeedableRng;

    /// Two independent `not_equals` constraints over disjoint pairs of variables: `a` is
    /// `not_equals(x0, x1)`, `b` is `not_equals(x2, x3)`. Callers control which one is violated
    /// by picking the assignment.
    fn two_constraint_problem() -> Arc<Problem> {
        let mut problem = Problem::default();
        let x0 = problem.add_variable(vec![0, 1], None);
        let x1 = problem.add_variable(vec![0, 1], None);
        let x2 = problem.add_variable(vec![0, 1], None);
        let x3 = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x0, x1);
        not_equals(&mut problem, x2, x3);
        Arc::new(problem)
    }

    #[test]
    fn chronically_violated_constraint_is_destroyed_far_more_often_than_a_satisfied_one() {
        let problem = two_constraint_problem();
        // `a` (vars 0,1) stays violated (x0 == x1); `b` (vars 2,3) stays satisfied (x2 != x3).
        let assignment = [0isize, 0, 0, 1];

        let op = WeightedRelatedDestroy::new(0.5, 1.0, 0.5);
        op.on_run_start(std::slice::from_ref(&problem));

        let mut rng = StdRng::seed_from_u64(42);
        // Let weights diverge: `a` keeps growing (always violated), `b` keeps decaying back
        // towards 1.0 (always satisfied).
        for _ in 0..100 {
            op.destroy(&problem, &assignment, &mut rng);
        }

        // Now measure firing frequency of each constraint's scope over many trials, without
        // resetting state (so the accumulated weight skew from the loop above is in effect).
        let (mut a_hits, mut b_hits) = (0u32, 0u32);
        for _ in 0..500 {
            let destroyed = op.destroy(&problem, &assignment, &mut rng);
            if destroyed.contains(&0) || destroyed.contains(&1) {
                a_hits += 1;
            }
            if destroyed.contains(&2) || destroyed.contains(&3) {
                b_hits += 1;
            }
        }

        assert!(
            a_hits > b_hits * 3,
            "chronically-violated constraint should be destroyed much more often: a={a_hits} b={b_hits}"
        );
    }

    #[test]
    fn on_run_start_resets_accumulated_weights() {
        let problem = two_constraint_problem();
        let assignment = [0isize, 0, 0, 1];

        let op = WeightedRelatedDestroy::new(0.5, 1.0, 0.5);
        op.on_run_start(std::slice::from_ref(&problem));

        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            op.destroy(&problem, &assignment, &mut rng);
        }

        // Reset: weights should go back to the uniform 1.0 baseline for every constraint.
        op.on_run_start(std::slice::from_ref(&problem));
        let weights = op.weights.lock().unwrap();
        let w = &weights[&WeightedRelatedDestroy::key(&problem)];
        assert_eq!(w, &vec![1.0, 1.0]);
    }
}
