//! Repair of an assignment by belief propagation over the MDDs of the constraints.
//!
//! # Meaning
//!
//! A search frees some variables of an assignment and asks for new values for them. [`MddRepair`]
//! gives them values that agree with the constraints: it combines a probability per value and
//! variable, whatever produced it (a trained network, a uniform guess), with belief propagation
//! over the decomposed MDDs of the problem, one per constraint, then takes the value of largest
//! marginal or samples one.
//!
//! # How it works
//!
//! The decomposed MDDs depend on the problem alone, not on the iteration. [`MddRepair::prepare`]
//! compiles them once per problem before the search, and the search owns the result, a
//! [`Compiled`] per problem, which it hands back to [`MddRepair::repair`] and drops as soon as the
//! problem is solved. The memory of a finished problem is freed at once and the repair itself
//! keeps no state between runs.
//!
//! [`BatchRepair`] holds those for the rows of a batch and repairs all of them in parallel.
//!
//! # Design notes
//!
//! - **Sampling follows the rng it is given.** [`MddRepair::repair`] runs for many rows in
//!   parallel, so each row gets its own generator, seeded beforehand; the process-wide one would
//!   make the result depend on the thread schedule.

use std::collections::HashMap;
use std::sync::Arc;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::bp::belief_propagation;
use crate::mdd::arena::{MddCompilationConfig, compile_constraints};
use crate::mdd::{CompiledConstraint, MddArena};
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::utils::{DecodeMode, argmax};

/// The decomposed MDDs of a problem, one compiled constraint per constraint. The `Arc` is shared
/// by the copies of the same problem in a batch.
pub(crate) type Compiled = Arc<Vec<CompiledConstraint>>;

/// Repairs assignments with belief propagation over the decomposed MDDs of the problem (one per
/// constraint, not one diagram for the whole problem): the value of largest marginal, or a sample
/// of the marginal, according to `mode`.
pub(crate) struct MddRepair {
    /// How the constraints are compiled.
    compilation: MddCompilationConfig,
    /// Size of the nominal domain `0..domain_size` that the constraints are compiled over, which
    /// is the number of values the probabilities hold per variable, see `crate::mdd::arena`'s
    /// module doc.
    domain_size: usize,
    /// How a value is taken from a marginal.
    mode: DecodeMode,
    /// Rounds of belief propagation per repair.
    bp_iterations: usize,
}

impl MddRepair {
    /// `domain_size` is the number of values per variable of the probabilities given to
    /// [`MddRepair::repair`]: the nominal alphabet every shared constraint structure is compiled
    /// against, see `crate::mdd::arena`'s module doc.
    pub(crate) fn new(
        compilation: MddCompilationConfig,
        domain_size: usize,
        mode: DecodeMode,
        bp_iterations: usize,
    ) -> Self {
        Self {
            compilation,
            domain_size,
            mode,
            bp_iterations,
        }
    }

    /// Smallest alphabet that covers every value any variable of any of `problems` can take.
    pub(crate) fn domain_size_covering(problems: &[Arc<Problem>]) -> usize {
        problems
            .iter()
            .flat_map(|problem| {
                problem
                    .iter_variables()
                    .flat_map(|v| problem[v].iter_domain().collect::<Vec<_>>())
            })
            .map(|value| value as usize + 1)
            .max()
            .unwrap_or(1)
    }

    /// Below this many distinct problems, `prepare` shows no progress bar: it would finish
    /// before it could be read.
    const PROGRESS_BAR_MIN_PROBLEMS: usize = 4;

    /// Compiles the constraints of every distinct problem of `problems` in parallel.
    ///
    /// The arena that shares the structures of the constraints is local to the call: it is
    /// dropped when `prepare` returns, and a diagram then lives exactly as long as the
    /// [`Compiled`] values that use it. A diagram that no other problem shares is therefore freed
    /// with its problem. An arena pays off when the constraints of the problems overlap a lot, as
    /// with one small diagram per constraint; with large grouped diagrams that nothing shares, it
    /// only adds a lookup, and the diagrams should be compiled directly into the list of the
    /// problem.
    pub(crate) fn prepare(&self, problems: &[Arc<Problem>]) -> Vec<Compiled> {
        let arena = MddArena::new();
        // The same problem can appear several times in a batch, for instance to run several
        // searches on it: it is compiled once.
        let mut first_of: HashMap<usize, usize> = HashMap::new();
        let mut unique: Vec<&Arc<Problem>> = Vec::new();
        let slots: Vec<usize> = problems
            .iter()
            .map(|problem| {
                *first_of
                    .entry(Arc::as_ptr(problem) as usize)
                    .or_insert_with(|| {
                        unique.push(problem);
                        unique.len() - 1
                    })
            })
            .collect();

        let compile = |problem: &&Arc<Problem>| {
            Arc::new(compile_constraints(
                problem,
                &arena,
                &self.compilation,
                self.domain_size,
            ))
        };
        let compiled: Vec<Arc<Vec<CompiledConstraint>>> =
            if unique.len() < Self::PROGRESS_BAR_MIN_PROBLEMS {
                crate::utils::worker_pool().install(|| unique.par_iter().map(compile).collect())
            } else {
                let progress = ProgressBar::new(unique.len() as u64);
                progress.set_style(
                    ProgressStyle::with_template(
                        "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                    )
                    .expect("hard-coded progress bar template should always be valid"),
                );
                progress.set_message("Compiling MDDs");
                let compiled = crate::utils::worker_pool().install(|| {
                    unique
                        .par_iter()
                        .progress_with(progress.clone())
                        .map(compile)
                        .collect()
                });
                progress.finish_and_clear();
                compiled
            };

        slots
            .into_iter()
            .map(|slot| Arc::clone(&compiled[slot]))
            .collect()
    }

    /// The next assignment of `problem`, given its `current` assignment and the `free` variables
    /// the search may change: the other values are kept. `probability(variable, value)` is the
    /// probability of a raw value for a variable, over `0..domain_size`.
    ///
    /// `rng` makes the sampling of a stochastic repair reproducible: each row should get its own,
    /// seeded before the rows are repaired in parallel.
    ///
    /// Returns `current` unchanged when nothing is free, which needs no `compiled`.
    pub(crate) fn repair(
        &self,
        problem: &Problem,
        compiled: Option<&Compiled>,
        current: &[isize],
        free: &[usize],
        rng: &mut StdRng,
        probability: impl Fn(usize, usize) -> f64,
    ) -> Vec<isize> {
        if free.is_empty() {
            return current.to_vec();
        }
        let compiled =
            compiled.expect("the MDD repair needs its problems prepared before it repairs them");
        let n = current.len();
        let domain_width = self.domain_size;

        let mut assignment = vec![ValueIndex(0); n];
        // The opposite of `free`: everything the destroy/repair loop is holding fixed this round.
        let mut decided = vec![true; n];
        for &v in free {
            decided[v] = false;
        }
        let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        for v in 0..n {
            let variable = VariableIndex(v);
            // `compiled`'s shared structures are compiled with no per-instance domain
            // restriction (see `crate::mdd::arena`'s module doc), so a `ValueIndex` in one of
            // their edges is exactly the raw domain value -- no translation through `problem`'s
            // own (possibly narrower) domain enumeration.
            assignment[v] = ValueIndex(current[v] as usize);
            // Masked the same way `layer_weights_from_probs` masks a training weight: a raw
            // value outside `problem`'s actual (possibly narrowed -- e.g. a fixed/given
            // position) domain gets zero weight, even though the shared structure has a real
            // edge for it -- see that function's doc.
            let probs_v: Vec<f64> = (0..domain_width)
                .map(|raw_value| {
                    if problem[variable].in_domain(raw_value as isize) {
                        probability(v, raw_value)
                    } else {
                        0.0
                    }
                })
                .collect();
            probs.push(probs_v);
        }

        let marginals =
            belief_propagation(compiled, &probs, &assignment, &decided, self.bp_iterations);

        let mut next = current.to_vec();
        for &v in free {
            // `marginals[v]` is indexed by raw domain value directly (see the masked `probs_v`
            // build above -- length `domain_width`, position == raw value), so `chosen` needs no
            // translation back through `problem`'s own domain enumeration.
            let chosen = match self.mode {
                DecodeMode::Greedy => argmax(&marginals[v]),
                DecodeMode::Sample => sample_weights(&marginals[v], rng),
            };
            next[v] = chosen as isize;
        }
        next
    }
}

/// An [`MddRepair`] prepared for the rows of a batch: it owns the compiled MDDs of every row and
/// repairs all the rows at once.
pub(crate) struct BatchRepair {
    /// Repairs one row.
    repair: MddRepair,
    /// What was compiled for each row, `None` once the row is solved.
    compiled: Vec<Option<Compiled>>,
}

impl BatchRepair {
    /// Compiles the constraints of `problems`, one row each.
    pub(crate) fn new(repair: MddRepair, problems: &[Arc<Problem>]) -> Self {
        let compiled = repair.prepare(problems).into_iter().map(Some).collect();
        Self { repair, compiled }
    }

    /// Repairs, in parallel, the variables each row of `destroyed` frees, in `assignments`.
    /// `seeds` holds one seed per row for its sampling, and `probability(row, variable, value)`
    /// the probability of a raw value for a variable of a row.
    pub(crate) fn repair_rows(
        &self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        destroyed: &[Vec<usize>],
        seeds: &[u64],
        probability: impl Fn(usize, usize, usize) -> f64 + Sync,
    ) {
        let next: Vec<Vec<isize>> = crate::utils::worker_pool().install(|| {
            (0..problems.len())
                .into_par_iter()
                .map(|row| {
                    let mut rng = StdRng::seed_from_u64(seeds[row]);
                    self.repair.repair(
                        &problems[row],
                        self.compiled[row].as_ref(),
                        &assignments[row],
                        &destroyed[row],
                        &mut rng,
                        |variable, value| probability(row, variable, value),
                    )
                })
                .collect()
        });
        for (row, values) in assignments.iter_mut().zip(next) {
            *row = values;
        }
    }

    /// Frees what was compiled for the rows flagged in `solved`.
    pub(crate) fn solved(&mut self, solved: &[bool]) {
        for (compiled, &is_solved) in self.compiled.iter_mut().zip(solved) {
            if is_solved {
                *compiled = None;
            }
        }
    }

    /// Drops the rows not flagged in `keep`.
    pub(crate) fn retain(&mut self, keep: &[bool]) {
        self.compiled = super::filtered(std::mem::take(&mut self.compiled), keep);
    }
}

/// An index drawn with probability proportional to its weight.
fn sample_weights(weights: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = weights.iter().sum();
    let mut target = rng.random_range(0.0..total);
    for (index, &weight) in weights.iter().enumerate() {
        target -= weight;
        if target <= 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use rand::SeedableRng;

    /// `size`-variable clique via a single `all_different` over `colours` values, compiled as its
    /// own diagram.
    fn clique_problem(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect(), None);
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    fn repairer(domain_size: usize) -> MddRepair {
        MddRepair::new(
            MddCompilationConfig::default(),
            domain_size,
            DecodeMode::Greedy,
            1,
        )
    }

    #[test]
    fn prepare_returns_one_state_per_problem_in_order() {
        let problems = vec![clique_problem(3, 3), clique_problem(4, 4)];
        let compiled = repairer(4).prepare(&problems);
        assert_eq!(compiled.len(), 2);
        assert_eq!(compiled[0].len(), 1);
        assert_eq!(compiled[1][0].number_decisions(), 4);
    }

    #[test]
    fn prepare_compiles_each_distinct_problem_once_even_with_duplicates() {
        let a = clique_problem(6, 5);
        let b = clique_problem(4, 4);

        // `a` repeated three times (several searches on the same problem) plus `b` once: the
        // copies of `a` share one compiled list.
        let compiled = repairer(5).prepare(&[a.clone(), a.clone(), a.clone(), b.clone()]);

        assert!(Arc::ptr_eq(&compiled[0], &compiled[1]));
        assert!(Arc::ptr_eq(&compiled[0], &compiled[2]));
        assert!(!Arc::ptr_eq(&compiled[0], &compiled[3]));
    }

    #[test]
    fn covering_domain_size_is_the_largest_value_plus_one() {
        let problems = vec![clique_problem(2, 2), clique_problem(2, 5)];
        assert_eq!(MddRepair::domain_size_covering(&problems), 5);
    }

    #[test]
    fn repair_changes_only_the_free_positions_and_stays_in_the_domain() {
        let problems = vec![clique_problem(3, 3)];
        let repairer = repairer(3);
        let compiled = repairer.prepare(&problems);

        let next = repairer.repair(
            &problems[0],
            Some(&compiled[0]),
            &[0, 1, 2],
            &[0],
            &mut StdRng::seed_from_u64(0),
            |_, _| 1.0 / 3.0,
        );
        assert_eq!(next[1..], [1, 2]);
        assert!((0..3).contains(&next[0]));
    }

    #[test]
    fn a_row_without_free_position_is_kept_and_needs_no_compiled_state() {
        let problems = vec![clique_problem(3, 3)];
        let next = repairer(3).repair(
            &problems[0],
            None,
            &[2, 1, 0],
            &[],
            &mut StdRng::seed_from_u64(0),
            |_, _| 0.0,
        );
        assert_eq!(next, [2, 1, 0]);
    }
}
