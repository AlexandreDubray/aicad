//! Repair of an assignment over MDDs: by belief propagation over the MDDs of the constraints, or
//! along a path of a relaxed MDD of the whole problem.
//!
//! # Meaning
//!
//! A search frees some variables of an assignment and asks for new values for them. [`MddRepair`]
//! gives them values that agree with the constraints: it combines a probability per value and
//! variable, whatever produced it (a trained network, a uniform guess), with belief propagation
//! over the decomposed MDDs of the problem, one per constraint, then takes the value of largest
//! marginal or samples one.
//!
//! [`RelaxedMddRepair`] instead compiles one relaxed MDD of the whole problem, of bounded width,
//! and samples a path of it, each edge weighted by the probability of its value: the freed
//! variables take the values of the path, the others keep theirs (almost surely). A relaxed MDD
//! holds every solution of its problem, so one without any path proves the problem
//! unsatisfiable.
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
//! The relaxed MDDs are prepared the same way, once per problem and in parallel, by
//! [`RelaxedMddRepair::prepare`], and held for a batch by [`BatchRelaxedRepair`]. A problem whose
//! diagram has no path is reported by [`BatchRelaxedRepair::unsatisfiable`] before the search
//! steps it, and one whose diagram yields a solution at once is given that solution at its first
//! repair. Otherwise each repair weights the layers with the probabilities of their variable (1
//! on the current value of a kept variable) and samples a path.
//!
//! # Design notes
//!
//! - **Sampling follows the rng it is given.** [`MddRepair::repair`] runs for many rows in
//!   parallel, so each row gets its own generator, seeded beforehand; the process-wide one would
//!   make the result depend on the thread schedule. [`RelaxedMddRepair::repair`] does not follow
//!   it yet: the sampling of a path takes no generator, so its repairs are not reproducible.
//! - **A probability is never zero on the relaxed MDD.** A confident network gives some values a
//!   probability that rounds to 0 in `f32`; a node whose edges all lead to such values would
//!   leave nothing to sample, so each weight of a freed variable is at least a small floor.
//! - **Compiling a relaxed MDD is the costly part.** Its memory grows with the number of layers,
//!   the width and the number of constraints, and a whole batch is compiled at once: wide
//!   diagrams of large problems need small batches.

use std::collections::HashMap;
use std::sync::Arc;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::bp::{BpRounds, belief_propagation};
use crate::mdd::arena::{MddCompilationConfig, compile_constraints};
use crate::mdd::heuristics::{
    CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
};
use crate::mdd::{CompiledConstraint, Compiler, Mdd, MddArena};
use crate::modelling::{ConstraintIndex, Problem, ValueIndex, VariableIndex};
use crate::sampling::{DecodeMode, argmax, sample_categorical_with, sample_solution_with};

/// The decomposed MDDs of a problem, one compiled constraint per constraint. The `Arc` is shared
/// by the copies of the same problem in a batch.
pub type Compiled = Arc<Vec<CompiledConstraint>>;

/// What [`RelaxedMddRepair`] compiles for a problem before the search.
pub struct PreparedRelaxedMdd {
    /// The relaxed MDD of the whole problem.
    mdd: Mdd,
    /// A solution of the problem read from a path of the diagram, when the first path found is
    /// one: the repair returns it at once.
    exact_path: Option<Vec<isize>>,
}

/// Repairs assignments along the paths of a relaxed MDD of the whole problem, of at most `width`
/// nodes per layer, weighted by a probability per value and variable.
pub struct RelaxedMddRepair {
    /// The largest number of nodes of a layer of the relaxed MDD.
    pub width: usize,
}

impl RelaxedMddRepair {
    /// The smallest weight a value of a free variable gets, so that every edge of the relaxed
    /// MDD can still be sampled.
    const MIN_WEIGHT: f64 = 1e-12;

    /// A repair on relaxed MDDs of at most `width` nodes per layer.
    pub fn new(width: usize) -> Self {
        Self { width }
    }

    /// Compiles the relaxed MDD of each problem, in parallel, one entry per problem in order.
    pub fn prepare(&self, problems: &[Arc<Problem>]) -> Vec<Option<PreparedRelaxedMdd>> {
        let progress = ProgressBar::new(problems.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");
        crate::utils::worker_pool().install(|| {
            problems
                .par_iter()
                .progress_with(progress.clone())
                .map(|problem: &Arc<Problem>| {
                    let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();

                    let mut compiler = Compiler::new(
                        Arc::clone(problem),
                        CompilationHeuristic {
                            ordering: OrderingHeuristic::MinDomMaxLinked,
                            merge: MergeHeuristic::StateSimilarity,
                            selection: SelectHeuristic::Greedy,
                        },
                        &constraints,
                    );

                    compiler.refine(self.width, usize::MAX);
                    let mdd = compiler.into_mdd();

                    let n = problem.number_variables();

                    let exact_path = mdd.get_solution().and_then(|(values, variables)| {
                        let mut solution: Vec<isize> = (0..n)
                            .map(|v| {
                                problem[VariableIndex(v)]
                                    .iter_domain()
                                    .next()
                                    .expect("Variables should not have empty domain")
                            })
                            .collect();

                        for (value, variable) in values.into_iter().zip(variables) {
                            solution[variable.0] = value;
                        }

                        problem.is_solution(&solution).then_some(solution)
                    });
                    Some(PreparedRelaxedMdd { mdd, exact_path })
                })
                .collect()
        })
    }
    /// The next assignment of `problem`, given its `current` assignment and the `free` variables
    /// the search may change: the other values are kept. `probability(variable, value)` is the
    /// probability of a raw value for a variable, over `0..domain_size`.
    ///
    /// `rng` is meant to make the sampling reproducible, as in the BP search.
    ///
    /// Return `current` unchanged when nothing is free and when the problem is unsat.
    /// If there exists an exact path, it is returned.
    pub fn repair(
        &self,
        problem: &Problem,
        prepared: Option<&PreparedRelaxedMdd>,
        current: &[isize],
        free: &[usize],
        rng: &mut StdRng,
        probability: impl Fn(usize, usize) -> f64,
    ) -> Vec<isize> {
        if free.is_empty() {
            return current.to_vec();
        }

        let n = problem.number_variables();

        // The opposite of `free` : those are the variables that are not destroyed.
        let mut kept: Vec<bool> = vec![true; n];
        for &v in free {
            kept[v] = false;
        }

        let PreparedRelaxedMdd { mdd, exact_path } =
            prepared.expect("The relaxed MDD repair expects its problem prepared before search");
        let mut next = current.to_vec();

        // Before sampling the MDD, we check if there is either
        // no s->t path at all, or if there is an exact one.
        if mdd.is_unsat() {
            return next;
        }

        if let Some(exact_path) = &exact_path {
            for (v, &value) in exact_path.iter().enumerate() {
                next[v] = value;
            }
            return next;
        }

        let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        for (v, &current_value) in current.iter().enumerate() {
            let variable = VariableIndex(v);

            // We reconstruct the probability vector of the variable
            // If it's to be kept, then we assign a probability of 1 to its raw value
            // Otherwise we copy the probability provided.
            let probs_v = problem[variable]
                .iter_domain()
                .map(|raw_value| {
                    if kept[variable.0] {
                        if raw_value == current_value {
                            1.0
                        } else {
                            10e-9
                        }
                    } else {
                        // A confident network gives some values a probability that rounds to
                        // exactly 0 in f32. On a node whose edges all lead to such values,
                        // `sample_solution` would have nothing to sample.
                        probability(v, raw_value as usize).max(RelaxedMddRepair::MIN_WEIGHT)
                    }
                })
                .collect();
            probs.push(probs_v);
        }

        // We then use the probabilities to get the weights as it is structured in the relaxed MDD:
        // the row of a layer is the probability vector of the variable branched on there, so the
        // kept variables stay (almost surely) on their current value.
        let weights: Vec<Vec<f64>> = (0..mdd.number_layers() - 1)
            .map(|layer| probs[mdd.decision_at_layer(layer).0].clone())
            .collect();

        // And use it to sample a path, which gives a new assignment to return.
        let path = sample_solution_with(mdd, &weights, rng);
        let (values, variables) = mdd.solution_from_path(&path);

        for (&value, variable) in values.iter().zip(variables) {
            next[variable.0] = value;
        }

        next
    }
}

/// A [`RelaxedMddRepair`] prepared for the rows of a batch: it owns the relaxed MDDs of every row
/// and repairs all the rows at once.
pub struct BatchRelaxedRepair {
    /// Repairs one row.
    repair: RelaxedMddRepair,
    /// What was compiled for each row, `None` once the row is solved.
    prepared: Vec<Option<PreparedRelaxedMdd>>,
}

impl BatchRelaxedRepair {
    /// Compiles the relaxed MDDs of `problems`, one row each.
    pub fn new(repair: RelaxedMddRepair, problems: &[Arc<Problem>]) -> Self {
        let prepared = repair.prepare(problems);
        Self { repair, prepared }
    }

    /// Repairs, in parallel, the variables each row of `destroyed` frees, in `assignments`.
    /// `seeds` holds one seed per row for its sampling, and `probability(row, variable, value)`
    /// the probability of a raw value for a variable of a row.
    pub fn repair_rows(
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
                        self.prepared[row].as_ref(),
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
    /// Whether the relaxed MDD of `row` has no path from the root to the sink. A relaxed MDD holds
    /// every solution of its problem, so this proves the problem has none. `false` once the row
    /// is solved, as nothing is kept for it then.
    pub fn unsatisfiable(&self, row: usize) -> bool {
        self.prepared[row]
            .as_ref()
            .is_some_and(|prepared| prepared.mdd.is_unsat())
    }

    /// Frees what was compiled for the rows flagged in `solved`.
    pub fn solved(&mut self, solved: &[bool]) {
        for (prepared, &is_solved) in self.prepared.iter_mut().zip(solved) {
            if is_solved {
                *prepared = None;
            }
        }
    }

    /// Drops the rows not flagged in `keep`.
    pub fn retain(&mut self, keep: &[bool]) {
        self.prepared = super::filtered(std::mem::take(&mut self.prepared), keep);
    }
}

/// Repairs assignments with belief propagation over the decomposed MDDs of the problem (one per
/// constraint, not one diagram for the whole problem): the value of largest marginal, or a sample
/// of the marginal, according to `mode`.
pub struct MddRepair {
    /// How the constraints are compiled.
    compilation: MddCompilationConfig,
    /// Size of the nominal domain `0..domain_size` that the constraints are compiled over, which
    /// is the number of values the probabilities hold per variable, see `crate::mdd::arena`'s
    /// module doc.
    domain_size: usize,
    /// How a value is taken from a marginal.
    mode: DecodeMode,
    /// Rounds of belief propagation per repair.
    bp_rounds: BpRounds,
}

impl MddRepair {
    /// `domain_size` is the number of values per variable of the probabilities given to
    /// [`MddRepair::repair`]: the nominal alphabet every shared constraint structure is compiled
    /// against, see `crate::mdd::arena`'s module doc.
    pub fn new(
        compilation: MddCompilationConfig,
        domain_size: usize,
        mode: DecodeMode,
        bp_rounds: impl Into<BpRounds>,
    ) -> Self {
        Self {
            compilation,
            domain_size,
            mode,
            bp_rounds: bp_rounds.into(),
        }
    }

    /// Smallest alphabet that covers every value any variable of any of `problems` can take.
    pub fn domain_size_covering(problems: &[Arc<Problem>]) -> usize {
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
    pub fn prepare(&self, problems: &[Arc<Problem>]) -> Vec<Compiled> {
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
    pub fn repair(
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

        let marginals = belief_propagation(compiled, &probs, &assignment, &decided, self.bp_rounds);

        let mut next = current.to_vec();
        for &v in free {
            // `marginals[v]` is indexed by raw domain value directly (see the masked `probs_v`
            // build above -- length `domain_width`, position == raw value), so `chosen` needs no
            // translation back through `problem`'s own domain enumeration.
            let chosen = match self.mode {
                DecodeMode::Greedy => argmax(&marginals[v]),
                DecodeMode::Sample => sample_categorical_with(&marginals[v], rng),
            };
            next[v] = chosen as isize;
        }
        next
    }
}

/// An [`MddRepair`] prepared for the rows of a batch: it owns the compiled MDDs of every row and
/// repairs all the rows at once.
pub struct BatchRepair {
    /// Repairs one row.
    repair: MddRepair,
    /// What was compiled for each row, `None` once the row is solved.
    compiled: Vec<Option<Compiled>>,
}

impl BatchRepair {
    /// Compiles the constraints of `problems`, one row each.
    pub fn new(repair: MddRepair, problems: &[Arc<Problem>]) -> Self {
        let compiled = repair.prepare(problems).into_iter().map(Some).collect();
        Self { repair, compiled }
    }

    /// Repairs, in parallel, the variables each row of `destroyed` frees, in `assignments`.
    /// `seeds` holds one seed per row for its sampling, and `probability(row, variable, value)`
    /// the probability of a raw value for a variable of a row.
    pub fn repair_rows(
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
    pub fn solved(&mut self, solved: &[bool]) {
        for (compiled, &is_solved) in self.compiled.iter_mut().zip(solved) {
            if is_solved {
                *compiled = None;
            }
        }
    }

    /// Drops the rows not flagged in `keep`.
    pub fn retain(&mut self, keep: &[bool]) {
        self.compiled = super::filtered(std::mem::take(&mut self.compiled), keep);
    }
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
        let vars = problem.add_variables(size, (0..colours as isize).collect());
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
        let problem = clique_problem(3, 3);
        let next = repairer(3).repair(
            &problem,
            None,
            &[2, 1, 0],
            &[],
            &mut StdRng::seed_from_u64(0),
            |_, _| 0.0,
        );
        assert_eq!(next, [2, 1, 0]);
    }

    #[test]
    fn the_relaxed_repair_reports_the_rows_whose_relaxed_mdd_has_no_path() {
        // Four variables pairwise different over three values have no solution; the width is
        // large enough for the diagram to be exact.
        let problems = vec![clique_problem(3, 3), clique_problem(4, 3)];
        let batch = BatchRelaxedRepair::new(RelaxedMddRepair::new(100), &problems);
        assert!(!batch.unsatisfiable(0));
        assert!(batch.unsatisfiable(1));
    }

    #[test]
    fn a_solved_row_is_not_reported_unsatisfiable() {
        let problems = vec![clique_problem(4, 3)];
        let mut batch = BatchRelaxedRepair::new(RelaxedMddRepair::new(100), &problems);
        batch.solved(&[true]);
        assert!(!batch.unsatisfiable(0));
    }

    #[test]
    fn the_relaxed_repair_samples_even_when_the_network_gives_every_value_a_zero_probability() {
        let problem = clique_problem(3, 3);
        let repair = RelaxedMddRepair::new(100);
        let mut prepared = repair.prepare(std::slice::from_ref(&problem));
        // Without its exact path, the repair has to sample one in the diagram.
        prepared[0].as_mut().unwrap().exact_path = None;

        let next = repair.repair(
            &problem,
            prepared[0].as_ref(),
            &[0, 0, 0],
            &[0, 1, 2],
            &mut StdRng::seed_from_u64(0),
            |_, _| 0.0,
        );
        assert!(next.iter().all(|v| (0..3).contains(v)));
    }
}
