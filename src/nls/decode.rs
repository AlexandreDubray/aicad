//! Decoding operators for neural local search.
//! The following decoding strategies are implemented:
//!     - Use an argmax: Always select the value associated with the highest logit
//!     - Use a softmax: sample proportionnaly to the logits
//!     - Use belief propagation over the problem's compiled MDDs to turn the network's raw,
//!       per-position logits into constraint-propagated marginals before decoding

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::learning::consformer::MddCompilationConfig;
use crate::mdd::Mdd;
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::sampling::bp::belief_propagation;
use crate::sampling::solve::value_to_index;
use crate::sampling::{argmax, sample_categorical, DecodeMode};
use crate::utils::tensor::to_rows;

/// Turns this iteration's logits into the next assignment. Only positions
/// flagged in `destroy_mask` may change; everywhere else the current value
/// is kept, regardless of what the network predicted there.
pub trait DecodingOperator<B: Backend>: Send + Sync {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
    ) -> Tensor<B, 2, Int>;

    fn detect_unsat(&self, _problem: &Arc<Problem>) -> bool {
        false
    }
    fn prepare(&self, _problems: &[Arc<Problem>]) {}
}

/// Greedy / MAP decoding: takes the most likely value per variable.
pub struct Argmax;

impl<B: Backend> DecodingOperator<B> for Argmax {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        _problems: &[Arc<Problem>],
    ) -> Tensor<B, 2, Int> {
        let proposed: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);
        current.mask_where(destroy_mask.equal_elem(1), proposed)
    }
}

/// Stochastic decoding: samples a value per variable from
/// `softmax(logits / temperature)`.
pub struct Sampling {
    pub temperature: f64,
}

impl<B: Backend> DecodingOperator<B> for Sampling {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        _problems: &[Arc<Problem>],
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let u = Tensor::<B, 3>::random(logits.dims(), Distribution::Uniform(1e-20, 1.0), &device);
        let neg_log_u = -u.log(); // -ln(u), > 0 since u in (0, 1)
        let gumbel = -neg_log_u.log(); // Gumbel(0, 1) noise: -ln(-ln(u))

        let scaled = logits.div_scalar(self.temperature) + gumbel;
        let proposed: Tensor<B, 2, Int> = scaled.argmax(2).squeeze_dim(2);
        current.mask_where(destroy_mask.equal_elem(1), proposed)
    }
}

fn compile_mdds_for(problem: &Arc<Problem>, compilation: &MddCompilationConfig) -> Vec<Mdd> {
    compilation
        .grouping
        .groups(problem)
        .into_iter()
        .map(|constraints| {
            let mut mdd = Mdd::new(
                Arc::clone(problem),
                compilation.ordering.clone(),
                compilation.merge,
                compilation.select,
                &constraints,
            );
            mdd.refine(compilation.max_width);
            mdd
        })
        .collect()
}

pub struct BeliefPropagationDecode {
    compilation: MddCompilationConfig,
    iterations: usize,
    mode: DecodeMode,
    cache: Mutex<HashMap<usize, Arc<Vec<Mdd>>>>,
}

impl BeliefPropagationDecode {
    pub fn new(compilation: MddCompilationConfig, iterations: usize, mode: DecodeMode) -> Self {
        Self {
            compilation,
            iterations,
            mode,
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn mdds_for(&self, problem: &Arc<Problem>) -> Arc<Vec<Mdd>> {
        let key = Arc::as_ptr(problem) as usize;
        {
            let cache = self.cache.lock().expect("mdd cache lock poisoned");
            if let Some(mdds) = cache.get(&key) {
                return Arc::clone(mdds);
            }
        }
        let mdds = Arc::new(compile_mdds_for(problem, &self.compilation));
        self.cache
            .lock()
            .expect("mdd cache lock poisoned")
            .insert(key, Arc::clone(&mdds));
        mdds
    }
}

impl<B: Backend> DecodingOperator<B> for BeliefPropagationDecode {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let dims = current.dims();
        let (rows, n) = (dims[0], dims[1]);
        let domain_width = logits.dims()[2];

        let probs_flat: Vec<f32> = softmax(logits, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");
        let current_rows = to_rows(&current, rows, n);
        let mask_rows = to_rows(&destroy_mask, rows, n);

        let mut next_data = vec![0i64; rows * n];
        crate::utils::worker_pool().install(|| {
            next_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(row, next_row)| {
                    let problem = &problems[row];
                    let mdds = self.mdds_for(problem);

                    let mut assignment = vec![ValueIndex(0); n];
                    let mut decided = vec![false; n];
                    let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
                    for v in 0..n {
                        let variable = VariableIndex(v);
                        assignment[v] = value_to_index(problem, variable, current_rows[row][v]);
                        // `destroy_mask == 1` marks a position as free to change this iteration --
                        // `decided` here is its opposite: everything the destroy/repair loop is
                        // holding fixed this round.
                        decided[v] = mask_rows[row][v] == 0;

                        let domain_size = problem[variable].domain_size();
                        let probs_v: Vec<f64> = (0..domain_size)
                            .map(|d| {
                                let value = problem[variable].value(ValueIndex(d));
                                let offset = row * n * domain_width + v * domain_width + value as usize;
                                probs_flat[offset] as f64
                            })
                            .collect();
                        probs.push(probs_v);
                    }

                    let marginals =
                        belief_propagation(&mdds, &probs, &assignment, &decided, self.iterations);

                    for v in 0..n {
                        if mask_rows[row][v] == 0 {
                            // Untouched position -- keep the current value exactly, same contract
                            // `Argmax`/`Sampling` honour via `mask_where`.
                            next_row[v] = current_rows[row][v] as i64;
                            continue;
                        }
                        let chosen = match self.mode {
                            DecodeMode::Greedy => argmax(&marginals[v]),
                            DecodeMode::Sample => sample_categorical(&marginals[v]),
                        };
                        next_row[v] = problem[VariableIndex(v)].value(ValueIndex(chosen)) as i64;
                    }
                });
        });

        Tensor::<B, 1, Int>::from_data(next_data.as_slice(), &device).reshape([rows, n])
    }

    fn detect_unsat(&self, problem: &Arc<Problem>) -> bool {
        self.mdds_for(problem).iter().any(Mdd::is_unsat)
    }

    fn prepare(&self, problems: &[Arc<Problem>]) {
        let mut seen = HashSet::new();
        let unique: Vec<&Arc<Problem>> = problems
            .iter()
            .filter(|p| seen.insert(Arc::as_ptr(*p) as usize))
            .collect();

        let progress = ProgressBar::new(unique.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");

        crate::utils::worker_pool().install(|| {
            unique
                .into_par_iter()
                .progress_with(progress.clone())
                .for_each(|problem| {
                    self.mdds_for(problem);
                });
            progress.finish_and_clear();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdd::heuristics::ConstraintGrouping;
    use crate::modelling::all_different;
    use burn::backend::ndarray::NdArray;

    /// `size`-variable clique via a single `all_different` over `colours < size` values -- always
    /// compiled as its own MDD regardless of grouping, so this exercises `detect_unsat` without
    /// depending on how `ConstraintGrouping::RollingWindow` happens to window the constraints.
    fn clique_problem(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect(), None);
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    fn belief_propagation_decode() -> BeliefPropagationDecode {
        BeliefPropagationDecode::new(
            MddCompilationConfig {
                grouping: ConstraintGrouping::new_rolling(1),
                ..MddCompilationConfig::default()
            },
            5,
            DecodeMode::Greedy,
        )
    }

    #[test]
    fn detect_unsat_is_true_when_a_clique_has_fewer_colours_than_variables() {
        let problem = clique_problem(6, 5);
        let op = belief_propagation_decode();
        assert!(
            <BeliefPropagationDecode as DecodingOperator<NdArray>>::detect_unsat(&op, &problem)
        );
    }

    #[test]
    fn detect_unsat_is_false_when_a_clique_has_enough_colours() {
        let problem = clique_problem(6, 6);
        let op = belief_propagation_decode();
        assert!(
            !<BeliefPropagationDecode as DecodingOperator<NdArray>>::detect_unsat(&op, &problem)
        );
    }

    #[test]
    fn detect_unsat_caches_so_a_second_call_does_not_recompile() {
        let problem = clique_problem(6, 5);
        let op = belief_propagation_decode();
        let first = op.mdds_for(&problem);
        let second = op.mdds_for(&problem);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prepare_warms_the_cache_so_decode_never_needs_to_compile() {
        let problem = clique_problem(6, 5);
        let op = belief_propagation_decode();
        <BeliefPropagationDecode as DecodingOperator<NdArray>>::prepare(&op, &[problem.clone()]);

        // `mdds_for` after `prepare` must be a pure cache hit -- calling it twice more should
        // keep returning the exact same `Arc`, never a freshly compiled one.
        let first = op.mdds_for(&problem);
        let second = op.mdds_for(&problem);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prepare_compiles_each_distinct_problem_once_even_with_duplicates() {
        let a = clique_problem(6, 5);
        let b = clique_problem(4, 4);
        let op = belief_propagation_decode();

        // `a` repeated three times (multiple search samples of the same problem) plus `b` once --
        // `prepare` must still only compile 2 distinct problems, not 4.
        <BeliefPropagationDecode as DecodingOperator<NdArray>>::prepare(
            &op,
            &[a.clone(), a.clone(), a.clone(), b.clone()],
        );

        assert_eq!(op.cache.lock().unwrap().len(), 2);
    }
}
