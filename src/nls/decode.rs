//! Decoding operators for neural local search.
//! The following decoding strategies are implemented:
//!     - Use an argmax: Always select the value associated with the highest logit
//!     - Use a softmax: sample proportionnaly to the logits
//!     - Use belief propagation over the problem's compiled MDDs to turn the network's raw,
//!       per-position logits into constraint-propagated marginals before decoding

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

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

        Tensor::<B, 1, Int>::from_data(next_data.as_slice(), &device).reshape([rows, n])
    }
}
