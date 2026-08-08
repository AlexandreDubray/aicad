use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use rand::RngExt;

use crate::learning::Batch;
use crate::modelling::{Problem, VariableIndex};

/// Computes the attention mask and variable indicator mask for ConsFormer
pub fn consformer_masks<B: Backend>(
    problem: &Problem,
    device: &B::Device,
) -> (Tensor<B, 2, Bool>, Tensor<B, 1, Bool>) {
    let n = problem.number_variables();
    let mut flat_attention_mask = vec![false; n * n];

    // Every variable always attends to itself, unconditionally. Without
    // this, a variable with no constraints at all would have a fully -inf
    // attention row, and softmax over an all -inf row produces NaN
    for i in 0..n {
        flat_attention_mask[i * n + i] = true;
    }

    // Each variable linked in the primal graph gets attention between them.
    for constraint in problem.iter_constraints() {
        let scope = problem[constraint]
            .iter_scope()
            .collect::<Vec<VariableIndex>>();
        for i in 0..scope.len() {
            let u = *scope[i];
            for j in i + 1..scope.len() {
                let v = *scope[j];
                flat_attention_mask[u * n + v] = true;
                flat_attention_mask[v * n + u] = true;
            }
        }
    }

    let attention_mask: Tensor<B, 2, Bool> =
        Tensor::<B, 1, Bool>::from_data(flat_attention_mask.as_slice(), device).reshape([n, n]);

    let is_var = problem
        .iter_variables()
        .map(|v| problem[v].domain_size() > 1)
        .collect::<Vec<bool>>();
    let var_mask = Tensor::from_data(is_var.as_slice(), device);

    (attention_mask, var_mask)
}

/// Sample used to train ConsFormer. We have the problem, the attention mask (derived from the
/// problem), and the var mask (derived from the problem).
/// The attention mask is the primal graph adjacency matrix with the diagonal set to 1
/// (self-attention).
/// The var mask indicates which variable as a domain of size > 1.
/// We pre-compute these tensors beforehand since they are invariant during training
#[derive(Clone)]
pub struct ConsFormerSample<B: Backend> {
    problem: Arc<Problem>,
    attention_mask: Tensor<B, 2, Bool>,
    var_mask: Tensor<B, 1, Bool>,
}

impl<B: Backend> std::fmt::Debug for ConsFormerSample<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerSample")
            .field("number_vars", &self.problem.number_variables())
            .field("attention_mask", &self.attention_mask)
            .field("var_mask", &self.var_mask)
            .finish()
    }
}

/// A dataset is just a vector of samples
pub struct ConsFormerDataset<B: Backend> {
    samples: Vec<ConsFormerSample<B>>,
}

impl<B: Backend> ConsFormerDataset<B> {
    pub fn new(problems: Vec<Arc<Problem>>, device: &B::Device) -> Self {
        // For each problem, compute the attention mask and the variable mask
        let samples = problems
            .into_iter()
            .map(|problem| {
                let (attention_mask, var_mask) = consformer_masks::<B>(&problem, device);
                ConsFormerSample {
                    problem,
                    attention_mask,
                    var_mask,
                }
            })
            .collect();
        Self { samples }
    }
}

impl<B: Backend> Dataset<ConsFormerSample<B>> for ConsFormerDataset<B> {
    fn get(&self, index: usize) -> Option<ConsFormerSample<B>> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Clone, Copy)]
pub struct ConsFormerBatcher {
    pub mask_fraction: f64,
}

/// A batch is just a collection of samples with the addition of initial assignments that are
/// passed to the neural network.
#[derive(Clone)]
pub struct ConsFormerBatch<B: Backend> {
    /// (batch_size, number_vars, number_vars)
    pub attention_masks: Tensor<B, 3, Bool>,
    /// (batch_size, number_vars)
    pub var_masks: Tensor<B, 2, Bool>,
    /// Problems, used to compute the loss
    pub problems: Vec<Arc<Problem>>,
    pub assignments: Tensor<B, 2, Int>,
}

impl<B: Backend> std::fmt::Debug for ConsFormerBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerBatch")
            .field("attention_masks", &self.attention_masks)
            .field("var_masks", &self.var_masks)
            .field("problems", &format!("{} problem(s)", self.problems.len()))
            .finish()
    }
}

impl<B: Backend> Batcher<B, ConsFormerSample<B>, ConsFormerBatch<B>> for ConsFormerBatcher {
    /// Computes the batch from a set of samples. This function just stack the associated tensors
    /// and create initial assignments by sampling uniformly each variable. Note that this respect
    /// the predefined problems (e.g., sudoku with hints) as the fixed variables have domain size
    /// 1, so sampling it always return the same value
    fn batch(&self, samples: Vec<ConsFormerSample<B>>, device: &B::Device) -> ConsFormerBatch<B> {
        let attention_masks = Tensor::stack(
            samples.iter().map(|s| s.attention_mask.clone()).collect(),
            0,
        );

        let n = samples[0].problem.number_variables();
        let mut rng = rand::rng();
        let mut var_masks_data: Vec<bool> = Vec::with_capacity(samples.len() * n);
        for s in &samples {
            let candidates: Vec<i64> = s
                .var_mask
                .clone()
                .int()
                .into_data()
                .to_vec::<B::IntElem>()
                .expect("var mask should convert to int")
                .into_iter()
                .map(|v| v.elem::<i64>())
                .collect();
            var_masks_data.extend(
                candidates
                    .into_iter()
                    .map(|is_candidate| is_candidate != 0 && rng.random_bool(self.mask_fraction)),
            );
        }
        let var_masks: Tensor<B, 2, Bool> =
            Tensor::<B, 1, Bool>::from_data(var_masks_data.as_slice(), device)
                .reshape([samples.len(), n]);

        let problems: Vec<Arc<Problem>> = samples.into_iter().map(|s| s.problem).collect();

        let init = problems
            .iter()
            .map(|problem| {
                problem
                    .iter_variables()
                    .map(|v| problem[v].sample() as i64)
                    .collect::<Vec<i64>>()
            })
            .flatten()
            .collect::<Vec<i64>>();
        let assignments: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::from_data(init.as_slice(), device).reshape([problems.len(), n]);

        ConsFormerBatch {
            assignments,
            attention_masks,
            var_masks,
            problems,
        }
    }
}

impl<B: Backend> Batch<B> for ConsFormerBatch<B> {
    fn problems(&self) -> &[Arc<Problem>] {
        &self.problems
    }

    /// Batch used for inference. This is basically a batch for the same problem. We allow to have
    /// multiple assignments for the same problem, allowing parallel execution of local search.
    fn for_assignments(
        problem: &Arc<Problem>,
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Bool>,
        device: &B::Device,
    ) -> Self {
        let number_assignments = assignments.dims()[0];
        let (attention_mask, _) = consformer_masks::<B>(problem, device);

        let attention_masks = attention_mask
            .unsqueeze::<3>()
            .repeat_dim(0, number_assignments);
        let problems: Vec<Arc<Problem>> = std::iter::repeat_with(|| Arc::clone(problem))
            .take(number_assignments)
            .collect();

        ConsFormerBatch {
            assignments,
            attention_masks,
            var_masks: destroy_mask,
            problems,
        }
    }
}
