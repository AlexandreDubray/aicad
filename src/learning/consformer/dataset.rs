use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use rand::RngExt;
use rayon::prelude::*;

use crate::learning::{Batch, BatchProblems};
use crate::modelling::{Problem, VariableIndex};

use super::ConsFormerInputs;

/// Plain-data (no tensors, no device) form of `consformer_masks`'s output. Computing this is the
/// only non-trivial CPU work in mask construction, so it's split out from tensor building: it can
/// be computed for many problems in parallel (see `ConsFormerDataset::new`), while the actual
/// `Tensor::from_data` calls -- which touch the device, and so are kept single-threaded to avoid
/// any backend-specific thread/context assumptions (e.g. around CUDA) -- happen afterward.
pub(super) struct ConsFormerMaskData {
    number_vars: usize,
    flat_attention_mask: Vec<bool>,
    is_var: Vec<bool>,
}

pub(super) fn consformer_mask_data(problem: &Problem) -> ConsFormerMaskData {
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

    let is_var = problem
        .iter_variables()
        .map(|v| problem[v].domain_size() > 1)
        .collect::<Vec<bool>>();

    ConsFormerMaskData {
        number_vars: n,
        flat_attention_mask,
        is_var,
    }
}

impl ConsFormerMaskData {
    pub(super) fn into_tensors<B: Backend>(
        self,
        device: &B::Device,
    ) -> (Tensor<B, 2, Bool>, Tensor<B, 1, Bool>) {
        let n = self.number_vars;
        let attention_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 1, Bool>::from_data(self.flat_attention_mask.as_slice(), device)
                .reshape([n, n]);
        let var_mask = Tensor::from_data(self.is_var.as_slice(), device);
        (attention_mask, var_mask)
    }
}

/// Computes the attention mask and variable indicator mask for ConsFormer
pub fn consformer_masks<B: Backend>(
    problem: &Problem,
    device: &B::Device,
) -> (Tensor<B, 2, Bool>, Tensor<B, 1, Bool>) {
    consformer_mask_data(problem).into_tensors::<B>(device)
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
        // Mask construction is independent per problem, so the CPU work (building the raw
        // boolean vectors) is spread across a capped worker pool (see `utils::worker_pool` --
        // deliberately not rayon's all-cores default, to leave headroom for the rest of the
        // system and for other jobs on a shared benchmarking machine). Tensor construction from
        // that raw data is kept single-threaded (see `ConsFormerMaskData`).
        let mask_data: Vec<ConsFormerMaskData> = crate::utils::worker_pool().install(|| {
            problems
                .par_iter()
                .map(|problem| consformer_mask_data(problem))
                .collect()
        });

        let samples = problems
            .into_iter()
            .zip(mask_data)
            .map(|(problem, mask_data)| {
                let (attention_mask, var_mask) = mask_data.into_tensors::<B>(device);
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

/// Shared by every ConsFormer-compatible batcher (see `ConsFormerBatcher` and
/// `mdd_dataset::ConsFormerMddBatcher`): stacks the per-sample attention masks, builds the
/// randomly-thinned (`mask_fraction`) variable mask, and samples each problem's initial
/// assignment. Kept here rather than duplicated per batcher, since both recipes need exactly this
/// -- only what they do with the MDD/penalty side of the batch differs.
///
/// `attention_mask_tensors` is consumed (moved into `Tensor::stack`); `var_mask_tensors` and
/// `problems` are only read. All three, plus `problems`, are expected to be in the same
/// sample order and the same length.
pub(super) fn stack_masks_and_sample_assignments<B: Backend>(
    attention_mask_tensors: Vec<Tensor<B, 2, Bool>>,
    var_mask_tensors: &[Tensor<B, 1, Bool>],
    problems: &[Arc<Problem>],
    mask_fraction: f64,
    device: &B::Device,
) -> (Tensor<B, 3, Bool>, Tensor<B, 2, Bool>, Tensor<B, 2, Int>) {
    let attention_masks = Tensor::stack(attention_mask_tensors, 0);

    let n = problems[0].number_variables();
    let mut var_masks_data: Vec<bool> = Vec::with_capacity(problems.len() * n);
    // One `with_rng` call for the whole batch (rather than one per candidate) so a seeded run
    // only takes the shared RNG's lock once here, not once per variable.
    crate::utils::with_rng(|rng| {
        for var_mask in var_mask_tensors {
            let candidates: Vec<i64> = var_mask
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
                    .map(|is_candidate| is_candidate != 0 && rng.random_bool(mask_fraction)),
            );
        }
    });
    let var_masks: Tensor<B, 2, Bool> =
        Tensor::<B, 1, Bool>::from_data(var_masks_data.as_slice(), device)
            .reshape([problems.len(), n]);

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

    (attention_masks, var_masks, assignments)
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
        let attention_mask_tensors: Vec<Tensor<B, 2, Bool>> =
            samples.iter().map(|s| s.attention_mask.clone()).collect();
        let var_mask_tensors: Vec<Tensor<B, 1, Bool>> =
            samples.iter().map(|s| s.var_mask.clone()).collect();
        let problems: Vec<Arc<Problem>> = samples.into_iter().map(|s| s.problem).collect();

        let (attention_masks, var_masks, assignments) = stack_masks_and_sample_assignments(
            attention_mask_tensors,
            &var_mask_tensors,
            &problems,
            self.mask_fraction,
            device,
        );

        ConsFormerBatch {
            assignments,
            attention_masks,
            var_masks,
            problems,
        }
    }
}

impl<B: Backend> BatchProblems<B> for ConsFormerBatch<B> {
    fn problems(&self) -> &[Arc<Problem>] {
        &self.problems
    }
}

impl<B: Backend> ConsFormerInputs<B> for ConsFormerBatch<B> {
    fn attention_masks(&self) -> Tensor<B, 3, Bool> {
        self.attention_masks.clone()
    }

    fn var_masks(&self) -> Tensor<B, 2, Bool> {
        self.var_masks.clone()
    }

    fn assignments(&self) -> Tensor<B, 2, Int> {
        self.assignments.clone()
    }
}

impl<B: Backend> Batch<B> for ConsFormerBatch<B> {
    /// Batch used for inference. Builds one block of `population_size` rows per
    /// problem, in the same order as `problems`, allowing parallel local search
    /// across a whole set of problems (and, within each, a population of candidates)
    /// in a single forward pass.
    fn for_assignments(
        problems: &[Arc<Problem>],
        population_size: usize,
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Bool>,
        device: &B::Device,
    ) -> Self {
        let attention_masks = Tensor::cat(
            problems
                .iter()
                .map(|problem| {
                    let (attention_mask, _) = consformer_masks::<B>(problem, device);
                    attention_mask.unsqueeze::<3>().repeat_dim(0, population_size)
                })
                .collect(),
            0,
        );
        let expanded_problems: Vec<Arc<Problem>> = problems
            .iter()
            .flat_map(|p| std::iter::repeat_with(|| Arc::clone(p)).take(population_size))
            .collect();

        ConsFormerBatch {
            assignments,
            attention_masks,
            var_masks: destroy_mask,
            problems: expanded_problems,
        }
    }
}
