//! The data ConsFormer is trained on and the batches it reads.
//!
//! # Meaning
//!
//! A [`ConsFormerDataset`] holds problems together with what ConsFormer needs to know about each:
//! the attention mask (which variables share a constraint) and which variables have a choice to
//! make. A [`ConsFormerBatcher`] turns a few of them into a [`ConsFormerBatch`], the input of the
//! network and of the losses: the stacked masks, a random assignment for every problem, and the
//! variables the network must re-predict. The same batch type is built at inference time by
//! [`Batch::for_assignments`], from the assignments of a local search.
//!
//! # How it works
//!
//! The masks of a problem do not change, so they are computed once, when the dataset is built
//! (the problems in parallel, the tensors afterwards on a single thread). A batch then stacks them
//! and draws a random assignment per problem. The variables to re-predict are either chosen by a
//! destroy operator, as local search does, or drawn at random among the variables with more than
//! one value, each with probability `mask_fraction`.
//!
//! # Design notes
//!
//! - **All problems of a batch have the same variables.** The network reads rectangular tensors and
//!   builds its positional embedding from one problem, so [`Batch::check_problems`] refuses a
//!   batch whose problems have different numbers of variables. Their domains and constraints may
//!   differ.
//! - **Random masks skip the variables without a choice.** A variable with a single value has
//!   nothing to re-predict, so drawing at random never picks it, which keeps the given values of a
//!   problem (the hints of a Sudoku) out of the loss. A destroy operator chooses freely.

use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::destroy::DestroyOperator;
use crate::modelling::{Problem, VariableIndex};
use crate::neural::Batch;

use super::ConsFormerInputs;

/// Plain-data (no tensors, no device) form of `consformer_masks`'s output. Computing this is the
/// only non-trivial CPU work in mask construction, so it's split out from tensor building: it can
/// be computed for many problems in parallel (see `ConsFormerDataset::new`), while the actual
/// `Tensor::from_data` calls -- which touch the device, and so are kept single-threaded to avoid
/// any backend-specific thread/context assumptions (e.g. around CUDA) -- happen afterward.
struct ConsFormerMaskData {
    /// Number of variables of the problem.
    number_vars: usize,
    /// The attention mask, row by row: `[i * number_vars + j]` is true if variable `i` may attend
    /// to variable `j`.
    flat_attention_mask: Vec<bool>,
    /// Whether each variable has more than one value in its domain.
    is_var: Vec<bool>,
}

/// Computes the attention mask and the variable indicator of `problem` as plain data. Variables
/// that share a constraint attend to each other, and every variable attends to itself.
fn consformer_mask_data(problem: &Problem) -> ConsFormerMaskData {
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
            for v in scope.iter().skip(i + 1).copied() {
                flat_attention_mask[u * n + *v] = true;
                flat_attention_mask[*v * n + u] = true;
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
    /// Builds the tensors of the masks on `device`: the `(number_vars, number_vars)` attention
    /// mask and the `(number_vars,)` indicator of the variables with a choice.
    fn into_tensors<B: Backend>(
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
pub(crate) fn consformer_masks<B: Backend>(
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
pub(crate) struct ConsFormerSample<B: Backend> {
    /// The problem.
    problem: Arc<Problem>,
    /// (number_vars, number_vars): true where a variable may attend to another.
    attention_mask: Tensor<B, 2, Bool>,
    /// (number_vars): true for the variables with more than one value in their domain.
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
pub(crate) struct ConsFormerDataset<B: Backend> {
    /// One sample per problem, in the order the problems were given.
    samples: Vec<ConsFormerSample<B>>,
}

impl<B: Backend> ConsFormerDataset<B> {
    /// Builds the dataset of `problems`, computing their masks on `device`.
    pub(crate) fn new(problems: Vec<Arc<Problem>>, device: &B::Device) -> Self {
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

/// ConsFormer reads a batch as rectangular tensors and builds its positional embedding once, from
/// the variables of the first problem, so every problem it handles must have the same variables
/// (the same count, in the same order). Their domains and constraints may differ.
///
/// Returns an error when `problems` is empty or its problems do not all have the same number of
/// variables.
pub(crate) fn try_shared_number_variables(problems: &[Arc<Problem>]) -> Result<usize, String> {
    let n = problems
        .first()
        .ok_or("a ConsFormer batch needs at least one problem")?
        .number_variables();
    if problems.iter().any(|p| p.number_variables() != n) {
        return Err(
            "ConsFormer needs every problem of a batch to have the same number of variables"
                .to_string(),
        );
    }
    Ok(n)
}

/// [`try_shared_number_variables`] for the callers that cannot recover from a mismatch.
///
/// # Panics
///
/// If `problems` is empty or its problems do not all have the same number of variables.
pub(super) fn shared_number_variables(problems: &[Arc<Problem>]) -> usize {
    try_shared_number_variables(problems).unwrap_or_else(|e| panic!("{e}"))
}

/// Stacks the attention masks of a batch, draws a random assignment for each problem and chooses
/// the variables to re-predict: those picked by `destroy` if given, else a random `mask_fraction`
/// of the variables with a choice. Returns the attention masks, the variables to re-predict and
/// the assignments, with the problems as rows.
fn stack_masks_and_sample_assignments<B: Backend>(
    attention_mask_tensors: Vec<Tensor<B, 2, Bool>>,
    var_mask_tensors: &[Tensor<B, 1, Bool>],
    problems: &[Arc<Problem>],
    mask_fraction: f64,
    destroy: Option<&dyn DestroyOperator>,
    device: &B::Device,
) -> (Tensor<B, 3, Bool>, Tensor<B, 2, Bool>, Tensor<B, 2, Int>) {
    let attention_masks = Tensor::stack(attention_mask_tensors, 0);

    let n = shared_number_variables(problems);
    let (var_masks_data, init) = match destroy {
        None => {
            let masks = random_var_masks::<B>(var_mask_tensors, mask_fraction);
            (masks, sample_initial_assignments(problems))
        }
        Some(destroy) => {
            let init = sample_initial_assignments(problems);
            (destroyed_var_masks(destroy, problems, &init, n), init)
        }
    };
    let var_masks: Tensor<B, 2, Bool> =
        Tensor::<B, 1, Bool>::from_data(var_masks_data.as_slice(), device)
            .reshape([problems.len(), n]);
    let assignments: Tensor<B, 2, Int> =
        Tensor::<B, 1, Int>::from_data(init.as_slice(), device).reshape([problems.len(), n]);

    (attention_masks, var_masks, assignments)
}

/// Picks at random, among the variables with a choice of each problem (given by
/// `var_mask_tensors`), those to re-predict, each with probability `mask_fraction`. Returns the
/// choice of all problems, one after the other.
fn random_var_masks<B: Backend>(
    var_mask_tensors: &[Tensor<B, 1, Bool>],
    mask_fraction: f64,
) -> Vec<bool> {
    let mut var_masks_data: Vec<bool> = Vec::new();
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
    var_masks_data
}

/// Draws a random assignment for each problem, all concatenated.
fn sample_initial_assignments(problems: &[Arc<Problem>]) -> Vec<i64> {
    problems
        .iter()
        .flat_map(|problem| {
            crate::sampling::random_assignment(problem)
                .into_iter()
                .map(|value| value as i64)
        })
        .collect()
}

/// Asks `destroy` which variables of each problem to re-predict, given the assignments `init` of
/// all problems (`n` variables each, concatenated). The problems are handled in parallel, each
/// with its own generator, so the result does not depend on the thread schedule.
fn destroyed_var_masks(
    destroy: &dyn DestroyOperator,
    problems: &[Arc<Problem>],
    init: &[i64],
    n: usize,
) -> Vec<bool> {
    let seeds: Vec<u64> =
        crate::utils::with_rng(|rng| (0..problems.len()).map(|_| rng.random::<u64>()).collect());
    let selected: Vec<Vec<usize>> = crate::utils::worker_pool().install(|| {
        problems
            .par_iter()
            .zip(seeds.par_iter())
            .enumerate()
            .map(|(i, (problem, &seed))| {
                let row: Vec<isize> = init[i * n..(i + 1) * n]
                    .iter()
                    .map(|&v| v as isize)
                    .collect();
                let mut rng = StdRng::seed_from_u64(seed);
                destroy.destroy(problem, &row, &mut rng)
            })
            .collect()
    });

    let mut masks = vec![false; problems.len() * n];
    for (i, vars) in selected.into_iter().enumerate() {
        for var in vars {
            masks[i * n + var] = true;
        }
    }
    masks
}

/// Builds the batches of a [`ConsFormerDataset`].
#[derive(Clone)]
pub(crate) struct ConsFormerBatcher {
    /// Fraction of the variables with a choice that are re-predicted, when there is no destroy
    /// operator.
    pub mask_fraction: f64,
    /// Chooses the variables to re-predict, as local search does. If absent, they are drawn at
    /// random with `mask_fraction`.
    pub destroy: Option<Arc<dyn DestroyOperator>>,
}

/// A batch is just a collection of samples with the addition of initial assignments that are
/// passed to the neural network.
#[derive(Clone)]
pub(crate) struct ConsFormerBatch<B: Backend> {
    /// (batch_size, number_vars, number_vars)
    pub attention_masks: Tensor<B, 3, Bool>,
    /// (batch_size, number_vars)
    pub var_masks: Tensor<B, 2, Bool>,
    /// Problems, used to compute the loss
    pub problems: Vec<Arc<Problem>>,
    /// (batch_size, number_vars): the current value of each variable.
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
    ///
    /// This operation is cheap because we clone Burn's tensor which only involve copying a
    /// reference. Every sample has already its own tensor representations so we don't create any
    /// tensor except when stacking.
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
            self.destroy.as_deref(),
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

impl<B: Backend> ConsFormerInputs<B> for ConsFormerBatch<B> {
    fn attention_masks(&self) -> Tensor<B, 3, Bool> {
        self.attention_masks.clone()
    }
}

impl<B: Backend> Batch<B> for ConsFormerBatch<B> {
    fn problems(&self) -> &[Arc<Problem>] {
        &self.problems
    }

    fn assignments(&self) -> Tensor<B, 2, Int> {
        self.assignments.clone()
    }

    fn var_masks(&self) -> Tensor<B, 2, Bool> {
        self.var_masks.clone()
    }

    fn check_problems(problems: &[Arc<Problem>]) -> Result<(), String> {
        try_shared_number_variables(problems).map(|_| ())
    }

    /// Mirrors how local search turns logits into an assignment: only the
    /// variables this sample actually masked (destroyed) take the network's argmax. Without
    /// this, a raw `logits.argmax(2)` asks the network to also reproduce every kept or given
    /// value (every Sudoku hint, for instance), which it has no training signal for
    /// (`ConsFormerLoss` and `DecomposedMddLoss` only score masked positions), so the
    /// satisfaction rate would stay near 0% even once training works.
    fn decode(&self, logits: Tensor<B, 3>) -> Vec<Vec<isize>> {
        let proposed: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);
        let assignment: Tensor<B, 2, Int> = self
            .assignments
            .clone()
            .mask_where(self.var_masks.clone(), proposed);
        let row_width = assignment.dims()[1];
        let values: Vec<isize> = assignment
            .into_data()
            .to_vec::<B::IntElem>()
            .expect("decoded assignment should be an integer tensor")
            .into_iter()
            .map(|v| v.elem::<i64>() as isize)
            .collect();
        values
            .chunks(row_width)
            .zip(&self.problems)
            .map(|(row, problem)| row[..problem.number_variables()].to_vec())
            .collect()
    }

    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Self {
        shared_number_variables(problems);
        let attention_masks = Tensor::cat(
            problems
                .iter()
                .map(|problem| {
                    let (attention_mask, _) = consformer_masks::<B>(problem, device);
                    attention_mask.unsqueeze::<3>()
                })
                .collect(),
            0,
        );

        ConsFormerBatch {
            assignments,
            attention_masks,
            var_masks: destroy_mask.equal_elem(1),
            problems: problems.to_vec(),
        }
    }
}

#[cfg(test)]
mod destroy_tests {
    use super::*;
    use crate::constraints::not_equals;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    struct OnlyFirst;

    impl DestroyOperator for OnlyFirst {
        fn destroy(&self, _: &Problem, _: &[isize], _: &mut StdRng) -> Vec<usize> {
            vec![0]
        }
    }

    fn batch_with(destroy: Option<Arc<dyn DestroyOperator>>) -> ConsFormerBatch<NdArray> {
        let device = NdArrayDevice::default();
        let problems: Vec<Arc<Problem>> = (0..4)
            .map(|_| {
                let mut problem = Problem::default();
                let vars: Vec<_> = (0..3)
                    .map(|_| problem.add_variable(vec![0, 1, 2]))
                    .collect();
                not_equals(&mut problem, vars[0], vars[1]);
                not_equals(&mut problem, vars[1], vars[2]);
                Arc::new(problem)
            })
            .collect();
        let dataset = ConsFormerDataset::<NdArray>::new(problems, &device);
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();
        ConsFormerBatcher {
            mask_fraction: 1.0,
            destroy,
        }
        .batch(samples, &device)
    }

    #[test]
    fn a_training_destroy_operator_decides_which_variables_are_masked() {
        let batch = batch_with(Some(Arc::new(OnlyFirst)));
        let masks: Vec<bool> = batch.var_masks.into_data().to_vec().unwrap();
        assert_eq!(masks, [true, false, false].repeat(4),);
    }

    #[test]
    fn without_a_destroy_operator_the_mask_fraction_decides() {
        let batch = batch_with(None);
        let masks: Vec<bool> = batch.var_masks.into_data().to_vec().unwrap();
        assert!(masks.iter().all(|&m| m));
    }

    #[test]
    fn decode_takes_the_network_choice_only_where_variables_are_masked() {
        let device = NdArrayDevice::default();
        let batch = batch_with(Some(Arc::new(OnlyFirst)));
        // Every variable's logits favour value 2.
        let logits = Tensor::<NdArray, 3>::from_data(
            burn::tensor::TensorData::new([0.0f32, 0.0, 1.0].repeat(12), [4, 3, 3]),
            &device,
        );
        let current: Vec<i64> = batch
            .assignments()
            .into_data()
            .to_vec()
            .expect("integer assignments");
        let rows = batch.decode(logits);
        assert_eq!(rows.len(), 4);
        for (i, row) in rows.iter().enumerate() {
            // Only the first variable is masked.
            assert_eq!(row[0], 2);
            assert_eq!(
                row[1..],
                [current[i * 3 + 1] as isize, current[i * 3 + 2] as isize]
            );
        }
    }
}
