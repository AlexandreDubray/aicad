//! Ready-made training runs of ConsFormer: a loss, a dataset and a checkpoint directory.
//!
//! [`train_consformer`] trains against the classical penalty of each constraint type and
//! [`train_consformer_mdd`] against the exact model count of each constraint's MDD. Both split the
//! problems into a training and a validation set, save the network configuration as
//! `config.json` in the checkpoint directory and train with [`train_model`], which saves the
//! weights next to it. The `_on` variants run on a backend of the caller's choice; the others take
//! the best device found (CUDA, else the CPU).

use std::path::Path;
use std::sync::Arc;

use burn::config::Config;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;

use crate::destroy::DestroyOperator;
use crate::learning::consformer::{ConsFormerLoss, DecomposedMddLoss};
use crate::learning::{TrainingConfig, train_model};
use crate::mdd::MddCompilationConfig;
use crate::modelling::Problem;
use crate::neural::consformer::{
    ConsFormerBatch, ConsFormerBatcher, ConsFormerConfig, ConsFormerDataset, ConsFormerSample,
};
use crate::neural::dispatch_backend;
use crate::utils::with_rng;

/// `(all, train, validation)`, as returned by [`split_train_validation`].
type ProblemSplit = (Vec<Arc<Problem>>, Vec<Arc<Problem>>, Vec<Arc<Problem>>);

/// Splits `problems` into an `(all, train, validation)` triple: `all` is an unshuffled copy (the
/// full set that `train_model` hands to the network and the loss to prepare); with a `validation`
/// set given, `train` is `problems`, otherwise `train` and `validation` are a random 80/20 split of
/// a shuffled copy. Shared by the recipes so that the split cannot drift between them.
fn split_train_validation(
    mut problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
) -> ProblemSplit {
    if let Some(validation_problems) = validation {
        let mut all_problems = problems.clone();
        all_problems.extend(validation_problems.iter().cloned());
        return (all_problems, problems, validation_problems);
    }
    let all_problems = problems.clone();
    let training_size = (problems.len() as f64 * 0.8).round() as usize;
    with_rng(|rng| problems.shuffle(rng));
    let validation_problems = problems.split_off(training_size);
    (all_problems, problems, validation_problems)
}

/// Creates `checkpoint_dir` if missing and saves `network_config` as `config.json` in it, so the
/// network can be rebuilt when loaded.
fn prepare_checkpoint_dir(
    checkpoint_dir: &Path,
    network_config: &ConsFormerConfig,
) -> Result<(), String> {
    std::fs::create_dir_all(checkpoint_dir).map_err(|e| {
        format!(
            "failed to create the checkpoint directory {}: {e}",
            checkpoint_dir.display()
        )
    })?;
    network_config
        .save(checkpoint_dir.join("config.json"))
        .map_err(|e| format!("failed to save the network configuration: {e}"))
}

/// Trains ConsFormer on `problems` with the classical loss, on the best device found. See
/// [`train_consformer_on`].
pub fn train_consformer(
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    destroy: Option<Arc<dyn DestroyOperator>>,
    checkpoint_dir: &Path,
) -> Result<(), String> {
    dispatch_backend!(
        training,
        train_consformer_on,
        problems,
        validation,
        network_config,
        training_config,
        destroy,
        checkpoint_dir,
    )
}

/// Trains ConsFormer on `problems` with the classical loss, a hand-written penalty per constraint
/// type, on the backend `B`.
///
/// Without a `validation` set, 20% of the problems are held out for it. `destroy` picks the
/// variables to re-predict in a training sample, a random subset of `mask_fraction` of them when
/// `None`. The configuration and the best network by validation loss are saved in
/// `checkpoint_dir` (see [`train_model`]).
pub fn train_consformer_on<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    destroy: Option<Arc<dyn DestroyOperator>>,
    checkpoint_dir: &Path,
) -> Result<(), String> {
    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    let (all_problems, problems, validation_problems) =
        split_train_validation(problems, validation);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
        destroy,
    };

    train_model::<
        B,
        ConsFormerConfig,
        ConsFormerSample<B>,
        ConsFormerBatch<B>,
        ConsFormerBatcher,
        ConsFormerLoss,
        ConsFormerSample<B::InnerBackend>,
        ConsFormerBatch<B::InnerBackend>,
    >(
        network_config,
        &all_problems,
        train_dataset,
        validation_dataset,
        batcher,
        ConsFormerLoss,
        training_config,
        checkpoint_dir,
        &device,
    )
    .map(|_| ())
}

/// Trains ConsFormer against the exact weighted model count of the MDD of each constraint, on the
/// best device found. See [`train_consformer_mdd_on`].
pub fn train_consformer_mdd(
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    compilation: MddCompilationConfig,
    bp_rounds: usize,
    checkpoint_dir: &Path,
) -> Result<(), String> {
    dispatch_backend!(
        training,
        train_consformer_mdd_on,
        problems,
        validation,
        network_config,
        training_config,
        compilation,
        bp_rounds,
        checkpoint_dir,
    )
}

/// Trains ConsFormer against the exact weighted model count of the MDD of each constraint (see
/// [`DecomposedMddLoss`]), on the backend `B`. Otherwise like [`train_consformer_on`]: same network,
/// same split, same checkpoint. Every variable is masked, so there is no destroy operator.
/// `compilation` says how the MDDs are compiled, once, for all the problems. `bp_rounds` is the
/// number of rounds of belief propagation that label the diagrams before the loss is computed:
/// with zero the diagrams are labelled with the probabilities of the network themselves, with more
/// the loss also sees what the other constraints say about each variable (see
/// [`DecomposedMddLoss`]).
#[allow(clippy::too_many_arguments)]
pub fn train_consformer_mdd_on<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    compilation: MddCompilationConfig,
    bp_rounds: usize,
    checkpoint_dir: &Path,
) -> Result<(), String> {
    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    let (all_problems, problems, validation_problems) =
        split_train_validation(problems, validation);
    let loss =
        DecomposedMddLoss::new(compilation, network_config.domain_size).with_bp_rounds(bp_rounds);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
        destroy: None,
    };

    train_model::<
        B,
        ConsFormerConfig,
        ConsFormerSample<B>,
        ConsFormerBatch<B>,
        ConsFormerBatcher,
        DecomposedMddLoss,
        ConsFormerSample<B::InnerBackend>,
        ConsFormerBatch<B::InnerBackend>,
    >(
        network_config,
        &all_problems,
        train_dataset,
        validation_dataset,
        batcher,
        loss,
        training_config,
        checkpoint_dir,
        &device,
    )
    .map(|_| ())
}
