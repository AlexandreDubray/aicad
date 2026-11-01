use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::optim::GradientsParams;
use burn::prelude::ElementConversion;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::learning::Loss;
use crate::learning::monitoring::SatisfactionReport;
use crate::learning::optimiser::OptimiserConfig;
use crate::modelling::Problem;
use crate::neural::{Batch, Network, NetworkConfig};

/// Configuration of the training loop
#[derive(Config, Debug)]
pub(crate) struct TrainingConfig {
    /// Number of epochs
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Batch size
    #[config(default = 512)]
    pub batch_size: usize,
    /// How often (in epochs) to evaluate the loss on the validation set and update the best-so-far
    /// model, the one with the lowest validation loss.
    #[config(default = 10)]
    pub validation_interval: usize,
    /// The optimiser, with its learning rate and hyper-parameters (Adam by default).
    #[config(default = "OptimiserConfig::default()")]
    pub optimiser: OptimiserConfig,
    /// Stop training once the validation loss hasn't improved by more than
    /// `early_stopping_min_delta` for this many consecutive validation checks. `None` (the
    /// default) disables early stopping, matching the previous always-run-`num_epochs` behavior.
    #[config(default = "None")]
    pub early_stopping_patience: Option<usize>,
    /// How much the validation loss must drop below `best_score` at a validation check to
    /// count as progress rather than stagnation. Only used when `early_stopping_patience` is set.
    #[config(default = 1e-4)]
    pub early_stopping_min_delta: f64,
    /// A checkpoint directory (as written to `out_dir` by an earlier run) whose weights initialise
    /// the network before the first epoch, to continue or fine-tune a previous run. `None` (the
    /// default) starts from a fresh random initialisation. The network configuration must be the
    /// one the weights were trained with.
    #[config(default = "None")]
    pub initial_weights: Option<PathBuf>,
}

/// Trains a model. Generic over the backend (B), the network configuration (NC), the training
/// sample type (S), the training batch type (TBatch), the Batcher (Ba), the loss function (L),
/// the validation sample type (SValid) and the validation batch type (VBatch). This is designed
/// so each neural network can be learned with this method against several different training
/// recipes sharing the same architecture -- e.g. the classical per-constraint-penalty ConsFormer
/// loss and the MDD-WMC ConsFormer loss both drive the same `ConsFormer` network, just by
/// instantiating this function with a different `(S, TBatch, L, SValid, VBatch)` tuple.
///
/// network_config: Neural network configuration (type and hyper-parameters)
/// train_dataset: training dataset, used for gradient updates
/// valid_dataset: held-out dataset, used only for model selection -- built on `B::InnerBackend`
///                since evaluating it never needs autodiff bookkeeping
/// batcher: shared by both dataloaders; needs to implement `Batcher` for both `B` and
///          `B::InnerBackend`, which is automatic for any batcher generic over `Backend`
/// loss_fn: The loss function; it is first prepared with `problems` (see `Loss::prepare`)
/// training: Configuration of the training loop
/// device: Device to launch the training on (cpu or gpu) -- also valid as `B::InnerBackend`'s
///         device, since `Autodiff<X>::Device == X::Device`
///
/// Returns the best model found, the one with the lowest validation loss, which is also saved in
/// `out_dir/weights`. When no validation check ran (for instance `num_epochs` is smaller than
/// `validation_interval`), nothing is saved and the network as it is at the end is returned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_model<B, NC, S, TBatch, Ba, L, SValid, VBatch>(
    network_config: NC,
    problems: &[Arc<Problem>],
    train_dataset: impl Dataset<S> + 'static,
    valid_dataset: impl Dataset<SValid> + 'static,
    batcher: Ba,
    mut loss_fn: L,
    training: TrainingConfig,
    out_dir: &Path,
    device: &B::Device,
) -> Result<NC::N, String>
where
    B: AutodiffBackend,
    NC: NetworkConfig<B>,
    NC::N: AutodiffModule<B> + Clone + Network<B, TBatch>,
    <NC::N as AutodiffModule<B>>::InnerModule: Network<B::InnerBackend, VBatch>,
    S: Send + Sync + Clone + std::fmt::Debug + 'static,
    SValid: Send + Sync + Clone + std::fmt::Debug + 'static,
    TBatch: Batch<B>,
    VBatch: Batch<B::InnerBackend>,
    Ba: Batcher<B, S, TBatch>
        + Batcher<B::InnerBackend, SValid, VBatch>
        + Clone
        + Send
        + Sync
        + 'static,
    L: Loss<B, TBatch> + Loss<B::InnerBackend, VBatch>,
{
    if training.num_epochs > 0
        && (training.validation_interval == 0 || training.validation_interval > training.num_epochs)
    {
        panic!(
            "Training over {} epochs and validation interval is {} (should be in [1, {}]",
            training.num_epochs, training.validation_interval, training.num_epochs
        );
    }
    // Let the loss prepare whatever it needs about the problems (e.g. compile their MDDs)
    <L as Loss<B, TBatch>>::prepare(&mut loss_fn, problems);

    // Initialise the network architecture with the given parameters
    let mut network = network_config.init(problems, device);
    if let Some(checkpoint) = &training.initial_weights {
        network = network
            .load_file(checkpoint.join("weights"), &CompactRecorder::new(), device)
            .map_err(|e| {
                format!(
                    "cannot load the initial weights from {}: {e} (expected a checkpoint \
                     directory holding `weights.mpk`, trained with this network configuration)",
                    checkpoint.display()
                )
            })?;
        log::info!("initial weights loaded from {}", checkpoint.display());
    }

    // Load the training and validation datasets
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training.batch_size)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(training.batch_size)
        .build(valid_dataset);

    let mut optim = training.optimiser.init::<B, NC::N>();

    let mut best_score = f64::INFINITY;
    // Whether `out_dir/weights` holds a best-so-far model (no validation may ever have run)
    let mut best_saved = false;
    let mut stagnant_checks = 0usize;

    for epoch in 0..training.num_epochs {
        let mut epoch_loss_sum = 0.0;
        let mut epoch_batches = 0usize;
        let mut epoch_report: Option<SatisfactionReport> = None;
        let epoch_start = Instant::now();

        // Some runtime statistics for logging. The statistics are over the batches; it is the sum
        // of each batch runtime, so it can be used as a comparison if the batch sizes are equal.
        // Total time passed in forward pass of the neural network.
        let mut t_forward = 0.0f64;
        let mut t_loss = 0.0f64;
        let mut t_backward = 0.0f64;

        let mut train_iter = train_dataloader.iter();
        loop {
            let next = train_iter.next();
            let Some(batch) = next else { break };

            let t = Instant::now();
            let logits = network.forward(&batch);
            B::sync(device).ok();
            t_forward += t.elapsed().as_secs_f64();

            let batch_report =
                SatisfactionReport::build(batch.problems(), &batch.decode(logits.clone()));
            match &mut epoch_report {
                Some(report) => report.merge(batch_report),
                None => epoch_report = Some(batch_report),
            }
            B::sync(device).ok();

            let t = Instant::now();
            let loss = loss_fn.loss(logits.clone(), &batch);
            B::sync(device).ok();
            t_loss += t.elapsed().as_secs_f64();
            let loss_scalar = loss.clone().into_scalar().elem::<f32>();
            if !loss_scalar.is_finite() {
                let logits_data: Vec<f32> = logits
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_else(|_| Vec::new());
                let nan_count = logits_data.iter().filter(|v| v.is_nan()).count();
                let inf_count = logits_data.iter().filter(|v| v.is_infinite()).count();
                let (finite_min, finite_max) = logits_data
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                        (lo.min(v), hi.max(v))
                    });
                panic!(
                    "epoch {epoch}: loss went non-finite ({loss_scalar}) -- stopping before this \
                     batch's optimizer step so the network/optimizer state isn't poisoned any \
                     further. logits: {nan_count} NaN, {inf_count} Inf out of {} values, finite \
                     range [{finite_min}, {finite_max}]. If validation checkpointing is enabled, \
                     the file saved at the best score before this epoch is your recovery point.",
                    logits_data.len(),
                );
            }

            let t = Instant::now();
            let grads = GradientsParams::from_grads(loss.backward(), &network);
            network = optim.step(network, grads);
            B::sync(device).ok();
            t_backward += t.elapsed().as_secs_f64();

            epoch_loss_sum += loss_scalar;
            epoch_batches += 1;
        }
        let epoch_rt = epoch_start.elapsed().as_secs();
        log::info!(
            "epoch {epoch}: forward={t_forward:.3}s loss={t_loss:.3}s backward+step={t_backward:.3}s"
        );
        let epoch_loss = epoch_loss_sum / epoch_batches as f32;
        if !epoch_loss.is_finite() {
            log::warn!("Non-finite train loss at epoch {epoch}");
        }
        log::info!("epoch {epoch}: loss = {epoch_loss} ({epoch_rt} seconds)");
        crate::diagnostics::data_log!(
            "train_epoch",
            epoch = epoch,
            t_forward = t_forward,
            t_loss = t_loss,
            t_backward = t_backward,
            epoch_rt_s = epoch_start.elapsed().as_secs_f64(),
            epoch_loss = epoch_loss,
        );
        if let Some(report) = epoch_report {
            report.print(40);
        }

        if (epoch + 1) % training.validation_interval == 0 {
            // Pass the network in validation mode
            let valid_network = network.valid();
            let mut valid_loss_sum = 0.0f64;
            let mut valid_batches = 0usize;

            let mut valid_iter = valid_dataloader.iter();
            loop {
                let next = valid_iter.next();
                let Some(batch) = next else { break };

                let logits = valid_network.forward(&batch);
                B::InnerBackend::sync(device).ok();

                let loss = loss_fn.loss(logits, &batch);
                valid_loss_sum += loss.into_scalar().elem::<f32>() as f64;
                B::InnerBackend::sync(device).ok();
                valid_batches += 1;
            }

            let score = valid_loss_sum / valid_batches as f64;
            if !score.is_finite() {
                log::warn!("Non-finite validation score at epoch {epoch}");
            }
            log::info!("epoch {epoch}: validation loss = {score:.4}");
            crate::diagnostics::data_log!("valid_epoch", epoch = epoch, score = score,);

            let improved = score < best_score - training.early_stopping_min_delta;

            if score < best_score {
                best_score = score;
                match network
                    .clone()
                    .save_file(out_dir.join("weights"), &CompactRecorder::new())
                {
                    Ok(()) => best_saved = true,
                    Err(e) => {
                        log::warn!("warning: failed to save checkpoint at epoch {epoch}: {e}")
                    }
                }
            }

            if let Some(patience) = training.early_stopping_patience {
                if improved {
                    stagnant_checks = 0;
                } else {
                    stagnant_checks += 1;
                    log::info!(
                        "epoch {epoch}: no improvement > {} for {stagnant_checks}/{patience} \
                         validation checks",
                        training.early_stopping_min_delta,
                    );
                    if stagnant_checks >= patience {
                        log::info!(
                            "epoch {epoch}: early stopping -- {stagnant_checks} validation \
                             checks without improvement > {}",
                            training.early_stopping_min_delta,
                        );
                        break;
                    }
                }
            }
        }

        log::info!("");
    }

    // Return the best model rather than the last one. Reading it back from the checkpoint avoids
    // keeping a second copy of the network in memory during training.
    if best_saved {
        network = network
            .load_file(out_dir.join("weights"), &CompactRecorder::new(), device)
            .map_err(|e| {
                format!(
                    "cannot reload the best model from {}: {e}",
                    out_dir.display()
                )
            })?;
    }
    Ok(network)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    use crate::constraints::not_equals;
    use crate::learning::consformer::ConsFormerLoss;
    use crate::modelling::Problem;
    use crate::neural::consformer::{ConsFormerBatcher, ConsFormerConfig, ConsFormerDataset};
    use crate::neural::load_network;

    #[test]
    fn early_stopping_stops_before_num_epochs_once_the_score_stops_improving() {
        type ADBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let out_dir =
            std::env::temp_dir().join(format!("aicad_test_early_stopping_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out_dir);

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problems = vec![Arc::new(problem)];

        let network_config = ConsFormerConfig::new(3, 4, 4, 1, 4, 0, 0.5, 1.0).with_num_layers(1);

        std::fs::create_dir_all(&out_dir).unwrap();
        network_config.save(out_dir.join("config.json")).unwrap();

        let train_dataset = ConsFormerDataset::<ADBackend>::new(problems.clone(), &device);
        let valid_dataset = ConsFormerDataset::<NdArray>::new(problems.clone(), &device);
        let batcher = ConsFormerBatcher {
            mask_fraction: 0.5,
            destroy: None,
        };

        // `early_stopping_min_delta` set unreasonably high so the very first improvement (from
        // `best_score = INFINITY`) is the only one that will ever count -- every validation check
        // after the first must register as stagnant, so training should stop at
        // epoch `early_stopping_patience + 1`, well short of `num_epochs`.
        let training = TrainingConfig::new()
            .with_num_epochs(10)
            .with_validation_interval(1)
            .with_batch_size(1)
            .with_early_stopping_patience(Some(1))
            .with_early_stopping_min_delta(1000.0);

        train_model::<ADBackend, ConsFormerConfig, _, _, _, ConsFormerLoss, _, _>(
            network_config,
            &problems,
            train_dataset,
            valid_dataset,
            batcher,
            ConsFormerLoss,
            training,
            &out_dir,
            &device,
        )
        .unwrap();

        // The first validation check is the only improvement, so training stops after the second
        // one and the best model is the one saved at the root.
        assert!(out_dir.join("weights.mpk").is_file());
        let _ = load_network::<NdArray, ConsFormerConfig>(&out_dir, &problems, &device)
            .expect("the best checkpoint should be loadable directly");

        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// A small ConsFormer problem set plus its checkpoint directory, with `config.json` written.
    fn setup(name: &str) -> (Vec<Arc<Problem>>, ConsFormerConfig, std::path::PathBuf) {
        let out_dir =
            std::env::temp_dir().join(format!("aicad_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out_dir);
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let network_config = ConsFormerConfig::new(3, 4, 4, 1, 4, 0, 0.5, 1.0).with_num_layers(1);
        std::fs::create_dir_all(&out_dir).unwrap();
        network_config.save(out_dir.join("config.json")).unwrap();
        (vec![Arc::new(problem)], network_config, out_dir)
    }

    fn train(
        problems: &[Arc<Problem>],
        network_config: ConsFormerConfig,
        training: TrainingConfig,
        out_dir: &Path,
    ) -> Result<<ConsFormerConfig as NetworkConfig<Autodiff<NdArray>>>::N, String> {
        type ADBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();
        train_model::<ADBackend, ConsFormerConfig, _, _, _, ConsFormerLoss, _, _>(
            network_config,
            problems,
            ConsFormerDataset::<ADBackend>::new(problems.to_vec(), &device),
            ConsFormerDataset::<NdArray>::new(problems.to_vec(), &device),
            ConsFormerBatcher {
                mask_fraction: 0.5,
                destroy: None,
            },
            ConsFormerLoss,
            training,
            out_dir,
            &device,
        )
    }

    #[test]
    fn initial_weights_initialise_the_network_before_training() {
        use crate::neural::consformer::ConsFormerBatch;
        use burn::tensor::{Int, Tensor};

        let device = NdArrayDevice::default();
        let (problems, network_config, first_dir) = setup("initial_weights_first");
        let training = TrainingConfig::new()
            .with_num_epochs(2)
            .with_validation_interval(1)
            .with_batch_size(1);
        train(&problems, network_config.clone(), training, &first_dir).unwrap();

        // Zero epochs: the returned network is exactly the one the weights initialised.
        let (_, _, second_dir) = setup("initial_weights_second");
        let training = TrainingConfig::new()
            .with_num_epochs(0)
            .with_batch_size(1)
            .with_initial_weights(Some(first_dir.clone()));
        let resumed = train(&problems, network_config, training, &second_dir).unwrap();
        let (_, saved) =
            load_network::<Autodiff<NdArray>, ConsFormerConfig>(&first_dir, &problems, &device)
                .unwrap();

        let batch = ConsFormerBatch::<Autodiff<NdArray>>::for_assignments(
            &problems,
            Tensor::<Autodiff<NdArray>, 2, Int>::from_ints([[0, 1]], &device),
            Tensor::<Autodiff<NdArray>, 2, Int>::from_ints([[1, 1]], &device),
            &device,
        );
        let a = resumed.forward(&batch).into_data();
        let b = saved.forward(&batch).into_data();
        assert_eq!(a, b);

        std::fs::remove_dir_all(&first_dir).ok();
        std::fs::remove_dir_all(&second_dir).ok();
    }

    #[test]
    fn the_returned_network_is_the_best_one_saved_in_the_checkpoint() {
        use crate::neural::consformer::ConsFormerBatch;
        use burn::tensor::{Int, Tensor};

        let device = NdArrayDevice::default();
        let (problems, network_config, out_dir) = setup("returns_best");
        let training = TrainingConfig::new()
            .with_num_epochs(3)
            .with_validation_interval(1)
            .with_batch_size(1);
        let returned = train(&problems, network_config, training, &out_dir).unwrap();
        let (_, saved) =
            load_network::<Autodiff<NdArray>, ConsFormerConfig>(&out_dir, &problems, &device)
                .unwrap();

        let batch = ConsFormerBatch::<Autodiff<NdArray>>::for_assignments(
            &problems,
            Tensor::<Autodiff<NdArray>, 2, Int>::from_ints([[0, 1]], &device),
            Tensor::<Autodiff<NdArray>, 2, Int>::from_ints([[1, 1]], &device),
            &device,
        );
        assert_eq!(
            returned.forward(&batch).into_data(),
            saved.forward(&batch).into_data()
        );
        std::fs::remove_dir_all(&out_dir).ok();
    }

    #[test]
    fn missing_initial_weights_are_reported_with_their_path() {
        let (problems, network_config, out_dir) = setup("missing_initial_weights");
        let training = TrainingConfig::new()
            .with_num_epochs(10)
            .with_initial_weights(Some(out_dir.join("nowhere")));
        let error = train(&problems, network_config, training, &out_dir).unwrap_err();
        assert!(error.contains("nowhere"), "{error}");
        std::fs::remove_dir_all(&out_dir).ok();
    }

    #[test]
    #[should_panic(expected = "validation interval is 0")]
    fn panic_on_zero_validation_interval() {
        let (problems, network_config, out_dir) = setup("tmp");
        let training = TrainingConfig::new()
            .with_num_epochs(3)
            .with_validation_interval(0)
            .with_batch_size(1);
        std::fs::remove_dir_all(&out_dir).ok();
        train(&problems, network_config, training, &out_dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "validation interval is 4")]
    fn panic_on_out_of_bound_validation_interval() {
        let (problems, network_config, out_dir) = setup("tmp");
        let training = TrainingConfig::new()
            .with_num_epochs(3)
            .with_validation_interval(4)
            .with_batch_size(1);
        std::fs::remove_dir_all(&out_dir).ok();
        train(&problems, network_config, training, &out_dir).unwrap();
    }
}
