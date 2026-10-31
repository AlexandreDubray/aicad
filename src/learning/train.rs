use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::optim::GradientsParams;
use burn::prelude::ElementConversion;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::learning::monitoring::SatisfactionReport;
use crate::learning::optimiser::OptimiserConfig;
use crate::learning::{Batch, Loss, Network, NetworkConfig};
use crate::modelling::Problem;

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
    /// If true, also save the best-so-far model at a decaying-density set of training-epoch
    /// horizons (see `compute_horizons`), so a later sweep can evaluate performance as a function
    /// of training budget without needing a separate run per budget. Each horizon is its own
    /// self-contained `out_dir/epoch_<N>/{config.json, weights}` pair, loadable the same way as
    /// the root checkpoint (`config.json` is just copied from `out_dir`, since it's the same for
    /// every horizon of a run). Off by default: purely additive to the existing root checkpoint,
    /// but costs one extra checkpoint directory per horizon on disk.
    #[config(default = false)]
    pub save_horizons: bool,
    /// How many horizon checkpoints to save across `[0, num_epochs]` when `save_horizons` is set.
    /// Spaced by geometric growth (dense early, sparse late) rather than evenly, since early
    /// training changes the model much more per epoch than late training does.
    #[config(default = 15)]
    pub num_checkpoints: usize,
    /// Stop training once the validation loss hasn't improved by more than
    /// `early_stopping_min_delta` for this many consecutive validation checks. `None` (the
    /// default) disables early stopping, matching the previous always-run-`num_epochs` behavior.
    #[config(default = "None")]
    pub early_stopping_patience: Option<usize>,
    /// How much the validation loss must drop below `best_score` at a validation check to
    /// count as progress rather than stagnation. Only used when `early_stopping_patience` is set.
    #[config(default = 1e-4)]
    pub early_stopping_min_delta: f64,
}

/// Generates a decaying-density set of training-epoch checkpoints: `h_0 = validation_interval`,
/// `h_{i+1} = h_i * r`, snapped to the nearest validation event (checkpoints can only happen at
/// validation events) and deduplicated, always ending at the final validation event on or before
/// `num_epochs`. `r` is solved from `num_checkpoints` so the schedule always spans the full run
/// regardless of `num_epochs`/`validation_interval`. Front-loaded by construction: consecutive
/// horizons are a constant multiplicative step apart, so absolute spacing grows over the run
/// (e.g. num_epochs=5000, validation_interval=10, num_checkpoints=15 gives roughly
/// `[10, 20, 40, 60, 90, 140, 220, 350, 540, 850, 1320, 2060, 3210, 5000]`).
fn compute_horizons(
    num_epochs: usize,
    validation_interval: usize,
    num_checkpoints: usize,
) -> Vec<usize> {
    if validation_interval == 0 || num_checkpoints == 0 {
        return Vec::new();
    }
    let final_epoch = (num_epochs / validation_interval) * validation_interval;
    if final_epoch == 0 {
        return Vec::new();
    }
    if num_checkpoints == 1 {
        return vec![final_epoch];
    }

    let h0 = validation_interval as f64;
    let target = final_epoch as f64;
    if target <= h0 {
        return vec![final_epoch];
    }

    let r = (target / h0).powf(1.0 / (num_checkpoints as f64 - 1.0));
    let mut horizons = Vec::new();
    let mut h = h0;
    while h < target {
        let snapped = ((h / validation_interval as f64).round() as usize) * validation_interval;
        if snapped >= validation_interval && horizons.last().copied() != Some(snapped) {
            horizons.push(snapped);
        }
        h *= r;
    }
    if horizons.last().copied() != Some(final_epoch) {
        horizons.push(final_epoch);
    }
    horizons
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
) -> NC::N
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
    // Let the loss prepare whatever it needs about the problems (e.g. compile their MDDs)
    <L as Loss<B, TBatch>>::prepare(&mut loss_fn, problems);

    // Initialise the network architecture with the given parameters
    let mut network = network_config.init(problems, device);

    // Load the training and validation datasets
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training.batch_size)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(training.batch_size)
        .build(valid_dataset);

    let mut optim = training.optimiser.init::<B, NC::N>();

    let mut best_score = f64::INFINITY;
    let mut stagnant_checks = 0usize;

    let horizons: std::collections::HashSet<usize> = if training.save_horizons {
        compute_horizons(
            training.num_epochs,
            training.validation_interval,
            training.num_checkpoints,
        )
        .into_iter()
        .collect()
    } else {
        std::collections::HashSet::new()
    };
    let mut horizon_manifest: Vec<serde_json::Value> = Vec::new();

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

            let batch_report = SatisfactionReport::build(logits.clone(), &batch);
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
            log::info!("epoch {epoch}: validation loss = {score:.4}");
            crate::diagnostics::data_log!("valid_epoch", epoch = epoch, score = score,);

            let improved = score < best_score - training.early_stopping_min_delta;

            if score < best_score {
                best_score = score;
                if let Err(e) = network
                    .clone()
                    .save_file(out_dir.join("weights"), &CompactRecorder::new())
                {
                    log::warn!("warning: failed to save checkpoint at epoch {epoch}: {e}");
                }
                if horizons.contains(&(epoch + 1)) {
                    let epoch_dir = out_dir.join(format!("epoch_{:05}", epoch + 1));
                    let saved = std::fs::create_dir_all(&epoch_dir)
                        .and_then(|_| {
                            std::fs::copy(
                                out_dir.join("config.json"),
                                epoch_dir.join("config.json"),
                            )
                        })
                        .map_err(|e| e.to_string())
                        .and_then(|_| {
                            network
                                .clone()
                                .save_file(epoch_dir.join("weights"), &CompactRecorder::new())
                                .map_err(|e| e.to_string())
                        });
                    match saved {
                        Ok(_) => {
                            horizon_manifest.push(serde_json::json!({
                                "epoch": epoch + 1,
                                "best_score_so_far": best_score,
                                "dir": epoch_dir.file_name().and_then(|n| n.to_str()),
                            }));
                            if let Err(e) = std::fs::write(
                                out_dir.join("horizons_manifest.json"),
                                serde_json::to_string_pretty(&horizon_manifest).unwrap_or_default(),
                            ) {
                                log::warn!("warning: failed to write horizons manifest: {e}");
                            }
                        }
                        Err(e) => log::warn!(
                            "warning: failed to save horizon checkpoint at epoch {epoch}: {e}"
                        ),
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
    network
}

#[cfg(test)]
mod test_compute_horizons {
    use super::compute_horizons;

    #[test]
    fn matches_the_real_nurse_rostering_config() {
        let horizons = compute_horizons(5000, 10, 15);
        assert_eq!(
            horizons,
            vec![
                10, 20, 40, 60, 90, 140, 220, 350, 540, 850, 1320, 2060, 3210, 5000
            ]
        );
    }

    #[test]
    fn is_front_loaded_and_ends_at_the_final_validation_event() {
        let horizons = compute_horizons(1000, 10, 8);
        assert!(
            horizons.windows(2).all(|w| w[0] < w[1]),
            "must be strictly increasing"
        );
        assert_eq!(*horizons.last().unwrap(), 1000);
        // front-loaded: the gaps should grow monotonically (geometric spacing)
        let gaps: Vec<i64> = horizons
            .windows(2)
            .map(|w| w[1] as i64 - w[0] as i64)
            .collect();
        assert!(
            gaps.windows(2).all(|w| w[0] <= w[1]),
            "gaps should be non-decreasing: {gaps:?}"
        );
    }

    #[test]
    fn every_horizon_lands_on_a_validation_event() {
        let horizons = compute_horizons(777, 25, 10);
        assert!(horizons.iter().all(|h| h % 25 == 0));
        assert!(horizons.iter().all(|h| *h <= 777));
    }

    #[test]
    fn num_epochs_smaller_than_validation_interval_yields_no_horizons() {
        assert!(compute_horizons(5, 10, 15).is_empty());
    }

    #[test]
    fn single_checkpoint_is_just_the_final_validation_event() {
        assert_eq!(compute_horizons(100, 10, 1), vec![100]);
    }
}

#[cfg(test)]
mod test_horizon_checkpoint_layout {
    use super::*;

    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    use crate::constraints::not_equals;
    use crate::learning::consformer::{
        ConsFormerBatcher, ConsFormerConfig, ConsFormerDataset, ConsFormerLoss,
    };
    use crate::local_search::load_network;
    use crate::modelling::Problem;

    /// End-to-end: a horizon checkpoint must be its own self-contained, loadable directory --
    /// `out_dir/epoch_<N>/{config.json, weights.mpk}` -- not just a bare weights file that needs
    /// the root's `config.json` to load, and the root checkpoint (`out_dir/{config.json,
    /// weights.mpk}`) must still be the best model directly, with nothing to search for.
    #[test]
    fn horizon_and_root_checkpoints_are_both_directly_loadable() {
        type ADBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let out_dir = std::env::temp_dir().join(format!(
            "aicad_test_horizon_checkpoint_layout_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out_dir);

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problems = vec![Arc::new(problem)];

        let network_config = ConsFormerConfig::new(3, 4, 4, 1, 4, 0, 0.5, 1.0).with_num_layers(1);

        // Mirrors `pyaicad::learn::prepare_checkpoint_dir`, called before `train_model` in every
        // real caller -- `train_model` itself only ever reads `out_dir/config.json` (to copy it
        // into each horizon dir), never writes it.
        std::fs::create_dir_all(&out_dir).unwrap();
        network_config.save(out_dir.join("config.json")).unwrap();

        let train_dataset = ConsFormerDataset::<ADBackend>::new(problems.clone(), &device);
        let valid_dataset = ConsFormerDataset::<NdArray>::new(problems.clone(), &device);
        let batcher = ConsFormerBatcher {
            mask_fraction: 0.5,
            destroy: None,
        };

        let training = TrainingConfig::new()
            .with_num_epochs(2)
            .with_validation_interval(1)
            .with_batch_size(1)
            .with_save_horizons(true)
            .with_num_checkpoints(1);

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
        );

        assert!(out_dir.join("config.json").is_file());
        assert!(out_dir.join("weights.mpk").is_file());

        // `num_checkpoints = 1` means exactly one horizon: the final validation event (epoch 2).
        let epoch_dir = out_dir.join("epoch_00002");
        assert!(epoch_dir.join("config.json").is_file());
        assert!(epoch_dir.join("weights.mpk").is_file());
        assert!(out_dir.join("horizons_manifest.json").is_file());

        // The whole point: a horizon checkpoint loads on its own, the same way the root does --
        // no need to reach back into `out_dir` for its `config.json`.
        let _ = load_network::<NdArray, ConsFormerConfig>(&epoch_dir, &problems, &device)
            .expect("a horizon checkpoint should be loadable directly, on its own");
        let _ = load_network::<NdArray, ConsFormerConfig>(&out_dir, &problems, &device)
            .expect("the root checkpoint should still be loadable directly");

        std::fs::remove_dir_all(&out_dir).ok();
    }

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
            .with_early_stopping_min_delta(1000.0)
            .with_save_horizons(true)
            .with_num_checkpoints(1);

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
        );

        // `num_checkpoints = 1` on a 10-epoch run means the only horizon is epoch 10 -- it must
        // never be reached if early stopping actually cut the run short.
        assert!(!out_dir.join("epoch_00010").exists());

        std::fs::remove_dir_all(&out_dir).ok();
    }
}
