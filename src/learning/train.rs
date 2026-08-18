use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::learning::monitoring::SatisfactionReport;
use crate::learning::{BatchProblems, Loss, Network, NetworkConfig};
use crate::modelling::Problem;

/// Heuristic to select the best model during training
#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
pub enum ModelSelection {
    /// Use the best training loss
    Loss,
    /// Use the best constraint satisfaction ratio
    ConstraintSatisfaction,
}

/// Configuration of the training loop
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Learning rate
    #[config(default = 3e-4)]
    pub lr: f64,
    /// Number of epochs
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Batch size
    #[config(default = 512)]
    pub batch_size: usize,
    /// How often (in epochs) to evaluate on the validation set and update
    /// the best-so-far model.
    #[config(default = 10)]
    pub validation_interval: usize,
    /// How to evaluate the best model found so far
    pub model_selection: ModelSelection,
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
/// loss_fn: The loss function
/// training: Configuration of the training loop
/// device: Device to launch the training on (cpu or gpu) -- also valid as `B::InnerBackend`'s
///         device, since `Autodiff<X>::Device == X::Device`
pub fn train_model<B, NC, S, TBatch, Ba, L, SValid, VBatch>(
    network_config: NC,
    problems: &[Arc<Problem>],
    train_dataset: impl Dataset<S> + Send + Sync + 'static,
    valid_dataset: impl Dataset<SValid> + Send + Sync + 'static,
    batcher: Ba,
    loss_fn: L,
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
    TBatch: BatchProblems<B> + Clone + Send + Sync + std::fmt::Debug + 'static,
    VBatch: BatchProblems<B::InnerBackend> + Clone + Send + Sync + std::fmt::Debug + 'static,
    Ba: Batcher<B, S, TBatch>
        + Batcher<B::InnerBackend, SValid, VBatch>
        + Clone
        + Send
        + Sync
        + 'static,
    L: Loss<B, TBatch> + Loss<B::InnerBackend, VBatch>,
{
    // Initialise the network architecture with the given parameters
    let mut network = network_config.init(problems, device);

    // Load the training and validation datasets
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training.batch_size)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(training.batch_size)
        .build(valid_dataset);

    let mut optim = AdamConfig::new().init();

    let mut best_score = f64::INFINITY;

    for epoch in 0..training.num_epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_report: Option<SatisfactionReport> = None;
        let epoch_start = Instant::now();

        for batch in train_dataloader.iter() {
            let logits = network.forward(&batch);

            let batch_report = SatisfactionReport::build(logits.clone(), &batch);
            match &mut epoch_report {
                Some(report) => report.merge(batch_report),
                None => epoch_report = Some(batch_report),
            }

            let loss = loss_fn.loss(logits.clone(), &batch);
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

            let grads = GradientsParams::from_grads(loss.backward(), &network);
            network = optim.step(training.lr, network, grads);

            epoch_loss += loss_scalar;
        }
        let epoch_rt = epoch_start.elapsed().as_secs();
        log::info!("epoch {epoch}: loss = {epoch_loss} ({epoch_rt} seconds)");
        if let Some(report) = epoch_report {
            report.print(40);
        }

        if (epoch + 1) % training.validation_interval == 0 {
            // Pass the network in validation mode
            let valid_network = network.valid();
            let mut valid_report: Option<SatisfactionReport> = None;
            let mut valid_loss_sum = 0.0f64;
            let mut valid_batches = 0usize;

            for batch in valid_dataloader.iter() {
                let logits = valid_network.forward(&batch);

                match training.model_selection {
                    ModelSelection::Loss => {
                        let loss = loss_fn.loss(logits, &batch);
                        valid_loss_sum += loss.into_scalar().elem::<f32>() as f64;
                    }
                    ModelSelection::ConstraintSatisfaction => {
                        let batch_report = SatisfactionReport::build(logits.clone(), &batch);
                        match &mut valid_report {
                            Some(report) => report.merge(batch_report),
                            None => valid_report = Some(batch_report),
                        }
                    }
                }
                valid_batches += 1;
            }

            let score = match training.model_selection {
                ModelSelection::Loss => {
                    let avg_valid_loss = valid_loss_sum / valid_batches as f64;
                    log::info!("epoch {epoch}: validation loss = {avg_valid_loss:.4}");
                    avg_valid_loss
                }
                ModelSelection::ConstraintSatisfaction => {
                    panic!("Constraint satisfaction for model selection is not implemented");
                }
            };

            if score < best_score {
                best_score = score;
                if let Err(e) = network
                    .clone()
                    .save_file(out_dir.join("weights"), &CompactRecorder::new())
                {
                    log::warn!("warning: failed to save checkpoint at epoch {epoch}: {e}");
                }
            }
        }

        log::info!("");
    }
    network
}
