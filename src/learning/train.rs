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

use crate::learning::monitoring::SatisfactionReport;
use crate::learning::{Loss, Network, NetworkConfig};

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
/// sample type (S), the validation sample type (SValid), the Batcher (Ba) and the loss function
/// (L). This is designed so each neural network can be learned with this method. For example, to
/// pass from ConsFormer to ConsFormer-MDD, the only thing that needs to change is the sample
/// type, to include MDDs.
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
pub fn train_model<B, NC, S, Ba, L, SValid>(
    network_config: NC,
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
    NC::N: AutodiffModule<B> + Clone,
    <NC::N as AutodiffModule<B>>::InnerModule: Network<B::InnerBackend>,
    S: Send + Sync + Clone + std::fmt::Debug + 'static,
    SValid: Send + Sync + Clone + std::fmt::Debug + 'static,
    Ba: Batcher<B, S, <NC::N as Network<B>>::Batch>
        + Batcher<
            B::InnerBackend,
            SValid,
            <<NC::N as AutodiffModule<B>>::InnerModule as Network<B::InnerBackend>>::Batch,
        > + Clone
        + Send
        + Sync
        + 'static,
    L: Loss<B, NC::N> + Loss<B::InnerBackend, <NC::N as AutodiffModule<B>>::InnerModule>,
{
    // Initialise the network architecture with the given parameters
    let mut network = network_config.init(device);

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

        for batch in train_dataloader.iter() {
            let logits = network.forward(&batch);

            let batch_report = SatisfactionReport::build(logits.clone(), &batch);
            match &mut epoch_report {
                Some(report) => report.merge(batch_report),
                None => epoch_report = Some(batch_report),
            }

            let loss = loss_fn.loss(logits, &batch);
            let grads = GradientsParams::from_grads(loss.backward(), &network);
            network = optim.step(training.lr, network, grads);

            epoch_loss += loss.into_scalar().elem::<f32>();
        }

        log::info!("epoch {epoch}: loss = {epoch_loss}");
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
