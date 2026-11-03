//! Training of the neural networks of [`crate::neural`].
//!
//! # Meaning
//!
//! [`train_model`] fits a network on problems with a [`Loss`], an optimiser
//! ([`OptimiserConfig`]) and a monitor of how often the proposed assignments satisfy the problems
//! ([`SatisfactionReport`]). [`consformer`] holds the losses of ConsFormer.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod consformer;
mod monitoring;
mod optimiser;
mod train;

pub use monitoring::SatisfactionReport;
pub use optimiser::{Optimiser, OptimiserConfig};
pub use train::{TrainingConfig, train_model};

use std::sync::Arc;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::modelling::Problem;
use crate::neural::Batch;

/// The loss a network is trained with.
pub trait Loss<B: Backend, Ba: Batch<B>> {
    /// Prepares whatever the loss needs about `problems`, once, before training starts. It is
    /// given every problem the loss will score, training and validation alike. Does nothing by
    /// default.
    fn prepare(&mut self, _problems: &[Arc<Problem>]) {}

    /// The loss of `batch` for the network output `logits`, a scalar.
    fn loss(&self, logits: Tensor<B, 3>, batch: &Ba) -> Tensor<B, 1>;
}
