//! Training of the neural networks of [`crate::neural`].
//!
//! # Meaning
//!
//! [`train::train_model`] fits a network on problems with a [`Loss`], an optimiser
//! ([`optimiser`]) and a monitor of how often the proposed assignments satisfy the problems
//! ([`monitoring`]). [`consformer`] holds the losses of ConsFormer.

pub(crate) mod consformer;
pub(crate) mod monitoring;
pub(crate) mod optimiser;
pub(crate) mod train;

use std::sync::Arc;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::modelling::Problem;
use crate::neural::Batch;

/// The loss a network is trained with.
pub(crate) trait Loss<B: Backend, Ba: Batch<B>> {
    /// Prepares whatever the loss needs about `problems`, once, before training starts. It is
    /// given every problem the loss will score, training and validation alike. Does nothing by
    /// default.
    fn prepare(&mut self, _problems: &[Arc<Problem>]) {}

    fn loss(&self, logits: Tensor<B, 3>, batch: &Ba) -> Tensor<B, 1>;
}
