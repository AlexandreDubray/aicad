pub mod architecture;
pub mod dataset;
pub mod loss;
pub mod mdd_dataset;

use super::*;
use crate::modelling::Problem;
pub use architecture::ConsFormer;
use architecture::*;
pub use dataset::{
    consformer_masks, ConsFormerBatch, ConsFormerBatcher, ConsFormerDataset, ConsFormerSample,
};
pub use loss::{ConsFormerLoss, ConsFormerMddLoss, ConstraintLoss};
pub use mdd_dataset::{
    ConsFormerMddBatch, ConsFormerMddBatcher, ConsFormerMddDataset, ConsFormerMddSample,
    MddBucketBatch, MddBucketKey, MddCompilationConfig, MddInstance,
};

use burn::config::Config;
use burn::module::Param;
use burn::nn::LinearConfig;

use std::sync::Arc;

/// Shared inputs every ConsFormer-compatible batch must provide for a
/// forward pass. Kept separate from `BatchProblems`/`Batch` (see
/// `learning::mod`) so `ConsFormer` can be driven by several different batch
/// structs -- e.g. `ConsFormerBatch` for the classical per-constraint-penalty
/// loss, and a future dedicated batch for the MDD-WMC loss -- without
/// merging their data into one shared batch type.
pub trait ConsFormerInputs<B: Backend> {
    /// (batch_size, number_vars, number_vars)
    fn attention_masks(&self) -> Tensor<B, 3, Bool>;
    /// (batch_size, number_vars)
    fn var_masks(&self) -> Tensor<B, 2, Bool>;
    /// (batch_size, number_vars)
    fn assignments(&self) -> Tensor<B, 2, Int>;
}

#[derive(Config, Debug)]
pub struct ConsFormerConfig {
    /// Size of the domains of the variables. Currently assumed fixed for all variables.
    pub domain_size: usize,
    /// Size of the embedding of the assignments
    pub embedding_size: usize,
    /// Size of the hidden layers in the multi-head attention blocks
    pub hidden_size: usize,
    /// Number of heads in the multi-head attention blocks
    pub num_heads: usize,
    /// Size of the internal layers in the feed-forward blocks
    pub expand_size: usize,
    /// Number of transformer blocks
    #[config(default = 1)]
    pub num_layers: usize,
    /// Dropout during training
    #[config(default = 0.0)]
    pub drop_out: f64,
    /// Insert bias in feed-forward block
    #[config(default = true)]
    pub bias: bool,
    pub positional_encoding_dimensions: usize,
    /// Fraction of free variables randomly marked eligible for update on
    /// each training step (see `ConsFormerBatcher::mask_fraction`).
    pub mask_fraction: f64,
    /// Logit scaling factor
    pub tau: f64,
}

impl<B: Backend> NetworkConfig<B> for ConsFormerConfig {
    type N = ConsFormer<B>;

    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N {
        // Constructs the num_layers transformer blocks with the given parameters
        let transformer_blocks = (0..self.num_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.embedding_size,
                    self.hidden_size,
                    self.num_heads,
                    self.expand_size,
                )
                .with_attn_drop(self.drop_out)
                .with_ffn_drop(self.drop_out)
                .with_bias(self.bias)
                .init(device)
            })
            .collect();

        let position_embedding = if self.positional_encoding_dimensions == 0 {
            None
        } else {
            let positions = problems[0]
                .iter_variables()
                .map(|variable| problems[0][variable].position().clone().unwrap())
                .collect();
            Some(StructuredPositionalEmbedding::new(
                self.embedding_size,
                self.positional_encoding_dimensions,
                positions,
                device,
            ))
        };

        ConsFormer {
            mask_embedding: Param::from_tensor(Tensor::random(
                [self.embedding_size],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                device,
            )),
            assignment_embedding: AssignmentEmbeddingConfig::new(
                self.domain_size,
                self.embedding_size,
            )
            .init(device),
            embedding_mixer: EmbeddingMixerConfig::new(self.embedding_size).init(device),
            transformer_blocks,
            head: LinearConfig::new(self.hidden_size, self.domain_size).init(device),
            position_embedding,
            tau: self.tau,
        }
    }
}
