pub(crate) mod architecture;
pub(crate) mod dataset;

use super::*;
use crate::modelling::Problem;
pub(crate) use architecture::ConsFormer;
use architecture::*;
pub(crate) use dataset::{ConsFormerBatch, ConsFormerBatcher, ConsFormerDataset, ConsFormerSample};

use burn::config::Config;
use burn::module::Param;
use burn::nn::LinearConfig;
use burn::tensor::Bool;

use std::sync::Arc;

/// A [`Batch`] that `ConsFormer` can read: on top of the assignments and the variables to
/// re-predict, the network needs the attention mask of each problem.
pub(crate) trait ConsFormerInputs<B: Backend>: Batch<B> {
    /// (batch_size, number_vars, number_vars)
    fn attention_masks(&self) -> Tensor<B, 3, Bool>;
}

#[derive(Config, Debug)]
pub(crate) struct ConsFormerConfig {
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
    /// How the multi-head attention blocks apply the primal-graph attention mask.
    #[config(default = "MaskingKind::Hard")]
    pub masking_kind: MaskingKind,
}

impl<B: Backend> NetworkConfig<B> for ConsFormerConfig {
    type N = ConsFormer<B>;

    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N {
        // Note: in the ConsFormer architecture it is assumed that hidden_size == embedding_size.
        //       It is unclear to me that we need this and how it affects learning. Especially
        //       since it seems to not be the norm when using transformers.
        if self.hidden_size != self.embedding_size {
            panic!(
                "ConsFormer architecture initialised with hidden size different than embedding size ({} != {}). Currently, it is not supported.",
                self.hidden_size, self.embedding_size
            );
        }

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
                .with_masking_kind(self.masking_kind)
                .init(device)
            })
            .collect();

        let position_embedding = if self.positional_encoding_dimensions == 0 {
            None
        } else {
            // Every problem has the same variables (see `dataset::shared_number_variables`), so the
            // positions are read from the first one.
            dataset::shared_number_variables(problems);
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
