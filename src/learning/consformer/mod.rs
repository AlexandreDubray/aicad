pub mod architecture;
pub mod dataset;
pub mod loss;

pub use architecture::ConsFormer;
pub use dataset::{ConsFormerBatch, ConsFormerBatcher, ConsFormerDataset, ConsFormerSample};
pub use loss::{ConsFormerLoss, ConstraintLoss};

use burn::config::Config;
use burn::module::Param;
use burn::nn::LinearConfig;

use super::*;

use architecture::*;

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
    /// Number of variables in the problem
    pub num_vars: usize,
}

impl<B: Backend> NetworkConfig<B> for ConsFormerConfig {
    type N = ConsFormer<B>;

    fn init(&self, device: &B::Device) -> Self::N {
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
            position_embedding: FixedPositionalEmbedding::new(
                self.embedding_size,
                self.num_vars,
                device,
            ),
        }
    }
}
