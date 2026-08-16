//! ConsFormer architecture, ported from the official repository
//! https://github.com/khalil-research/ConsFormer

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{
    Dropout, DropoutConfig, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{backend::Backend, Bool, Int, Tensor};

use super::ConsFormerInputs;
use crate::learning::*;

// --- Embedding --- //

/// Embeds a discrete assignment to the variables into the embedding space
#[derive(Module, Debug)]
pub struct AssignmentEmbedding<B: Backend> {
    embedding: burn::nn::Embedding<B>,
}

/// Parameters of the embedding module
#[derive(Config, Debug)]
pub struct AssignmentEmbeddingConfig {
    /// Domain size of the variables
    pub domain_size: usize,
    /// Embedding size
    pub embedding_size: usize,
}

impl AssignmentEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AssignmentEmbedding<B> {
        AssignmentEmbedding {
            embedding: EmbeddingConfig::new(self.domain_size, self.embedding_size).init(device),
        }
    }
}

impl<B: Backend> AssignmentEmbedding<B> {
    /// x: Input assignments (batch_size, number_variables)
    /// returns:             (batch_size, number_variables, embedding_size)
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(x)
    }
}

/// Positional embedding used for identify variables in the attention masks.
#[derive(Module, Debug)]
pub struct FixedPositionalEmbedding<B: Backend> {
    /// table: pre-computed table to map variable index to position. Tensor of shape (16384, embedding_size)
    table: Tensor<B, 2>,
}

impl<B: Backend> FixedPositionalEmbedding<B> {
    /// Creates a new positional embedding for the variables. Pre-compute the table for each
    /// possible variable id (0, ..., max_len - 1) computes its (sin, cos) position.
    pub fn new(embedding_size: usize, device: &B::Device) -> Self {
        let half = embedding_size / 2;

        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / 10000f32.powf((2 * i) as f32 / embedding_size as f32))
            .collect();
        let inv_freq: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).reshape([1, half]);

        let t: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..16384, device)
            .float()
            .reshape([16384, 1]);

        let sinusoid_inp = t.matmul(inv_freq);

        let table = Tensor::cat(vec![sinusoid_inp.clone().sin(), sinusoid_inp.cos()], 1);

        FixedPositionalEmbedding { table }
    }

    /// x: Input variable ids (batch_size, number_variables)
    /// returns:              (batch_size, number_variables, embedding_size)
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, number_variables] = x.dims();
        let embedding_size = self.table.dims()[1];

        let flat_ids: Tensor<B, 1, Int> = x.reshape([batch_size * number_variables]);
        // (batch_size * number_variables, embedding_size)
        let gathered = self.table.clone().select(0, flat_ids);

        gathered.reshape([batch_size, number_variables, embedding_size])
    }
}

/// Sums a `FixedPositionalEmbedding` per positional axis (e.g. Sudoku: row
/// axis + column axis; nurse rostering: nurse/day/shift axes), built from a
/// `PositionalStructure`. Each axis is looked up independently and the
/// results are summed (rather than concatenated, the way the original
/// ConsFormer's hardcoded 2-axis case does) -- summing composes to any
/// number of axes without needing `embedding_size` to be divisible by the
/// axis count, and matches how the rest of the embedding mix (assignment,
/// mask, position) is already combined by `EmbeddingMixer`.
#[derive(Module, Debug)]
pub struct StructuredPositionalEmbedding<B: Backend> {
    /// One table per axis, each sized to that axis's cardinality.
    axes: Vec<FixedPositionalEmbedding<B>>,
    #[module(skip)]
    axis_ids: Vec<Vec<usize>>,
}

impl<B: Backend> StructuredPositionalEmbedding<B> {
    /// Creates a new structural positional embedding
    pub fn new(
        embedding_size: usize,
        num_axes: usize,
        positions: Vec<Vec<usize>>,
        device: &B::Device,
    ) -> Self {
        let axes = vec![FixedPositionalEmbedding::new(embedding_size, device); num_axes];

        let axis_ids: Vec<Vec<usize>> = (0..num_axes)
            .map(|a| positions.iter().map(|coords| coords[a]).collect())
            .collect();

        StructuredPositionalEmbedding {
            axes,
            axis_ids: axis_ids,
        }
    }

    /// batch_size: number of parallel assignments in this forward call.
    /// returns: (batch_size, number_vars, embedding_size), summed across
    /// axes.
    pub fn forward(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        let mut acc: Option<Tensor<B, 3>> = None;
        for (axis_embed, ids) in self.axes.iter().zip(self.axis_ids.iter()) {
            let number_vars = ids.len();
            let ids_flat: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
            let ids_2d: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(ids_flat.as_slice(), device)
                    .reshape([1, number_vars])
                    .repeat_dim(0, batch_size);
            let embed = axis_embed.forward(ids_2d);
            acc = Some(match acc {
                None => embed,
                Some(prev) => prev + embed,
            });
        }
        acc.expect("PositionalStructure must have at least one axis")
    }
}

#[derive(Module, Debug)]
pub struct EmbeddingMixer<B: Backend> {
    /// Weight the assignment embedding for mixing. Learnable float parameter.
    assignment_weight: Param<Tensor<B, 3>>,
    /// Weight the mask (i.e., which variable is being modified) embedding for mixing. Learnable float parameter.
    mask_weight: Param<Tensor<B, 3>>,
    /// Weight the positional embedding for mixing. Learnable float parameter.
    position_weight: Param<Tensor<B, 3>>,
}

#[derive(Config, Debug)]
pub struct EmbeddingMixerConfig {
    pub embedding_size: usize,
}

impl EmbeddingMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingMixer<B> {
        EmbeddingMixer {
            assignment_weight: Param::from_tensor(Tensor::ones([1, 1, 1], device)),
            mask_weight: Param::from_tensor(Tensor::ones([1, 1, 1], device)),
            position_weight: Param::from_tensor(Tensor::ones([1, 1, 1], device)),
        }
    }
}

impl<B: Backend> EmbeddingMixer<B> {
    /// assignment_embeddings: Embeddings of the current assignment. Tensor of shape (batch_size, number_var, embedding_size)
    /// mask_embedding: Learned mask for the embeddings.             Tensor of shape (embedding_size,)
    /// position_embeddings: Positional embeddings                   Tensor of shape (batch_size, number_var, embedding_size)
    /// mask_vars: Mask indicating which variable to change.         Tensor of shape (bathc_size, number_var)
    pub fn forward(
        &self,
        assignment_embeddings: Tensor<B, 3>,
        mask_embedding: Tensor<B, 1>,
        position_embeddings: Option<Tensor<B, 3>>,
        mask_vars: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, embed_dim] = assignment_embeddings.dims();
        // Reshaping the tensors so that they have the right dimensions
        let mask_inds_3d = mask_vars.reshape([batch, seq_len, 1]);
        let mask_embed_bc = mask_embedding.reshape([1, 1, embed_dim]);
        // Combine the mask as a linear combination of the embeddings
        let combined = match position_embeddings {
            None => assignment_embeddings * self.assignment_weight.val(),
            Some(pos) => {
                assignment_embeddings * self.assignment_weight.val()
                    + pos * self.position_weight.val()
            }
        };

        // Computing the mask contribution to the embedding
        let masked_component = mask_embed_bc * self.mask_weight.val();
        let mask_inds_f = mask_inds_3d.float();

        combined + masked_component * mask_inds_f
    }
}

// --- Feed forward block --- //

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    drop: Dropout,
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub hidden_size: usize,
    pub expand_size: usize,
    #[config(default = 0.1)]
    pub drop: f64,
    #[config(default = true)]
    pub bias: bool,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            fc1: LinearConfig::new(self.hidden_size, self.expand_size)
                .with_bias(self.bias)
                .init(device),
            act: Gelu::new(),
            fc2: LinearConfig::new(self.expand_size, self.hidden_size)
                .with_bias(self.bias)
                .init(device),
            drop: DropoutConfig::new(self.drop).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.fc2.forward(x);
        self.drop.forward(x)
    }
}

// --- Multi-head attention --- //

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    drop_out: Dropout,
    projection: Linear<B>,
    hidden_size: usize,
    head_count: usize,
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// Size of the embedding of the assignments
    pub embedding_size: usize,
    /// Size of the hidden representation of the multi-head attention block
    pub hidden_size: usize,
    /// Number of heads
    #[config(default = 8)]
    pub head_count: usize,
    /// Drop-out used during training
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        MultiHeadAttention {
            q_proj: LinearConfig::new(self.embedding_size, self.hidden_size * self.head_count)
                .init(device),
            k_proj: LinearConfig::new(self.embedding_size, self.hidden_size * self.head_count)
                .init(device),
            v_proj: LinearConfig::new(self.embedding_size, self.hidden_size * self.head_count)
                .init(device),
            drop_out: DropoutConfig::new(self.dropout).init(),
            projection: LinearConfig::new(self.hidden_size * self.head_count, self.hidden_size)
                .init(device),
            hidden_size: self.hidden_size,
            head_count: self.head_count,
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    /// x: Embedding of the current assignment. Tensor of shape (batch_size, number_vars, embedding_size)
    /// attention_mask: Mask the attention of variables not neighbors in the primal graph of the
    /// problem.                                Tensor of shape (batch_size, number_vars, number_vars)
    /// Returns: TODO
    pub fn forward(&self, x: Tensor<B, 3>, attention_mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let [batch_size, number_vars, _] = x.dims();

        // What information each variable is looking from other
        let queries = self.reshape_qkv(self.q_proj.forward(x.clone()), batch_size, number_vars);
        // What information each variable advertise about itself
        let keys = self.reshape_qkv(self.k_proj.forward(x.clone()), batch_size, number_vars);
        // What content this variable offers
        let values = self.reshape_qkv(self.v_proj.forward(x), batch_size, number_vars);

        // Energy between to variables i, j: energy[i,j] = (Q[i] x K[j]) / sqrt(hidden_size)
        // The energy between variables i compared to j measure how the query for i (what variable
        // i looks for) is aligned with what information j offers.
        // The higher the dot product, the more the two variables must be taken into account when
        // generating a new assignment.
        // The dot product is normalised to avoid explosion when dimensionality of the hidden size
        // grows.
        let mut energy = queries.matmul(keys.transpose()) / (self.hidden_size as f64).sqrt();

        // Applies the attention mask. Remove the energy between variables that are not linked in
        // the primal graph of the CSP
        let mask_4d: Tensor<B, 4, Bool> =
            attention_mask.reshape([batch_size, 1, number_vars, number_vars]);
        energy = energy.mask_fill(mask_4d, f32::NEG_INFINITY);

        // Create a probability distribution from the energy mask
        let attention = burn::tensor::activation::softmax(energy, 3);
        // Drop some attention during training for generalisation
        let attention = self.drop_out.forward(attention);

        // At this point, the attention combined, for each variable, a weighted sum of its
        // neighbors (in the primal graph) learned representation and normalised into a probability
        // distribution (with random drop-out during training). Hence, attention[i,j] represents
        // how much the value of variable j impact variable i

        // attention: (batch_size, head, number_vars, number_var)
        // values:    (batch_size, head, number_vars, hidden_size)
        //
        // out:       (batch_size, number_heads, number_vars, hidden_size)
        let out = attention.matmul(values);
        //            (batch_size, number_vars, number_heads hidden_size)
        let out = out.swap_dims(1, 2);
        //            (batch_size, number_vars, number_heads*hidden_size)
        let out = out.reshape([batch_size, number_vars, self.head_count * self.hidden_size]);

        // Project back to (batch_size, number_vars, hidden_size)
        // TODO: note, in the ConsFormer architecture it is assumed that hidden_size == embedding_size.
        //       It is unclear to me that we need this and how it affects learning. Especially
        //       since it seems to not be the norm when using transformers.
        self.projection.forward(out)
    }

    /// x: Input tensor for attention mechanism, either the queries (q), the keys (k), or the
    ///    values (v). Tensors of shape (batch_size, number_vars, number_heads*hidden_size)
    /// batch_size: usize
    /// number_var: usize
    ///
    /// Returns a reshaped vector of shape (batch_size, number_vars, number_heads, hidden_size)
    /// further swapped to (batch_size, number_heads, number_vars, hidden_size)
    fn reshape_qkv(&self, x: Tensor<B, 3>, batch_size: usize, number_vars: usize) -> Tensor<B, 4> {
        x.reshape([batch_size, number_vars, self.head_count, self.hidden_size])
            .swap_dims(1, 2)
    }
}

// --- Transformer --- //

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ffn: FeedForward<B>,
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    /// Size of the embedding of the assignments
    pub embedding_size: usize,
    /// Size of the hidden representation of the multi-head attention block
    pub hidden_size: usize,
    /// Number of heads in the multi-head attention block
    pub num_heads: usize,
    pub expand_size: usize,
    /// Dropout of the multi-head attention block during training
    #[config(default = 0.1)]
    pub attn_drop: f64,
    /// Dropout of the feed-forward block during training
    #[config(default = 0.1)]
    pub ffn_drop: f64,
    /// If present, include bias in the feed-forward block
    #[config(default = true)]
    pub bias: bool,
}

impl TransformerBlockConfig {
    /// Initialise the transformer block. It consist of the following steps:
    ///    1. Normalisation of the input (shift to 0 mean, unit variance). Useful to avoid value
    ///       explosion when stacking layers
    ///    2. Multi-head attention block. Computes the interaction between each variable and
    ///       project back to the embedding size. After this block, each variable has gather
    ///       information about its neighbors (in the primal graph) to guide its own distribution
    ///       generation
    ///    3. A new normalisation layer
    ///    4. A feed-forward block that allows each variable to look at its internal weight
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            norm1: LayerNormConfig::new(self.hidden_size).init(device),
            attn: MultiHeadAttentionConfig::new(self.embedding_size, self.hidden_size)
                .with_head_count(self.num_heads)
                .with_dropout(self.attn_drop)
                .init(device),
            norm2: LayerNormConfig::new(self.hidden_size).init(device),
            ffn: FeedForwardConfig::new(self.hidden_size, self.expand_size)
                .with_drop(self.ffn_drop)
                .with_bias(self.bias)
                .init(device),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, attention_mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        // Applies the first normaliation block and the multi-head attention block
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), attention_mask);
        // Applies the second normalisation block and the feed-forward block
        x.clone() + self.ffn.forward(self.norm2.forward(x))
    }
}

// --- Consformer architecture, embedding the transformer blocks --- //

#[derive(Module, Debug)]
pub struct ConsFormer<B: Backend> {
    /// Learned embedding for the masked variables
    pub(crate) mask_embedding: Param<Tensor<B, 1>>,
    /// Module to map assignments to their embedding
    pub(crate) assignment_embedding: AssignmentEmbedding<B>,
    /// Embedding mixer
    pub(crate) embedding_mixer: EmbeddingMixer<B>,
    /// All transformer blocks
    pub(crate) transformer_blocks: Vec<TransformerBlock<B>>,
    /// Linear layer at the end to re-combine the output of the last multi-head transformer block
    pub(crate) head: Linear<B>,
    /// Positional structure, built from the config's `PositionalEncoding`
    /// (see `ConsFormerConfig::init`). `None` means the network gets no
    /// positional signal at all and relies purely on attention
    pub(crate) position_embedding: Option<StructuredPositionalEmbedding<B>>,
    /// Logit scaling factor (`ConsFormerConfig::tau`).
    #[module(skip)]
    pub(crate) tau: f64,
}

impl<B: Backend, Ba: ConsFormerInputs<B>> Network<B, Ba> for ConsFormer<B> {
    fn forward(&self, batch: &Ba) -> Tensor<B, 3> {
        let assignments = batch.assignments();
        let [batch_size, _seq_len] = assignments.dims();
        let device = assignments.device();

        let position_embeds = self
            .position_embedding
            .as_ref()
            .map(|pe| pe.forward(batch_size, &device));

        let x = self.assignment_embedding.forward(assignments);
        let mut x = self.embedding_mixer.forward(
            x,
            self.mask_embedding.val(),
            position_embeds,
            batch.var_masks(),
        );

        for block in &self.transformer_blocks {
            x = block.forward(x, batch.attention_masks());
        }

        self.head.forward(x).div_scalar(self.tau)
    }
}
