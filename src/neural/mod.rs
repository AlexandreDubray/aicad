//! The neural networks of the crate, and what both training and local search need to use them.
//!
//! # Meaning
//!
//! A network reads a [`Batch`] of problems with their current assignments and returns one logit
//! per value of every variable ([`Network`]). It is built from a [`NetworkConfig`] and the
//! problems it will read, and [`load_network`] brings a trained one back from a checkpoint.
//! [`consformer`] is the architecture the crate provides. `learning` trains these networks and
//! `local_search` runs them; neither depends on the other.
//!
//! # Design notes
//!
//! - **What a batch can hold is the architecture's business.** [`Batch::check_problems`] states
//!   the requirement (ConsFormer needs the same variables in every problem) next to the code that
//!   relies on it, and every caller asks the batch type instead of assuming it.

pub(crate) mod consformer;

use std::path::Path;

use std::sync::Arc;

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::modelling::Problem;

/// A group of problems prepared for one forward pass of a network: what the network reads, what
/// the loss scores and what the training monitor decodes.
///
/// Every row is one problem. What problems a batch can hold together (for instance whether they
/// need the same number of variables) is a requirement of each architecture, checked by its own
/// batch type.
pub(crate) trait Batch<B: Backend>: Clone + Send + Sync + std::fmt::Debug + 'static {
    /// The problems of the batch, one per row.
    fn problems(&self) -> &[Arc<Problem>];

    /// The current value of every variable, of shape `(batch_size, number_vars)`: one row per
    /// problem.
    fn assignments(&self) -> Tensor<B, 2, Int>;

    /// The variables the network must re-predict, of shape `(batch_size, number_vars)`. Every
    /// other variable keeps its value in [`Batch::assignments`].
    fn var_masks(&self) -> Tensor<B, 2, Bool>;

    /// The assignments the network proposes for `logits` (shape `(batch_size, number_vars,
    /// domain_size)`), one row per problem with one value per variable of that problem. Only the
    /// variables of [`Batch::var_masks`] take the network's choice; every other keeps its value in
    /// [`Batch::assignments`], as it does in local search.
    fn decode(&self, logits: Tensor<B, 3>) -> Vec<Vec<isize>>;

    /// Whether `problems` can be handled together by this kind of batch: the requirement of the
    /// architecture on the problems it reads (for instance that they all have the same variables).
    /// Returns a message explaining what is wrong when they cannot.
    fn check_problems(problems: &[Arc<Problem>]) -> Result<(), String>;

    /// Builds the batch for `assignments` (shape `(problems.len(), number_vars)`) at inference
    /// time, where `destroy_mask` holds 1 for the variables to re-predict and 0 for the others.
    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Self;
}

pub(crate) trait Network<B: Backend, Ba: Batch<B>>: Module<B> {
    fn forward(&self, batch: &Ba) -> Tensor<B, 3>;
}

pub(crate) trait NetworkConfig<B: Backend> {
    type N: Module<B>;

    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N;
}

/// Loads a network's hyperparameters (JSON config) and trained weights from a checkpoint
/// directory produced by `train_model`/`run_training`. Returns the config alongside the network
/// so callers also have access to training hyperparameters.
pub(crate) fn load_network<B, NC>(
    checkpoint_dir: &Path,
    problems: &[Arc<Problem>],
    device: &B::Device,
) -> Result<(NC, NC::N), Box<dyn std::error::Error>>
where
    B: Backend,
    NC: NetworkConfig<B> + Config + Clone,
    NC::N: Module<B>,
{
    // First, loads the config from the checkpoint
    let config: NC = NC::load(checkpoint_dir.join("config.json"))?;
    // Inits the network from the config and loads the weights.
    // The network initialisation is responsible for initiating problem-specific network's
    // component. For example, in ConsFormer, the variables positions are used in the positional
    // embedding. See each architecture's doc for further comment.
    let network = config.clone().init(problems, device).load_file(
        checkpoint_dir.join("weights"),
        &CompactRecorder::new(),
        device,
    )?;
    Ok((config, network))
}
