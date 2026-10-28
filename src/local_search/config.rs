//! Configuration of a neural-local-search run.

use crate::destroy::DestroyKind;

/// Architecture of the network producing the priors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NetworkKind {
    /// The ConsFormer transformer.
    #[default]
    ConsFormer,
}

/// Where the priors of each step come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PriorKind {
    /// The checkpoint's network is evaluated at each step.
    #[default]
    Network,
    /// Every value gets the same score and no network is evaluated. Only informative with a
    /// decoder that uses the constraints, such as [`DecodeKind::MddSampling`].
    Uniform,
}

/// How the priors become the next assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum DecodeKind {
    /// Decode the priors directly: greedily, or by sampling when `stochastic_decode` is set.
    #[default]
    Logits,
    /// Refine the priors with belief propagation over the problem's decomposed MDDs, then decode.
    MddSampling,
}

/// Every knob of a neural-local-search run that is not the problems or the checkpoint.
#[derive(Debug, Clone)]
pub(crate) struct SolveConfig {
    /// Time limit in seconds, applied to each batch separately; `None` means no time limit.
    pub time_limit: Option<u64>,
    /// Maximum number of search iterations per batch; `None` means no iteration limit.
    pub iteration_limit: Option<usize>,
    /// Seed of the destroy sequence. `None` draws one from the process-wide generator, so it is
    /// reproducible when that generator was seeded.
    pub seed: Option<u64>,
    /// Architecture of the network producing the priors.
    pub network_kind: NetworkKind,
    /// Number of problems stepped together on the device; batches run one after the other.
    /// `None` puts every problem in a single batch. All problems of a batch must have the same
    /// number of variables.
    pub batch_size: Option<usize>,
    /// Destroy operator choosing the variables to free at each iteration.
    pub destroy_kind: DestroyKind,
    /// Fraction of the variables the destroy operator frees at each iteration, in `[0, 1]`.
    pub destroy_fraction: f64,
    /// Sample values from the decoder's distribution instead of taking the most likely one.
    pub stochastic_decode: bool,
    /// Softmax temperature of the [`DecodeKind::Logits`] decoder when sampling; unused by greedy
    /// decoding and by [`DecodeKind::MddSampling`].
    pub temperature: f64,
    /// How the priors become the next assignment.
    pub decode_kind: DecodeKind,
    /// Number of belief-propagation rounds of [`DecodeKind::MddSampling`]; unused otherwise.
    pub bp_iterations: usize,
    /// Source of the priors.
    pub prior_kind: PriorKind,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            time_limit: None,
            iteration_limit: None,
            seed: None,
            network_kind: NetworkKind::default(),
            batch_size: None,
            destroy_kind: DestroyKind::default(),
            destroy_fraction: 1.0,
            stochastic_decode: false,
            temperature: 1.0,
            decode_kind: DecodeKind::default(),
            bp_iterations: 1,
            prior_kind: PriorKind::default(),
        }
    }
}
