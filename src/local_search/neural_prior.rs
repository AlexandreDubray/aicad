//! The local searches driven by a trained network: NeuralPrior, which decodes the logits of the
//! network, and NeuralPriorBP, which combines them with belief propagation over the constraints.
//!
//! # Meaning
//!
//! The network reads the current assignments of a batch and returns a logit per value of every
//! variable: a prior over what each freed variable should become. [`NeuralPriorSearch`] takes
//! the most likely value (or samples one) from it. [`NeuralPriorBpSearch`] treats it as the prior
//! of belief propagation over the decomposed MDDs of the constraints, so the values it chooses
//! agree with the constraints.
//!
//! Both are generic over the network and the [`Batch`] it reads, so any architecture of
//! [`crate::neural`] can drive them. What problems a batch can hold together is the batch type's
//! requirement ([`Batch::check_problems`]): setup asks it, and nothing else here knows it.
//!
//! # How it works
//!
//! The engine keeps the assignments on the CPU. A step destroys the pending rows there, copies the
//! assignments into a `[rows, n]` tensor, evaluates the network on the whole batch, and turns its
//! logits into the next assignments. The two searches share everything but that last part.
//!
//! # Design notes
//!
//! - **The logits are the whole interface to the network.** Nothing here depends on how they are
//!   computed, which is what lets another network be plugged in.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use rand::rngs::StdRng;

use crate::destroy::{DestroyKind, DestroyOperator};
use crate::mdd::arena::MddCompilationConfig;
use crate::modelling::Problem;
use crate::neural::{Batch, Network};
use crate::sampling::DecodeMode;
use crate::utils::tensor::{rows_to_tensor, to_rows};

use super::repair::{BatchRepair, MddRepair};
use super::{SearchArchitecture, SearchState, destroy_rows, row_seeds};

/// The settings of the NeuralPrior local search, which decodes the logits of the network.
#[derive(Debug, Clone)]
pub(crate) struct NeuralPriorSearchConfig {
    /// Destroy operator choosing the variables to free at each iteration.
    pub destroy_kind: DestroyKind,
    /// Fraction of the variables the destroy operator frees at each iteration, in `[0, 1]`.
    pub destroy_fraction: f64,
    /// Sample values from the softmax of the logits instead of taking the most likely one.
    pub stochastic: bool,
    /// Softmax temperature when sampling; unused by greedy decoding.
    pub temperature: f64,
}

impl Default for NeuralPriorSearchConfig {
    fn default() -> Self {
        Self {
            destroy_kind: DestroyKind::default(),
            destroy_fraction: 1.0,
            stochastic: false,
            temperature: 1.0,
        }
    }
}

/// The settings of the NeuralPriorBP local search, which uses the probabilities of the network as
/// the prior of belief propagation over the decomposed MDDs of the constraints.
#[derive(Debug, Clone)]
pub(crate) struct NeuralPriorBpSearchConfig {
    /// Destroy operator choosing the variables to free at each iteration.
    pub destroy_kind: DestroyKind,
    /// Fraction of the variables the destroy operator frees at each iteration, in `[0, 1]`.
    pub destroy_fraction: f64,
    /// Sample values from the refined marginals instead of taking the most likely one.
    pub stochastic: bool,
    /// Number of belief-propagation rounds per iteration.
    pub bp_iterations: usize,
}

impl Default for NeuralPriorBpSearchConfig {
    fn default() -> Self {
        Self {
            destroy_kind: DestroyKind::default(),
            destroy_fraction: 1.0,
            stochastic: false,
            bp_iterations: 1,
        }
    }
}

/// The network and the destroy operator, which the two searches share.
struct Core<B: Backend, Ba, N> {
    /// The trained network evaluated at each step.
    network: N,
    /// Heuristic choosing the variables to free.
    destroy_op: Box<dyn DestroyOperator>,
    /// Device used (cpu or gpu).
    device: B::Device,
    /// The kind of batch the network reads.
    batch: PhantomData<fn() -> Ba>,
}

/// What a step learns before it chooses the new values.
struct Proposal<B: Backend> {
    /// The variables each row frees.
    destroyed: Vec<Vec<usize>>,
    /// The assignments as the network read them, `[rows, n]`.
    current: Tensor<B, 2, Int>,
    /// 1 for the freed positions, `[rows, n]`.
    destroy_mask: Tensor<B, 2, Int>,
    /// The logits of the network, `[rows, n, domain_size]`.
    logits: Tensor<B, 3>,
}

impl<B, Ba, N> Core<B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    /// Destroys the pending rows and evaluates the network on the batch.
    fn propose(
        &self,
        problems: &[Arc<Problem>],
        assignments: &[Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    ) -> Proposal<B> {
        let rows = problems.len();
        // The batch type has accepted the problems at setup, so they share their variables.
        let n = problems[0].number_variables();
        let destroyed = destroy_rows(
            self.destroy_op.as_ref(),
            problems,
            assignments,
            pending,
            rng,
        );

        // The positions the network must re-predict, which it applies in its architecture.
        let mut mask_data = vec![0i64; rows * n];
        for (row, free) in destroyed.iter().enumerate() {
            for &var in free {
                mask_data[row * n + var] = 1;
            }
        }
        let destroy_mask: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::from_data(mask_data.as_slice(), &self.device).reshape([rows, n]);

        let current = rows_to_tensor(assignments, n, &self.device);
        let batch = Ba::for_assignments(
            problems,
            current.clone(),
            destroy_mask.clone(),
            &self.device,
        );
        let logits = self.network.forward(&batch);
        Proposal {
            destroyed,
            current,
            destroy_mask,
            logits,
        }
    }
}

/// How the logits of the network become values directly: the highest logit, or a sample of the
/// softmax of the logits at a temperature.
enum Decoder {
    /// Greedy / MAP decoding: takes the most likely value per variable.
    Argmax,
    /// Stochastic decoding: samples a value per variable from `softmax(logits / temperature)`.
    Sampling {
        /// Temperature of the softmax.
        temperature: f64,
    },
}

impl Decoder {
    /// `logits` has shape `[rows, variables, domain_size]`. Only the positions flagged in
    /// `destroy_mask` change; everywhere else `current` is kept, whatever the logits say.
    fn decode<B: Backend>(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        let proposed: Tensor<B, 2, Int> = match self {
            Decoder::Argmax => logits.argmax(2).squeeze_dim(2),
            Decoder::Sampling { temperature } => {
                let device = logits.device();
                let u = Tensor::<B, 3>::random(
                    logits.dims(),
                    Distribution::Uniform(1e-20, 1.0),
                    &device,
                );
                let neg_log_u = -u.log(); // -ln(u), > 0 since u in (0, 1)
                let gumbel = -neg_log_u.log(); // Gumbel(0, 1) noise: -ln(-ln(u))
                let scaled = logits.div_scalar(*temperature) + gumbel;
                scaled.argmax(2).squeeze_dim(2)
            }
        };
        current.mask_where(destroy_mask.equal_elem(1), proposed)
    }
}

/// The NeuralPrior local search: a trained network whose logits are decoded directly.
pub(crate) struct NeuralPriorSearch<B: Backend, Ba, N> {
    /// The network and destroy operator.
    core: Core<B, Ba, N>,
    /// How the logits become values.
    decoder: Decoder,
}

impl<B, Ba, N> NeuralPriorSearch<B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    /// Builds the search from its network and settings. Nothing depends on the problems being
    /// solved, so it can be reused across batches.
    pub(crate) fn new(network: N, config: &NeuralPriorSearchConfig, device: B::Device) -> Self {
        Self {
            core: Core {
                network,
                destroy_op: config.destroy_kind.build(config.destroy_fraction),
                device,
                batch: PhantomData,
            },
            decoder: if config.stochastic {
                Decoder::Sampling {
                    temperature: config.temperature,
                }
            } else {
                Decoder::Argmax
            },
        }
    }
}

impl<B, Ba, N> SearchArchitecture for NeuralPriorSearch<B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    fn setup<'a>(&'a self, problems: &[Arc<Problem>]) -> Result<Box<dyn SearchState + 'a>, String> {
        Ba::check_problems(problems)?;
        Ok(Box::new(NeuralPriorState { search: self }))
    }
}

/// The state of a NeuralPrior search over a batch: it holds nothing besides the engine's
/// assignments.
struct NeuralPriorState<'a, B: Backend, Ba, N> {
    /// The components of the search.
    search: &'a NeuralPriorSearch<B, Ba, N>,
}

impl<B, Ba, N> SearchState for NeuralPriorState<'_, B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    fn step(
        &mut self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    ) {
        let proposal = self
            .search
            .core
            .propose(problems, assignments, pending, rng);
        let next =
            self.search
                .decoder
                .decode(proposal.logits, proposal.destroy_mask, proposal.current);
        let n = assignments.first().map_or(0, Vec::len);
        for (row, values) in assignments
            .iter_mut()
            .zip(to_rows(&next, problems.len(), n))
        {
            *row = values;
        }
    }
}

/// The NeuralPriorBP local search: a trained network whose probabilities are the prior of belief
/// propagation over the MDDs of the constraints.
pub(crate) struct NeuralPriorBpSearch<B: Backend, Ba, N> {
    /// The network and destroy operator.
    core: Core<B, Ba, N>,
    /// Number of belief-propagation rounds per iteration.
    bp_iterations: usize,
    /// How a value is taken from a marginal.
    mode: DecodeMode,
}

impl<B, Ba, N> NeuralPriorBpSearch<B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    /// Builds the search from its network and settings. Nothing depends on the problems being
    /// solved, so it can be reused across batches.
    pub(crate) fn new(network: N, config: &NeuralPriorBpSearchConfig, device: B::Device) -> Self {
        Self {
            core: Core {
                network,
                destroy_op: config.destroy_kind.build(config.destroy_fraction),
                device,
                batch: PhantomData,
            },
            bp_iterations: config.bp_iterations,
            mode: if config.stochastic {
                DecodeMode::Sample
            } else {
                DecodeMode::Greedy
            },
        }
    }
}

impl<B, Ba, N> SearchArchitecture for NeuralPriorBpSearch<B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    /// Checks the problems before compiling their constraints, which is the costly part.
    fn setup<'a>(&'a self, problems: &[Arc<Problem>]) -> Result<Box<dyn SearchState + 'a>, String> {
        Ba::check_problems(problems)?;
        let repair = MddRepair::new(
            MddCompilationConfig::default(),
            MddRepair::domain_size_covering(problems),
            self.mode,
            self.bp_iterations,
        );
        Ok(Box::new(NeuralPriorBpState {
            search: self,
            repair: BatchRepair::new(repair, problems),
        }))
    }
}

/// The state of a NeuralPriorBP search over a batch.
struct NeuralPriorBpState<'a, B: Backend, Ba, N> {
    /// The components of the search.
    search: &'a NeuralPriorBpSearch<B, Ba, N>,
    /// Repairs the freed variables of the rows, with the alphabet of this batch.
    repair: BatchRepair,
}

impl<B, Ba, N> SearchState for NeuralPriorBpState<'_, B, Ba, N>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    fn step(
        &mut self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    ) {
        let proposal = self
            .search
            .core
            .propose(problems, assignments, pending, rng);
        let seeds = row_seeds(problems.len(), rng);
        let [_, n, domain_width] = proposal.logits.dims();
        let probs: Vec<f32> = softmax(proposal.logits, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");
        self.repair.repair_rows(
            problems,
            assignments,
            &proposal.destroyed,
            &seeds,
            |row, variable, value| {
                // The network may predict fewer values than the batch's alphabet.
                if value < domain_width {
                    probs[row * n * domain_width + variable * domain_width + value] as f64
                } else {
                    0.0
                }
            },
        );
    }

    fn solved(&mut self, solved: &[bool]) {
        self.repair.solved(solved);
    }

    fn retain(&mut self, keep: &[bool]) {
        self.repair.retain(keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use crate::local_search::{Budget, LocalSearch};
    use crate::neural::NetworkConfig;
    use crate::neural::consformer::{ConsFormer, ConsFormerBatch, ConsFormerConfig};
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type Net = ConsFormer<NdArray>;
    type Batch = ConsFormerBatch<NdArray>;

    fn clique(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect());
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    /// An untrained network for `problems`.
    fn network(problems: &[Arc<Problem>]) -> Net {
        let config = ConsFormerConfig::new(3, 4, 4, 1, 4, 0, 0.5, 1.0).with_num_layers(1);
        NetworkConfig::<NdArray>::init(&config, problems, &NdArrayDevice::default())
    }

    fn engines(problems: &[Arc<Problem>]) -> Vec<LocalSearch> {
        let device = NdArrayDevice::default();
        let prior = NeuralPriorSearch::<NdArray, Batch, Net>::new(
            network(problems),
            &NeuralPriorSearchConfig::default(),
            device,
        );
        let with_bp = NeuralPriorBpSearch::<NdArray, Batch, Net>::new(
            network(problems),
            &NeuralPriorBpSearchConfig::default(),
            device,
        );
        vec![
            LocalSearch::new(Box::new(prior)),
            LocalSearch::new(Box::new(with_bp)),
        ]
    }

    #[test]
    fn both_refuse_problems_with_different_numbers_of_variables() {
        let problems = vec![clique(3, 3), clique(4, 3)];
        for engine in engines(&problems[..1]) {
            let error = engine
                .run(&problems, Budget::default(), 0, None)
                .expect_err("the problems cannot be searched together");
            assert!(error.contains("same number of variables"), "{error}");
        }
    }

    #[test]
    fn both_stay_in_the_domain_over_a_batch() {
        let problems = vec![clique(3, 3), clique(3, 3)];
        for engine in engines(&problems) {
            let budget = Budget {
                iteration_limit: 30,
                ..Budget::default()
            };
            let solutions = engine.run(&problems, budget, 0, None).unwrap();
            assert_eq!(solutions.len(), 2);
            for solution in solutions {
                if let Some(values) = solution.assignment() {
                    assert!(values.iter().all(|v| (0..3).contains(v)));
                }
            }
        }
    }

    #[test]
    fn decoding_only_changes_the_flagged_positions() {
        let device = NdArrayDevice::default();
        // Every variable's logits favour value 2.
        let logits = Tensor::<NdArray, 3>::from_data(
            burn::tensor::TensorData::new([0.0f32, 0.0, 1.0].repeat(3), [1, 3, 3]),
            &device,
        );
        let mask = Tensor::<NdArray, 2, Int>::from_data([[1, 0, 1]], &device);
        let current = Tensor::<NdArray, 2, Int>::from_data([[0, 1, 0]], &device);
        let next = Decoder::Argmax.decode(logits, mask, current);
        let values: Vec<i64> = next.into_data().to_vec().unwrap();
        assert_eq!(values, [2, 1, 2]);
    }
}
