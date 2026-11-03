//! A network defined outside the library, trained and used through the public API only.
//!
//! This file is compiled as a separate crate, so it can only use what a downstream crate can: the
//! test fails to build if a step of the pipeline stops being reachable from outside. The network is
//! a toy (it reads one variable at a time and learns a made-up target), since the point is the
//! plumbing: a [`Batch`], a [`Network`], a [`NetworkConfig`], a [`Loss`], training with
//! [`train_model`], loading with [`load_network`], and searching with [`solve_with_network`].
#![cfg(feature = "learning")]

use std::sync::Arc;

use aicad::constraints::all_different;
use aicad::learning::{Loss, TrainingConfig, train_model};
use aicad::local_search::{NeuralPriorSearchConfig, SearchConfig, SolveConfig, solve_with_network};
use aicad::modelling::Problem;
use aicad::neural::{
    Batch, InferenceJob, Network, NetworkConfig, TrainingJob, load_network, run_inference,
    run_training,
};
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::ElementConversion;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const COLOURS: usize = 3;

// --- The batch: what the network reads ------------------------------------------------------

// [docs:start batch]
#[derive(Clone, Debug)]
struct ToyBatch<B: Backend> {
    problems: Vec<Arc<Problem>>,
    assignments: Tensor<B, 2, Int>,
    var_masks: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batch<B> for ToyBatch<B> {
    fn problems(&self) -> &[Arc<Problem>] {
        &self.problems
    }

    fn assignments(&self) -> Tensor<B, 2, Int> {
        self.assignments.clone()
    }

    fn var_masks(&self) -> Tensor<B, 2, Bool> {
        self.var_masks.clone()
    }

    fn decode(&self, logits: Tensor<B, 3>) -> Vec<Vec<isize>> {
        let proposed: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);
        let assignment = self
            .assignments
            .clone()
            .mask_where(self.var_masks.clone(), proposed);
        let width = assignment.dims()[1];
        let values: Vec<isize> = assignment
            .into_data()
            .to_vec::<B::IntElem>()
            .unwrap()
            .into_iter()
            .map(|v| v.elem::<i64>() as isize)
            .collect();
        values.chunks(width).map(<[isize]>::to_vec).collect()
    }

    fn check_problems(problems: &[Arc<Problem>]) -> Result<(), String> {
        // The toy reads rectangular tensors: the problems must have as many variables.
        let n = problems[0].number_variables();
        match problems.iter().find(|p| p.number_variables() != n) {
            Some(p) => Err(format!(
                "the toy network needs {n} variables in every problem, found {}",
                p.number_variables()
            )),
            None => Ok(()),
        }
    }

    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        _device: &B::Device,
    ) -> Self {
        ToyBatch {
            problems: problems.to_vec(),
            assignments,
            var_masks: destroy_mask.greater_elem(0),
        }
    }
}

// [docs:end batch]
// --- How training builds a batch -------------------------------------------------------------

// [docs:start batcher]
#[derive(Clone)]
struct ToyBatcher;

impl<B: Backend> Batcher<B, Arc<Problem>, ToyBatch<B>> for ToyBatcher {
    fn batch(&self, problems: Vec<Arc<Problem>>, device: &B::Device) -> ToyBatch<B> {
        let n = problems[0].number_variables();
        let mut rng = StdRng::seed_from_u64(0);
        let values: Vec<i64> = (0..problems.len() * n)
            .map(|_| rng.random_range(0..COLOURS as i64))
            .collect();
        // Every variable is re-predicted.
        let masks = vec![true; problems.len() * n];
        ToyBatch {
            assignments: Tensor::from_data(TensorData::new(values, [problems.len(), n]), device),
            var_masks: Tensor::from_data(TensorData::new(masks, [problems.len(), n]), device),
            problems,
        }
    }
}

// [docs:end batcher]
// --- The network and its configuration --------------------------------------------------------

// [docs:start network]
#[derive(Module, Debug)]
struct Toy<B: Backend> {
    embedding: Embedding<B>,
    output: Linear<B>,
}

impl<B: Backend> Network<B, ToyBatch<B>> for Toy<B> {
    fn forward(&self, batch: &ToyBatch<B>) -> Tensor<B, 3> {
        self.output
            .forward(self.embedding.forward(batch.assignments()))
    }
}

#[derive(Config, Debug)]
struct ToyConfig {
    domain_size: usize,
    hidden: usize,
}

impl<B: Backend> NetworkConfig<B> for ToyConfig {
    type N = Toy<B>;

    fn init(&self, _problems: &[Arc<Problem>], device: &B::Device) -> Toy<B> {
        Toy {
            embedding: EmbeddingConfig::new(self.domain_size, self.hidden).init(device),
            output: LinearConfig::new(self.hidden, self.domain_size).init(device),
        }
    }
}

// [docs:end network]
// --- The loss ---------------------------------------------------------------------------------

// [docs:start loss]
/// A made-up target: the network learns to predict the value 0 everywhere.
struct ZeroLoss;

impl<B: Backend> Loss<B, ToyBatch<B>> for ZeroLoss {
    fn loss(&self, logits: Tensor<B, 3>, _batch: &ToyBatch<B>) -> Tensor<B, 1> {
        let [b, n, _] = logits.dims();
        log_softmax(logits, 2)
            .narrow(2, 0, 1)
            .reshape([b * n])
            .mean()
            .neg()
            .reshape([1])
    }
}

// [docs:end loss]
// --- The pipeline, on the backend the library picks --------------------------------------------

fn problems(count: usize) -> Vec<Arc<Problem>> {
    (0..count)
        .map(|_| {
            let mut problem = Problem::default();
            let vars = problem.add_variables(4, (0..COLOURS as isize).collect());
            all_different(&mut problem, vars);
            Arc::new(problem)
        })
        .collect()
}

// [docs:start train]
struct Train {
    problems: Vec<Arc<Problem>>,
    dir: std::path::PathBuf,
}

impl TrainingJob for Train {
    type Output = ();

    fn run<B: AutodiffBackend>(self, device: B::Device) {
        let config = ToyConfig::new(COLOURS, 8);
        std::fs::create_dir_all(&self.dir).unwrap();
        // The training loop saves the weights; the configuration is the caller's to save.
        config.save(self.dir.join("config.json")).unwrap();
        let training = TrainingConfig::new()
            .with_num_epochs(3)
            .with_validation_interval(1)
            .with_batch_size(4);
        train_model::<B, ToyConfig, _, _, _, ZeroLoss, _, _>(
            config,
            &self.problems,
            InMemDataset::new(self.problems.clone()),
            InMemDataset::new(self.problems.clone()),
            ToyBatcher,
            ZeroLoss,
            training,
            &self.dir,
            &device,
        )
        .unwrap();
    }
}

// [docs:end train]
// [docs:start solve]
struct Solve {
    problems: Vec<Arc<Problem>>,
    dir: std::path::PathBuf,
}

impl InferenceJob for Solve {
    type Output = Result<usize, String>;

    fn run<B: Backend>(self, device: B::Device) -> Self::Output {
        ToyBatch::<B>::check_problems(&self.problems)?;
        let (_, network) = load_network::<B, ToyConfig>(&self.dir, &self.problems, &device)
            .map_err(|e| e.to_string())?;
        let config =
            SolveConfig::default()
                .with_iteration_limit(50)
                .with_search(SearchConfig::NeuralPrior(NeuralPriorSearchConfig {
                    destroy_fraction: 0.5,
                    ..NeuralPriorSearchConfig::default()
                }));
        let solutions = solve_with_network::<B, ToyBatch<B>, Toy<B>>(
            network,
            device,
            &self.problems,
            &config,
            0,
        )?;
        Ok(solutions.len())
    }
}

// [docs:end solve]
#[test]
fn a_network_from_another_crate_trains_loads_and_searches() {
    let dir = std::env::temp_dir().join(format!("aicad-custom-network-{}", std::process::id()));
    run_training(Train {
        problems: problems(8),
        dir: dir.clone(),
    });
    assert!(dir.join("config.json").exists());
    assert!(dir.join("weights.mpk").exists());

    let solved = run_inference(Solve {
        problems: problems(3),
        dir: dir.clone(),
    });
    assert_eq!(solved, Ok(3));
    std::fs::remove_dir_all(dir).ok();
}
