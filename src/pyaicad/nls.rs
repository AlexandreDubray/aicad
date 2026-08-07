use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;

use rand::RngExt;

use crate::learning::consformer::{ConsFormer, ConsFormerConfig};
use crate::modelling::Problem;
use crate::nls::decode::{Argmax, DecodingOperator, Sampling};
use crate::nls::destroy::{DestroyOperator, RandomDestroy, RelatedDestroy, WorstDestroy};
use crate::nls::{load_network, Budget, NeuralLocalSearch};

use super::learn::cuda_available;
use super::problem::PyProblem;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyNetworkKind {
    ConsFormer,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDestroyKind {
    Random,
    Worst,
    Related,
}

impl PyDestroyKind {
    fn build(&self, fraction: f64) -> Box<dyn DestroyOperator> {
        match self {
            PyDestroyKind::Random => Box::new(RandomDestroy { fraction }),
            PyDestroyKind::Worst => Box::new(WorstDestroy { fraction }),
            PyDestroyKind::Related => Box::new(RelatedDestroy { fraction }),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDecodeKind {
    Argmax,
    Sampling,
}

impl PyDecodeKind {
    fn build<B: Backend>(&self, temperature: f64) -> Box<dyn DecodingOperator<B>> {
        match self {
            PyDecodeKind::Argmax => Box::new(Argmax),
            PyDecodeKind::Sampling => Box::new(Sampling { temperature }),
        }
    }
}

/// Runs neural local search on `problem` using a network loaded from
/// `checkpoint_dir` (the `config.json` + `weights` produced by
/// `train_consformer`).
#[pyfunction]
#[pyo3(signature = (
    problem,
    checkpoint_dir,
    network_kind=PyNetworkKind::ConsFormer,
    time_limit=None,
    iteration_limit=None,
    population_size=1,
    destroy_kind=PyDestroyKind::Random,
    destroy_fraction=0.5,
    decode_kind=PyDecodeKind::Argmax,
    temperature=1.0,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn neural_local_search(
    problem: &PyProblem,
    checkpoint_dir: String,
    network_kind: PyNetworkKind,
    time_limit: Option<u64>,
    iteration_limit: Option<usize>,
    population_size: usize,
    destroy_kind: PyDestroyKind,
    destroy_fraction: f64,
    decode_kind: PyDecodeKind,
    temperature: f64,
    seed: Option<u64>,
) -> PyResult<Option<Vec<isize>>> {
    let problem = problem.arc();
    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let budget = Budget {
        time_limit: time_limit.map(Duration::from_secs).unwrap_or(Duration::MAX),
        iteration_limit: iteration_limit.unwrap_or(usize::MAX),
    };
    let seed = seed.unwrap_or_else(|| rand::rng().random_range(0..u64::MAX));
    let destroy_op = destroy_kind.build(destroy_fraction);

    if cuda_available() {
        Ok(run::<Cuda>(
            CudaDevice::default(),
            problem,
            &checkpoint_dir,
            &network_kind,
            destroy_op,
            &decode_kind,
            temperature,
            population_size,
            budget,
            seed,
        ))
    } else {
        Ok(run::<NdArray>(
            NdArrayDevice::default(),
            problem,
            &checkpoint_dir,
            &network_kind,
            destroy_op,
            &decode_kind,
            temperature,
            population_size,
            budget,
            seed,
        ))
    }
}

/// Monomorphized per-backend run: loads the requested network onto `B`,
/// builds the decode operator (backend-typed, unlike destroy), and runs the
/// search loop.
#[allow(clippy::too_many_arguments)]
fn run<B: Backend>(
    device: B::Device,
    problem: Arc<Problem>,
    checkpoint_dir: &Path,
    network_kind: &PyNetworkKind,
    destroy_op: Box<dyn DestroyOperator>,
    decode_kind: &PyDecodeKind,
    temperature: f64,
    population_size: usize,
    budget: Budget,
    seed: u64,
) -> Option<Vec<isize>> {
    let decode_op = decode_kind.build::<B>(temperature);

    match network_kind {
        PyNetworkKind::ConsFormer => {
            let network = load_network::<B, ConsFormerConfig>(checkpoint_dir, &device);
            let nls = NeuralLocalSearch::<B, ConsFormer<B>>::new(
                problem,
                network,
                destroy_op,
                decode_op,
                population_size,
                device,
            );
            nls.run(budget, seed)
        }
    }
}
