use std::sync::Arc;
use std::time::{Duration, Instant};

use aicad::learning::consformer::{ConsFormer, ConsFormerConfig};
use aicad::learning::NetworkConfig;
use aicad::modelling::{all_different, Problem, VariableIndex};
use aicad::nls::decode::{Argmax, DecodingOperator};
use aicad::nls::destroy::{DestroyOperator, RandomDestroy};
use aicad::nls::{Budget, NeuralLocalSearch, Solution};

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;

/// Iterations run per measurement point. Fixed (not time-based) so every
/// point does exactly the same amount of work, however fast or slow that
/// batch size turns out to be.
const ITERATIONS: usize = 30;

const POP_FOR_PROBLEM_SWEEP: usize = 1;
const PROBLEMS_FOR_POP_SWEEP: usize = 1;

const PROBLEM_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const POPULATION_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

/// Network hyper-parameters. Keep these in line with whatever config you're
/// actually training/benchmarking with -- network size shifts where the
/// "knee" (batching stops being free) sits.
fn network_config() -> ConsFormerConfig {
    ConsFormerConfig {
        domain_size: 9,
        embedding_size: 128,
        hidden_size: 128,
        num_heads: 1,
        expand_size: 128,
        num_layers: 1,
        drop_out: 0.0,
        bias: true,
        positional_encoding: None,
        mask_fraction: 0.5,
        tau: 0.1,
    }
}

const SUDOKU_N: usize = 9;
const SUDOKU_BLOCK: usize = 3;

fn sudoku_cell(r: usize, c: usize) -> usize {
    r * SUDOKU_N + c
}

/// A standard 9x9 Sudoku with an empty grid (no givens). Only the *shape*
/// (81 variables, domain size 9, row/col/block all-different constraints)
/// matters here -- it's never meant to be solved, just to drive the
/// attention mask/embedding sizes the network processes each iteration.
fn empty_sudoku() -> Problem {
    let mut problem = Problem::default();

    for _ in 0..SUDOKU_N * SUDOKU_N {
        problem.add_variable((0..SUDOKU_N as isize).collect(), None);
    }

    for r in 0..SUDOKU_N {
        let row: Vec<VariableIndex> = (0..SUDOKU_N).map(|c| VariableIndex(sudoku_cell(r, c))).collect();
        all_different(&mut problem, row);
    }
    for c in 0..SUDOKU_N {
        let col: Vec<VariableIndex> = (0..SUDOKU_N).map(|r| VariableIndex(sudoku_cell(r, c))).collect();
        all_different(&mut problem, col);
    }
    for br in 0..SUDOKU_BLOCK {
        for bc in 0..SUDOKU_BLOCK {
            let mut cells: Vec<VariableIndex> = Vec::with_capacity(SUDOKU_BLOCK * SUDOKU_BLOCK);
            for dr in 0..SUDOKU_BLOCK {
                for dc in 0..SUDOKU_BLOCK {
                    cells.push(VariableIndex(sudoku_cell(
                        br * SUDOKU_BLOCK + dr,
                        bc * SUDOKU_BLOCK + dc,
                    )));
                }
            }
            all_different(&mut problem, cells);
        }
    }

    problem
}

/// Runtime CUDA availability check, duplicated from `pyaicad::learn::cuda_available`
/// (that one is crate-private to the pyo3 bindings module, so it isn't reachable
/// from an example). See that copy for why this is how burn's "auto-detection" works.
fn cuda_available() -> bool {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        let device = CudaDevice::default();
        let tensor = burn::tensor::Tensor::<Cuda, 1>::from_data([0.0f32], &device);
        let _ = tensor.into_data();
    });
    std::panic::set_hook(prev_hook);
    result.is_ok()
}

/// Measures average wall-clock seconds per NLS iteration for a batch of
/// `num_problems` copies of `problem`, each solved with `population_size`
/// parallel candidates.
fn measure<B: Backend>(
    device: &B::Device,
    config: &ConsFormerConfig,
    problem: &Arc<Problem>,
    num_problems: usize,
    population_size: usize,
    iterations: usize,
) -> f64 {
    let problems: Vec<Arc<Problem>> = vec![Arc::clone(problem); num_problems];

    let network = config.init(device);
    let destroy_op: Box<dyn DestroyOperator> = Box::new(RandomDestroy { fraction: 0.3 });
    let decode_op: Box<dyn DecodingOperator<B>> = Box::new(Argmax);
    let nls = NeuralLocalSearch::<B, ConsFormer<B>>::new(
        network,
        destroy_op,
        decode_op,
        population_size,
        device.clone(),
    );

    // Warm-up, uncounted: burn (especially the CUDA/cubecl backend) lazily
    // compiles and caches kernels on first use, so an uncounted short run
    // keeps that one-time cost out of the measurement.
    let warmup_budget = Budget {
        time_limit: Duration::MAX,
        iteration_limit: 2,
    };
    let _ = nls.run(&problems, warmup_budget, 0);

    let budget = Budget {
        time_limit: Duration::MAX,
        iteration_limit: iterations,
    };
    let start = Instant::now();
    let solutions = nls.run(&problems, budget, 42);
    let elapsed = start.elapsed();

    let actual_iters = solutions.iter().map(Solution::iterations).max().unwrap_or(0);
    if actual_iters < iterations {
        eprintln!(
            "  warning: {num_problems} problem(s) x population {population_size} converged after \
             {actual_iters}/{iterations} iterations (an instance got solved by chance with the \
             untrained network) -- timing for this point may be slightly off"
        );
    }

    elapsed.as_secs_f64() / iterations as f64
}

fn run_sweeps<B: Backend>(device: B::Device, config: &ConsFormerConfig) {
    let problem = Arc::new(empty_sudoku());

    eprintln!("Sweep 1/2: num_problems (population_size = {POP_FOR_PROBLEM_SWEEP})");
    for &num_problems in PROBLEM_COUNTS {
        eprintln!("  num_problems = {num_problems}");
        let secs_per_iter = measure::<B>(
            &device,
            config,
            &problem,
            num_problems,
            POP_FOR_PROBLEM_SWEEP,
            ITERATIONS,
        );
        let total_rows = num_problems * POP_FOR_PROBLEM_SWEEP;
        let total_secs = secs_per_iter * ITERATIONS as f64;
        println!(
            "num_problems,{num_problems},{total_rows},{ITERATIONS},{total_secs:.6},{secs_per_iter:.6}"
        );
    }

    eprintln!("Sweep 2/2: population_size (num_problems = {PROBLEMS_FOR_POP_SWEEP})");
    for &population_size in POPULATION_SIZES {
        eprintln!("  population_size = {population_size}");
        let secs_per_iter = measure::<B>(
            &device,
            config,
            &problem,
            PROBLEMS_FOR_POP_SWEEP,
            population_size,
            ITERATIONS,
        );
        let total_rows = PROBLEMS_FOR_POP_SWEEP * population_size;
        let total_secs = secs_per_iter * ITERATIONS as f64;
        println!(
            "population_size,{population_size},{total_rows},{ITERATIONS},{total_secs:.6},{secs_per_iter:.6}"
        );
    }
}

fn main() {
    let config = network_config();
    println!("sweep,x,total_rows,iterations,total_secs,secs_per_iter");

    if cuda_available() {
        eprintln!("Running on CUDA");
        run_sweeps::<Cuda>(CudaDevice::default(), &config);
    } else {
        eprintln!(
            "CUDA not available -- falling back to NdArray (CPU). Expect secs_per_iter to grow \
             with batch size here; that's the CPU baseline, not the GPU behaviour this benchmark \
             is meant to characterise."
        );
        run_sweeps::<NdArray>(NdArrayDevice::default(), &config);
    }
}
