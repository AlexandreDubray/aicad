//! Chapter "Training a network", which needs the `learning` feature.

use std::sync::Arc;

use aicad::learning::TrainingConfig;
use aicad::learning::consformer::train_consformer_mdd;
use aicad::local_search::{
    NeuralPriorBpSearchConfig, NeuralPriorSearchConfig, SearchConfig, SolveConfig, solve,
};
use aicad::mdd::MddCompilationConfig;
use aicad::modelling::Status;
use aicad::neural::consformer::ConsFormerConfig;
use burn::config::Config;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::local_search::coloring;

#[test]
fn a_network_is_trained_then_used_as_the_prior_of_a_search() {
    let checkpoint = std::env::temp_dir().join(format!("aicad-rust-guide-{}", std::process::id()));

    // [docs:start data]
    // The problems have their own constraints but the same variables. The problems to evaluate on
    // are generated apart and never given to training, so that the network cannot have seen them.
    let mut rng = StdRng::seed_from_u64(0);
    let mut training: Vec<_> = (0..108).map(|_| coloring(&mut rng, 20, 3, 3.5)).collect();
    let test: Vec<_> = (0..12).map(|_| coloring(&mut rng, 20, 3, 3.5)).collect();

    // The validation problems pick the best network during training. Without them, training
    // holds out 20% of the problems itself.
    let validation: Vec<Arc<_>> = training.split_off(training.len() - 12);
    println!(
        "{} training, {} validation, {} test",
        training.len(),
        validation.len(),
        test.len()
    );
    // [docs:end data]

    // [docs:start train]
    // The architecture and the optimisation are settings of two structs. The network here is tiny
    // and trained for a few epochs, so it will not beat belief propagation alone on problems this
    // easy: the example shows the workflow.
    let network = ConsFormerConfig::new(
        3,   // domain_size: the number of values the network can predict
        16,  // embedding_size
        16,  // hidden_size
        2,   // num_heads
        32,  // expand_size
        0,   // positional_encoding_dimensions
        0.5, // mask_fraction
        0.1, // tau
    );
    let training_config = TrainingConfig::new()
        .with_num_epochs(4)
        .with_validation_interval(2)
        .with_batch_size(32);

    // `consformer-mdd` scores the network with the exact probability that its predictions satisfy
    // each constraint, computed on the constraint's diagram, labelled after a round of belief
    // propagation (0 scores the probabilities as they are). The best network and its
    // configuration are written to the checkpoint directory.
    train_consformer_mdd(
        training,
        Some(validation),
        network,
        training_config,
        MddCompilationConfig::default(),
        1, // rounds of belief propagation that label the diagrams before the loss
        &checkpoint,
    )
    .unwrap();
    // [docs:end train]

    // [docs:start search]
    // The checkpoint is a parameter of the neural searches, which otherwise take the same options
    // as the belief-propagation one. `NeuralPrior` repairs with the network directly;
    // `NeuralPriorBp` refines its predictions with belief propagation before choosing.
    let limits = SolveConfig::default()
        .with_runtime_limit(10)
        .with_iteration_limit(100)
        .with_seed(0);
    let searches = [
        (
            "neural prior",
            SearchConfig::NeuralPrior(NeuralPriorSearchConfig {
                destroy_fraction: 0.2,
                ..NeuralPriorSearchConfig::default()
            }),
        ),
        (
            "neural prior + BP",
            SearchConfig::NeuralPriorBp(NeuralPriorBpSearchConfig {
                destroy_fraction: 0.2,
                ..NeuralPriorBpSearchConfig::default()
            }),
        ),
    ];
    for (name, search) in searches {
        let config = limits.clone().with_search(search);
        let solutions = solve(&test, &config, Some(&checkpoint)).unwrap();
        let solved = solutions
            .iter()
            .filter(|s| s.status() == Status::Satisfiable)
            .count();
        println!(
            "{name:>20}: solved {solved} of {} test problems",
            test.len()
        );
    }
    // [docs:end search]

    // [docs:start reuse]
    // A checkpoint is a directory: the configuration is `config.json`, the weights are next to it.
    // New problems that fit the network are solved with it directly. `with_batch_size(1)` solves
    // them one after the other, each with its own limits, as a deployment would.
    let network = ConsFormerConfig::load(checkpoint.join("config.json")).unwrap();
    println!("a network over {} values", network.domain_size);

    let mut rng = StdRng::seed_from_u64(123);
    let new_problems: Vec<_> = (0..3).map(|_| coloring(&mut rng, 20, 3, 3.5)).collect();
    let config = limits
        .with_batch_size(1)
        .with_search(SearchConfig::NeuralPriorBp(NeuralPriorBpSearchConfig {
            destroy_fraction: 0.2,
            ..NeuralPriorBpSearchConfig::default()
        }));
    for (problem, solution) in new_problems
        .iter()
        .zip(solve(&new_problems, &config, Some(&checkpoint)).unwrap())
    {
        println!(
            "{:?} after {:?} iterations",
            solution.status(),
            solution.iterations()
        );
        if let Some(assignment) = solution.assignment() {
            assert!(problem.is_solution(assignment));
        }
    }
    // [docs:end reuse]

    assert_eq!(network.domain_size, 3);
    assert!(checkpoint.join("weights.mpk").exists());
    std::fs::remove_dir_all(checkpoint).ok();
}
