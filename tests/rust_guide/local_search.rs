//! Chapter "Local search".

// [docs:start coloring]
use std::sync::Arc;

use aicad::constraints::not_equals;
use aicad::modelling::Problem;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A random feasible graph colouring: hide a colouring, then draw edges between nodes of other
/// colours only, so that the hidden colouring is a solution. `degree` is the average number of
/// neighbours of a node.
pub fn coloring(rng: &mut StdRng, nodes: usize, colours: usize, degree: f64) -> Arc<Problem> {
    let hidden: Vec<usize> = (0..nodes).map(|_| rng.random_range(0..colours)).collect();
    let mut problem = Problem::default();
    let colour = problem.add_variables(nodes, (0..colours as isize).collect());
    let pairs: Vec<(usize, usize)> = (0..nodes)
        .flat_map(|i| (i + 1..nodes).map(move |j| (i, j)))
        .filter(|&(i, j)| hidden[i] != hidden[j])
        .collect();
    let probability = (degree * nodes as f64 / 2.0 / pairs.len() as f64).min(1.0);
    for (i, j) in pairs {
        if rng.random::<f64>() < probability {
            not_equals(&mut problem, colour[i], colour[j]);
        }
    }
    Arc::new(problem)
}
// [docs:end coloring]

#[test]
fn belief_propagation_repairs_assignments() {
    // [docs:start search]
    use aicad::destroy::DestroyKind;
    use aicad::local_search::{BeliefPropagationSearchConfig, SearchConfig, SolveConfig, solve};
    use aicad::modelling::Status;

    let mut rng = StdRng::seed_from_u64(0);
    let problems: Vec<_> = (0..20).map(|_| coloring(&mut rng, 30, 3, 4.0)).collect();

    // The limits apply to each batch; whichever is reached first stops the search. The default
    // architecture is belief propagation, and the default destroys everything at each iteration,
    // which is a restart from scratch.
    let limits = SolveConfig::default()
        .with_runtime_limit(5)
        .with_iteration_limit(300)
        .with_seed(0);
    let solutions = solve(&problems, &limits, None).unwrap();
    let solved = solutions
        .iter()
        .filter(|s| s.status() == Status::Satisfiable)
        .count();
    println!("destroy everything: solved {solved} of {}", problems.len());

    // Destroying only a fraction of the variables makes it a real local search; `destroy_kind`
    // chooses which ones: at random, where the constraints are violated, or around a variable.
    for destroy_kind in [
        DestroyKind::Random,
        DestroyKind::Worst,
        DestroyKind::Related,
    ] {
        let search = BeliefPropagationSearchConfig {
            destroy_kind,
            destroy_fraction: 0.2,
            ..BeliefPropagationSearchConfig::default()
        };
        let config = limits
            .clone()
            .with_search(SearchConfig::BeliefPropagation(search));
        let solutions = solve(&problems, &config, None).unwrap();
        let solved = solutions
            .iter()
            .filter(|s| s.status() == Status::Satisfiable)
            .count();
        println!(
            "destroy {destroy_kind:?} 20%: solved {solved} of {}",
            problems.len()
        );
    }
    // [docs:end search]
}

#[test]
fn a_solution_carries_the_assignment_and_how_the_search_went() {
    use aicad::destroy::DestroyKind;
    use aicad::local_search::{BeliefPropagationSearchConfig, SearchConfig, SolveConfig, solve};
    use aicad::modelling::Status;

    let mut rng = StdRng::seed_from_u64(0);
    let problems: Vec<_> = (0..20).map(|_| coloring(&mut rng, 30, 3, 4.0)).collect();

    // [docs:start result]
    let search = BeliefPropagationSearchConfig {
        destroy_kind: DestroyKind::Worst,
        destroy_fraction: 0.2,
        ..BeliefPropagationSearchConfig::default()
    };
    let config = SolveConfig::default()
        .with_runtime_limit(5)
        .with_iteration_limit(300)
        .with_seed(0)
        .with_search(SearchConfig::BeliefPropagation(search));
    let solutions = solve(&problems[..1], &config, None).unwrap();
    let solution = &solutions[0];
    println!(
        "{:?} after {:?} iterations",
        solution.status(),
        solution.iterations()
    );
    if let Some(assignment) = solution.assignment() {
        assert!(problems[0].is_solution(assignment));
    }

    // The same seed gives the same run.
    let again = solve(&problems[..1], &config, None).unwrap();
    println!(
        "reproducible: {}",
        again[0].iterations() == solution.iterations()
    );
    // [docs:end result]
    assert_eq!(again[0].iterations(), solution.iterations());
    assert!(matches!(
        solution.status(),
        Status::Satisfiable | Status::Unknown
    ));
}
