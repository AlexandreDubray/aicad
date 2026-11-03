//! A local-search architecture defined outside the library, run through the public API only.
//!
//! Compiled as a separate crate, like `custom_network.rs`, but without the `learning` feature:
//! plugging in a search needs no neural network. The architecture is a toy that repairs by
//! redrawing the destroyed variables at random.

use std::sync::Arc;

use aicad::constraints::all_different;
use aicad::destroy::{DestroyKind, DestroyOperator};
use aicad::local_search::{
    Budget, LocalSearch, SearchArchitecture, SearchState, destroy_rows, run_batched,
};
use aicad::modelling::{Problem, Status};
use aicad::sampling::random_assignment;
use rand::rngs::StdRng;

// [docs:start architecture]
/// Destroys some variables and gives them a random value.
struct Redraw {
    destroy: Box<dyn DestroyOperator>,
}

struct RedrawState<'a> {
    architecture: &'a Redraw,
}

impl SearchArchitecture for Redraw {
    fn setup<'a>(&'a self, problems: &[Arc<Problem>]) -> Result<Box<dyn SearchState + 'a>, String> {
        if problems.iter().any(|p| p.number_variables() == 0) {
            return Err("a problem without variables cannot be searched".to_string());
        }
        Ok(Box::new(RedrawState { architecture: self }))
    }
}

impl SearchState for RedrawState<'_> {
    fn step(
        &mut self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    ) {
        let destroyed = destroy_rows(
            self.architecture.destroy.as_ref(),
            problems,
            assignments,
            pending,
            rng,
        );
        for ((row, problem), freed) in assignments.iter_mut().zip(problems).zip(destroyed) {
            let fresh = random_assignment(problem);
            for variable in freed {
                row[variable] = fresh[variable];
            }
        }
    }
}

// [docs:end architecture]
fn chain(length: usize) -> Arc<Problem> {
    let mut problem = Problem::default();
    let vars = problem.add_variables(length, vec![0, 1, 2]);
    all_different(&mut problem, vars);
    Arc::new(problem)
}

// [docs:start run]
#[test]
fn a_search_from_another_crate_runs_in_the_engine() {
    let engine = LocalSearch::new(Box::new(Redraw {
        destroy: DestroyKind::Random.build(0.5),
    }));
    let problems = vec![chain(3), chain(3)];
    let budget = Budget {
        iteration_limit: 5_000,
        ..Budget::default()
    };
    let solutions = run_batched(&engine, &problems, None, budget, 0).unwrap();
    assert_eq!(solutions.len(), 2);
    for solution in &solutions {
        assert_eq!(solution.status(), Status::Satisfiable);
    }
}

// [docs:end run]
#[test]
fn the_error_of_setup_reaches_the_caller() {
    let engine = LocalSearch::new(Box::new(Redraw {
        destroy: DestroyKind::Random.build(0.5),
    }));
    let empty = Arc::new(Problem::default());
    let error = run_batched(&engine, &[empty], None, Budget::default(), 0).unwrap_err();
    assert!(error.contains("without variables"), "{error}");
}
