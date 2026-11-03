//! Chapter "Relaxed diagrams and refinement".

use std::sync::Arc;

use aicad::constraints::all_different;
use aicad::mdd::Compiler;
use aicad::mdd::heuristics::{CompilationHeuristic, MergeHeuristic, OrderingHeuristic};
use aicad::modelling::{ConstraintIndex, Problem};

use crate::compiling::{latin_square, sample_assignment, widths};

#[test]
fn a_relaxed_diagram_has_more_paths_than_solutions() {
    // [docs:start relaxed]
    let (problem, _grid) = latin_square(4);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let problem = Arc::new(problem);

    let mut exact = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    exact.refine(usize::MAX);
    let solutions = exact.mdd().count_solutions();

    // At most 2 nodes per layer: every solution is still a path, and many paths are not solutions.
    let mut compiler = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    compiler.refine(2);
    println!(
        "width 2: {} paths for {solutions} solutions",
        compiler.mdd().count_solutions()
    );
    // [docs:end relaxed]
    assert!(compiler.mdd().count_solutions() >= solutions);

    // [docs:start relaxed_path]
    // A path of a relaxed diagram is not necessarily a solution: check it against the problem.
    let mut relaxed = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    relaxed.refine(256);
    println!("width 256: {} paths", relaxed.mdd().count_solutions());
    let path = sample_assignment(relaxed.mdd());
    println!(
        "a random path is a solution: {}",
        problem.is_solution(&path)
    );
    // [docs:end relaxed_path]
}

#[test]
fn refinement_tightens_a_diagram_in_place() {
    let (problem, _grid) = latin_square(4);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let problem = Arc::new(problem);
    let mut exact = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    exact.refine(usize::MAX);
    let solutions = exact.mdd().count_solutions();

    // [docs:start refine]
    let mut compiler = Compiler::new(problem, CompilationHeuristic::default(), &constraints);
    compiler.refine(2);
    let mut previous = compiler.mdd().count_solutions();
    for width in [4, 8, 32, usize::MAX] {
        compiler.refine(width);
        let paths = compiler.mdd().count_solutions();
        let widest = widths(compiler.mdd()).into_iter().max().unwrap();
        println!("width {width}: {paths} paths, widest layer {widest}");
        assert!(paths <= previous); // the number of paths only goes down...
        assert!(paths >= solutions); // ...and never below the number of solutions
        previous = paths;
    }
    assert_eq!(compiler.mdd().count_solutions(), solutions);
    // [docs:end refine]

    // [docs:start freeze]
    // A compiler keeps what refining needs: the constraints, their properties and the heuristics,
    // which is often most of the memory. `into_mdd` consumes it and keeps the diagram, which can
    // still be counted, sampled and drawn. Refining it any more is not possible, and the types say
    // so: only a `Compiler` has `refine`.
    let mdd = compiler.into_mdd();
    println!("frozen, still {} paths", mdd.count_solutions());
    // [docs:end freeze]
    assert_eq!(mdd.count_solutions(), solutions);
}

#[test]
fn a_subset_of_the_constraints_can_be_compiled() {
    // [docs:start subset]
    let (problem, _grid) = latin_square(4);
    // The first four constraints of `latin_square` are the rows.
    let rows: Vec<ConstraintIndex> = problem.iter_constraints().take(4).collect();
    let mut compiler = Compiler::new(Arc::new(problem), CompilationHeuristic::default(), &rows);
    compiler.refine(usize::MAX);
    let mdd = compiler.mdd();
    // A variable in none of the compiled constraints has no layer.
    let scope: Vec<_> = (0..mdd.number_decisions())
        .map(|layer| mdd.decision_at_layer(layer))
        .collect();
    println!(
        "rows alone: {} paths over {} variables",
        mdd.count_solutions(),
        scope.len()
    );
    // [docs:end subset]
    assert_eq!(mdd.count_solutions().to_string(), "331776");
    assert_eq!(scope.len(), 16);
}

#[test]
fn the_merge_heuristic_changes_how_tight_a_relaxation_is() {
    // [docs:start merge]
    let (problem, grid) = latin_square(4);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let problem = Arc::new(problem);
    let column_major: Vec<usize> = (0..4)
        .flat_map(|j| (0..4).map(move |i| (i, j)))
        .map(|(i, j)| grid[i][j].0)
        .collect();

    // `MostLikely` reads the probability of every value of every variable: here, all equal.
    let uniform = vec![vec![0.25; 4]; problem.number_variables()];
    for (name, merge) in [
        ("LessRelaxed", MergeHeuristic::LessRelaxed),
        ("MostLikely", MergeHeuristic::MostLikely(uniform)),
        ("StateSimilarity", MergeHeuristic::StateSimilarity),
    ] {
        let heuristic = CompilationHeuristic {
            ordering: OrderingHeuristic::Custom(column_major.clone()),
            merge,
            ..CompilationHeuristic::default()
        };
        let mut compiler = Compiler::new(problem.clone(), heuristic, &constraints);
        compiler.refine(4);
        println!("{name}: {} paths", compiler.mdd().count_solutions());
    }
    // [docs:end merge]
}

// [docs:start pigeons]
/// Every pigeon in a hole, no two in the same hole: feasible iff `pigeons <= holes`.
fn pigeons(pigeons: usize, holes: usize) -> (Arc<Problem>, Vec<ConstraintIndex>) {
    let mut problem = Problem::default();
    let hole = problem.add_variables(pigeons, (0..holes as isize).collect());
    let constraint = all_different(&mut problem, hole);
    (Arc::new(problem), vec![constraint])
}
// [docs:end pigeons]

#[test]
fn a_relaxed_diagram_decides_less_than_an_exact_one() {
    // [docs:start decide]
    // An exact diagram decides a problem: a solution, or a proof that there is none.
    let (problem, constraints) = pigeons(5, 5);
    for width in [1, 4, usize::MAX] {
        let mut compiler = Compiler::new(
            problem.clone(),
            CompilationHeuristic::default(),
            &constraints,
        );
        compiler.refine(width);
        let mdd = compiler.mdd();
        // `get_solution` returns a solution only if the diagram holds a path of exact nodes, so
        // `None` proves nothing, unless the diagram is unsatisfiable.
        println!(
            "5 pigeons, 5 holes, width {width}: unsat={}, solution={:?}",
            mdd.is_unsat(),
            mdd.get_solution()
        );
    }

    // Even a relaxed diagram can prove that a problem has no solution.
    let (problem, constraints) = pigeons(5, 4);
    let mut compiler = Compiler::new(problem, CompilationHeuristic::default(), &constraints);
    compiler.refine(1);
    println!(
        "5 pigeons, 4 holes, width 1: unsat = {}",
        compiler.mdd().is_unsat()
    );
    // [docs:end decide]
    assert!(compiler.mdd().is_unsat());

    let (problem, constraints) = pigeons(5, 5);
    let mut narrow = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    narrow.refine(1);
    assert!(!narrow.mdd().is_unsat());
    assert!(narrow.mdd().get_solution().is_none());
    let mut exact = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    exact.refine(usize::MAX);
    let (values, _) = exact.mdd().get_solution().unwrap();
    assert_eq!(values.len(), 5);
}
