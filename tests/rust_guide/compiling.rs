//! Chapter "Compiling a model".

// [docs:start latin]
use std::sync::Arc;

use aicad::constraints::all_different;
use aicad::mdd::Mdd;
use aicad::modelling::{Problem, VariableIndex, equal};
use aicad::sampling::{sample_solution, uniform_weights};

/// A latin square of order `n`: every row and every column holds each of `0..n` once.
pub fn latin_square(n: usize) -> (Problem, Vec<Vec<VariableIndex>>) {
    let mut problem = Problem::default();
    let grid: Vec<Vec<VariableIndex>> = (0..n)
        .map(|_| problem.add_variables(n, (0..n as isize).collect()))
        .collect();
    for row in &grid {
        all_different(&mut problem, row.clone());
    }
    for j in 0..n {
        all_different(&mut problem, grid.iter().map(|row| row[j]).collect());
    }
    (problem, grid)
}
// [docs:end latin]

// [docs:start helpers]
/// The number of nodes of every layer.
pub fn widths(mdd: &Mdd) -> Vec<usize> {
    (0..mdd.number_layers())
        .map(|layer| mdd.number_nodes_in_layer(layer))
        .collect()
}

/// A path drawn with every solution equally likely, as an assignment: one value per variable of
/// the problem, in the order of their indices.
pub fn sample_assignment(mdd: &Mdd) -> Vec<isize> {
    let path = sample_solution(mdd, &uniform_weights(mdd));
    let (values, variables) = mdd.solution_from_path(&path);
    // A path lists its values in the order of the layers: put them back in variable order.
    let mut assignment = vec![0; values.len()];
    for (value, variable) in values.into_iter().zip(variables) {
        assignment[variable.0] = value;
    }
    assignment
}
// [docs:end helpers]

#[test]
fn an_exact_diagram_counts_and_samples_its_solutions() {
    // [docs:start exact]
    use aicad::mdd::Compiler;
    use aicad::mdd::heuristics::CompilationHeuristic;

    let (problem, _grid) = latin_square(4);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let problem = Arc::new(problem);

    // A compiler is made for a problem, the constraints to compile and the heuristics that drive
    // the compilation. `refine(usize::MAX)` lifts the width limit: the diagram is exact.
    let mut compiler = Compiler::new(
        problem.clone(),
        CompilationHeuristic::default(),
        &constraints,
    );
    compiler.refine(usize::MAX, usize::MAX);
    let mdd = compiler.mdd();
    println!("solutions: {}", mdd.count_solutions()); // 576 latin squares of order 4
    // [docs:end exact]
    assert_eq!(mdd.count_solutions().to_string(), "576");
    assert_eq!(mdd.number_layers(), 17);

    // [docs:start sample]
    aicad::set_seed(0);
    for _ in 0..3 {
        let values = sample_assignment(mdd);
        assert!(problem.is_solution(&values));
        println!("{values:?}");
    }
    let (values, _variables) = mdd.get_solution().expect("the diagram has a solution");
    println!("one solution: {values:?}");
    // [docs:end sample]
    assert_eq!(values.len(), 16);
}

#[test]
fn the_heuristics_change_the_size_not_the_solutions() {
    // [docs:start heuristics]
    use aicad::mdd::Compiler;
    use aicad::mdd::heuristics::{
        CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
    };

    let (problem, grid) = latin_square(4);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let problem = Arc::new(problem);

    // These are the defaults, written out. `MinDomMaxLinked` starts with the variable of smallest
    // domain and then takes the one most linked to those already placed.
    let auto = CompilationHeuristic {
        ordering: OrderingHeuristic::MinDomMaxLinked,
        merge: MergeHeuristic::StateSimilarity,
        selection: SelectHeuristic::Greedy,
    };
    let mut compiler = Compiler::new(problem.clone(), auto, &constraints);
    compiler.refine(usize::MAX, usize::MAX);
    println!("nodes per layer, auto order: {:?}", widths(compiler.mdd()));

    // Any other order is a list of variable indices, from the first layer to the last. It changes
    // the size of the diagram, not the solutions it holds: here the grid is filled row by row.
    let row_by_row: Vec<usize> = grid.iter().flatten().map(|variable| variable.0).collect();
    let by_rows = CompilationHeuristic {
        ordering: OrderingHeuristic::Custom(row_by_row),
        ..CompilationHeuristic::default()
    };
    let mut other = Compiler::new(problem, by_rows, &constraints);
    other.refine(usize::MAX, usize::MAX);
    println!("nodes per layer, row by row: {:?}", widths(other.mdd()));
    // [docs:end heuristics]
    assert_eq!(
        other.mdd().count_solutions(),
        compiler.mdd().count_solutions()
    );
    assert_ne!(widths(other.mdd()), widths(compiler.mdd()));
}

#[test]
fn a_compiler_works_on_the_problem_it_was_given() {
    use aicad::mdd::Compiler;
    use aicad::mdd::heuristics::CompilationHeuristic;

    // [docs:start clue]
    // The compiler owns an `Arc<Problem>`, so a problem is final once compiled: to add a clue,
    // build the problem again (or before wrapping it).
    let (mut problem, grid) = latin_square(4);
    equal(&mut problem, grid[0][0], 2);
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let mut compiler = Compiler::new(
        Arc::new(problem),
        CompilationHeuristic::default(),
        &constraints,
    );
    compiler.refine(usize::MAX, usize::MAX);
    println!("with grid[0][0] = 2: {}", compiler.mdd().count_solutions()); // 576 / 4
    // [docs:end clue]
    assert_eq!(compiler.mdd().count_solutions().to_string(), "144");
}

#[test]
fn contradictory_constraints_are_detected_while_compiling() {
    use aicad::mdd::Compiler;
    use aicad::mdd::heuristics::CompilationHeuristic;

    // [docs:start unsat]
    let (mut problem, grid) = latin_square(4);
    equal(&mut problem, grid[0][0], 2);
    equal(&mut problem, grid[0][1], 2); // the same value twice in the first row
    let constraints: Vec<_> = problem.iter_constraints().collect();
    let mut compiler = Compiler::new(
        Arc::new(problem),
        CompilationHeuristic::default(),
        &constraints,
    );
    compiler.refine(usize::MAX, usize::MAX);
    let impossible = compiler.mdd();
    println!(
        "unsatisfiable: {} - solutions: {} - {:?}",
        impossible.is_unsat(),
        impossible.count_solutions(),
        impossible.get_solution()
    );
    // [docs:end unsat]
    assert!(impossible.is_unsat());
    assert_eq!(impossible.count_solutions().to_string(), "0");
    assert!(impossible.get_solution().is_none());
}

#[test]
fn a_diagram_can_be_drawn_with_graphviz() {
    use aicad::mdd::Compiler;
    use aicad::mdd::heuristics::CompilationHeuristic;

    // [docs:start dot]
    let mut small = Problem::default();
    let x = small.add_variables(3, vec![0, 1, 2]);
    let constraint = all_different(&mut small, x);
    let mut compiler = Compiler::new(
        Arc::new(small),
        CompilationHeuristic::default(),
        &[constraint],
    );
    compiler.refine(usize::MAX, usize::MAX);
    let dot = compiler.mdd().as_graphviz(); // `dot -Tpdf`
    println!("{}", &dot[..60]);
    // [docs:end dot]
    assert!(dot.starts_with("digraph"));
}
