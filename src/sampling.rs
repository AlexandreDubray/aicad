//! Everything that draws or chooses from a distribution: values of a categorical distribution,
//! random assignments, and solutions sampled from a diagram.
//!
//! A categorical distribution is a slice of non-negative weights, one per value, that need not be
//! normalised. [`DecodeMode`] says how one becomes a value, [`argmax`] for the largest weight or
//! [`sample_categorical`] for a draw proportional to the weights. A distribution is always given
//! by the caller: a problem carries none, so a belief, a network's output or a uniform guess are
//! equally fine. [`sample_solution`] draws a whole path from a diagram with weights per layer
//! and value, and [`uniform_weights`] gives the weights that make every value of a layer equally
//! likely.
//!
//! # How samplign works
//!
//! A draw adds the weights up, picks a number uniformly in the total and returns the value whose
//! share of the total contains it. [`sample_solution`] does so at each node of the diagram on the
//! weights of the values of its edges, from the root to the sink.
//!
//! # Design notes
//!
//! - **Two sources of randomness.** [`sample_categorical`] draws from the process-wide generator
//!   (see `crate::utils`), so a run follows the seed given with `set_seed`;
//!   [`sample_categorical_with`] takes the generator of the caller, for work done in parallel,
//!   where the process-wide one would make the result depend on the thread schedule.

use rand::{Rng, RngExt};

use crate::mdd::{Mdd, NodeIndex};
use crate::modelling::{Problem, ValueIndex};

/// How a categorical distribution given as weights turns into a value: [`sample_categorical`] or
/// [`argmax`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodeMode {
    /// Draw a value proportionally to its weight.
    Sample,
    /// Take the value of largest weight.
    Greedy,
}

/// The index of the largest weight; on ties, the last of the largest ones (that of `Iterator::max_by`).
///
/// # Panics
///
/// Panics if `weights` is empty or holds a NaN.
pub(crate) fn argmax(weights: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("weights should never be NaN"))
        .map(|(index, _)| index)
        .expect("weights should not be empty")
}

/// Draws an index with probability proportional to its weight, from the generator `rng`. The
/// weights need not be normalised.
///
/// # Panics
///
/// Panics unless the weights have a positive, finite sum.
pub(crate) fn sample_categorical_with<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> usize {
    let total: f64 = weights.iter().sum();
    assert!(
        total.is_finite() && total > 0.0,
        "cannot sample from weights {weights:?}: they must have a positive, finite sum"
    );
    let mut target = rng.random_range(0.0..total);
    for (index, &weight) in weights.iter().enumerate() {
        target -= weight;
        if target <= 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

/// [`sample_categorical_with`] the process-wide generator ([`with_rng`](crate::utils::with_rng)).
pub(crate) fn sample_categorical(weights: &[f64]) -> usize {
    crate::utils::with_rng(|rng| sample_categorical_with(weights, rng))
}

/// A random assignment of `problem`: every variable takes a value of its domain, drawn uniformly
/// from the process-wide generator, whatever the constraints say.
pub(crate) fn random_assignment(problem: &Problem) -> Vec<isize> {
    problem
        .iter_variables()
        .map(|variable| {
            let size = problem[variable].domain_size();
            let position = crate::utils::with_rng(|rng| rng.random_range(0..size));
            problem[variable].value(ValueIndex(position))
        })
        .collect()
}

/// Weights for [`sample_solution`] that make every value of every variable of `mdd` equally
/// likely: `result[layer][value]` is 1 for each value of the variable branched on at `layer`.
pub fn uniform_weights(mdd: &Mdd) -> Vec<Vec<f64>> {
    (0..mdd.number_layers() - 1)
        .map(|layer| vec![1.0; mdd.problem()[mdd.decision_at_layer(layer)].domain_size()])
        .collect()
}

/// Sample a root-to-sink path in the MDD according to the given weights. Returns the path of
/// labels followed.
///
/// At each node, the weights of the values of its edges are added up, a number is drawn uniformly
/// in that total, and the edge whose share of the total contains it is taken.
/// The weights are not renormalised by what lies below a node, so the draw follows the weights
/// restricted to the values the diagram still allows, and not the distribution of the solutions
/// conditioned on the constraints (that is what the weighted model count and its gradient give).
///
/// The path is a solution if `mdd` is exact; a relaxed diagram can yield an assignment that
/// violates a constraint. The random generator is the thread's own (see `crate::utils`).
///
/// # Panics
///
/// Panics if `mdd` is unsatisfiable, if `weights` does not have one row per layer but the sink's,
/// if a row is shorter than the largest value of its layer, or if every edge leaving a node on the
/// way has a weight of zero.
pub fn sample_solution(mdd: &Mdd, weights: &[Vec<f64>]) -> Vec<ValueIndex> {
    assert!(
        !mdd.is_unsat(),
        "cannot sample a solution from an unsatisfiable diagram"
    );
    assert_eq!(
        weights.len(),
        mdd.number_layers() - 1,
        "sample_solution needs one row of weights per layer but the sink's"
    );
    let mut path = Vec::with_capacity(weights.len());
    let mut node = mdd.root();
    while node != mdd.sink() {
        let NodeIndex(layer, _) = node;
        let edges: Vec<_> = mdd[node].iter_children().collect();
        let edge_weights: Vec<f64> = edges
            .iter()
            .map(|&edge| weights[layer][mdd[edge].assignment().0])
            .collect();
        assert!(
            edge_weights.iter().sum::<f64>() > 0.0,
            "every edge leaving node {node:?} has a weight of zero: nothing can be sampled"
        );
        let edge = edges[sample_categorical(&edge_weights)];
        path.push(mdd[edge].assignment());
        node = mdd[edge].to();
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, not_equals};
    use crate::mdd::heuristics::{
        CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
    };
    use crate::mdd::{Compiler, Mdd};
    use crate::modelling::Problem;
    use std::sync::Arc;

    /// The diagram of all the constraints of `problem`, refined without a width limit.
    fn compile(problem: Problem) -> Mdd {
        let problem = Arc::new(problem);
        let constraints: Vec<_> = problem.iter_constraints().collect();
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::MinDomMaxLinked,
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        compiler.refine(usize::MAX);
        compiler.into_mdd()
    }

    /// `x` and `y` take 0 or 1 and must differ.
    fn skewed() -> Mdd {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1]);
        not_equals(&mut problem, x, y);
        compile(problem)
    }

    #[test]
    fn every_sample_of_an_exact_diagram_is_a_solution() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        all_different(&mut problem, vars);
        let mdd = compile(problem);
        let weights = vec![vec![1.0; 3]; 3];
        for _ in 0..200 {
            let path = sample_solution(&mdd, &weights);
            let (mut solution, _) = mdd.solution_from_path(&path);
            solution.sort();
            assert_eq!(solution, vec![0, 1, 2]);
        }
    }

    #[test]
    fn values_are_drawn_in_proportion_to_their_weight() {
        let mdd = skewed();
        // `x` is 0 with weight 0.9 and 1 with weight 0.1; `y` is indifferent.
        let weights: Vec<Vec<f64>> = (0..2)
            .map(|layer| match mdd.decision_at_layer(layer).0 {
                0 => vec![0.9, 0.1],
                _ => vec![1.0, 1.0],
            })
            .collect();
        let draws = 4000;
        let zeros = (0..draws)
            .filter(|_| {
                let (values, variables) = mdd.solution_from_path(&sample_solution(&mdd, &weights));
                values[variables.iter().position(|v| v.0 == 0).unwrap()] == 0
            })
            .count();
        let frequency = zeros as f64 / draws as f64;
        // 0.9 with a standard deviation of about 0.005: always taking the last edge would give 0.
        assert!((frequency - 0.9).abs() < 0.04, "frequency {frequency}");
    }

    #[test]
    fn the_weights_are_those_of_the_caller() {
        let mdd = skewed();
        let (first, second) = (mdd.decision_at_layer(0), mdd.decision_at_layer(1));
        // Whichever variable comes first, force 1 on it: the other one is then forced to 0.
        let weights = vec![vec![0.0, 1.0], vec![0.5, 0.5]];
        for _ in 0..100 {
            let (solution, variables) = mdd.solution_from_path(&sample_solution(&mdd, &weights));
            assert_eq!(variables, vec![first, second]);
            assert_eq!(solution, vec![1, 0]);
        }
    }

    #[test]
    fn uniform_weights_have_a_row_per_layer_and_a_one_per_value() {
        let mdd = skewed();
        assert_eq!(uniform_weights(&mdd), vec![vec![1.0; 2]; 2]);
    }

    #[test]
    fn a_zero_weight_is_never_drawn() {
        for _ in 0..200 {
            assert_eq!(sample_categorical(&[0.0, 1.0, 0.0]), 1);
        }
    }

    #[test]
    fn a_draw_with_a_given_generator_follows_its_seed() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let weights = [0.2, 0.3, 0.5];
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..50)
                .map(|_| sample_categorical_with(&weights, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    #[should_panic(expected = "must have a positive, finite sum")]
    fn weights_summing_to_zero_cannot_be_drawn() {
        sample_categorical(&[0.0, 0.0]);
    }

    #[test]
    fn argmax_picks_the_largest_weight() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.9, 0.05, 0.05]), 0);
    }

    #[test]
    fn a_random_assignment_has_a_value_of_each_domain() {
        let mut problem = Problem::default();
        problem.add_variable(vec![4, 5]);
        problem.add_variable(vec![9]);
        problem.add_variable(vec![-1, 0, 1]);
        for _ in 0..100 {
            let assignment = random_assignment(&problem);
            assert!([4, 5].contains(&assignment[0]));
            assert_eq!(assignment[1], 9);
            assert!([-1, 0, 1].contains(&assignment[2]));
        }
    }

    #[test]
    #[should_panic(expected = "cannot sample a solution from an unsatisfiable diagram")]
    fn an_unsatisfiable_diagram_cannot_be_sampled() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0]);
        let y = problem.add_variable(vec![0]);
        not_equals(&mut problem, x, y);
        sample_solution(&compile(problem), &[]);
    }

    #[test]
    #[should_panic(expected = "one row of weights per layer")]
    fn the_weights_need_a_row_per_layer() {
        sample_solution(&skewed(), &[vec![1.0, 1.0]]);
    }

    #[test]
    #[should_panic(expected = "has a weight of zero")]
    fn weights_that_forbid_every_edge_cannot_be_sampled() {
        sample_solution(&skewed(), &[vec![0.0, 0.0], vec![1.0, 1.0]]);
    }
}
