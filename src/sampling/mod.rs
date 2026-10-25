mod bp;

pub(crate) use bp::belief_propagation;

use crate::mdd::{CompiledConstraint, Mdd, NodeIndex};
use crate::modelling::ValueIndex;

use rand::RngExt;

const LOG_ZERO: f64 = -745.0;

fn safe_ln(p: f64) -> f64 {
    if p > 0.0 { p.ln() } else { LOG_ZERO }
}

/// How a variable's combined distribution turns into a value: `sample_categorical` or `argmax`.
#[derive(Clone, Copy, Debug)]
pub enum DecodeMode {
    /// Draw a value proportionally to the combined distribution
    Sample,
    /// Take the most likely value
    Greedy,
}

/// Normalises a categorical probability distribution if at least one element has non-zero weight,
/// otherwise returns a uniform distribution
fn normalize_or_uniform(mut weights: Vec<f64>, domain_size: usize) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / domain_size as f64; domain_size];
    }
    for w in &mut weights {
        *w /= total;
    }
    weights
}

/// For each global variable, every `(mdd_index, layer)` pair where that MDD has this variable in
/// its scope, at that layer. Used by `bp::belief_propagation` (multi-round marginal aggregation)
/// to know "which MDDs does this variable belong to, and at what layer in each".
///
/// `num_vars` is passed explicitly rather than read off a problem since a `CompiledConstraint`
/// shares its diagram with other constraints, and the diagram's problem is the template's (see the
/// arena module doc).
fn build_var_to_mdds(mdds: &[CompiledConstraint], num_vars: usize) -> Vec<Vec<(usize, usize)>> {
    let mut var_to_mdds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_vars];
    for (mdd_index, mdd) in mdds.iter().enumerate() {
        for layer in 0..mdd.number_decisions() {
            let variable = mdd.decision_at_layer(layer);
            var_to_mdds[variable.0].push((mdd_index, layer));
        }
    }
    var_to_mdds
}

fn log_combine_and_normalize(log_combined: Vec<f64>) -> Vec<f64> {
    let max_log = log_combined
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut combined: Vec<f64> = log_combined.iter().map(|&l| (l - max_log).exp()).collect();
    let total: f64 = combined.iter().sum();
    for c in &mut combined {
        *c /= total;
    }
    combined
}

pub(crate) fn argmax(weights: &[f64]) -> usize {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("weights should never be NaN"))
        .map(|(index, _)| index)
        .expect("weights should not be empty")
}

pub(crate) fn sample_categorical(weights: &[f64]) -> usize {
    crate::utils::with_rng(|rng| {
        let total: f64 = weights.iter().sum();
        let mut target = rng.random_range(0.0..total);
        for (index, &w) in weights.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                return index;
            }
        }
        weights.len() - 1
    })
}

/// The probabilities of the variables of the problem of `mdd`, as weights for [`sample_solution`]:
/// `result[layer][value]` for the variable branched on at `layer`.
pub fn problem_weights(mdd: &Mdd) -> Vec<Vec<f64>> {
    (0..mdd.number_layers() - 1)
        .map(|layer| {
            let variable = mdd.decision_at_layer(layer);
            (0..mdd.problem()[variable].domain_size())
                .map(|value| mdd.problem()[variable].probability(ValueIndex(value)))
                .collect()
        })
        .collect()
}

/// Sample a root-to-sink path in the MDD according to the given weights. Returns the path of
/// labels followed.
///
/// `weights[layer][value]` is the weight of the value at index `value` of the variable branched
/// on at `layer`, as for [`log_wmc`](crate::mdd::wmc::log_wmc): one row per layer but the sink's,
/// and a weight for every value an edge of the layer can carry. At each node, the weights of the
/// values of its edges are added up, a number is drawn uniformly in that total, and the edge whose
/// share of the total contains it is taken. The weights are not renormalised by what lies below a
/// node, so the draw follows the weights restricted to the values the diagram still allows, and
/// not the distribution of the solutions conditioned on the constraints (that is what the
/// weighted model count and its gradient give). The weights can come from anywhere: those of the
/// variables of the problem, the output of a network, a belief.
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

    /// `x` takes 0 with probability 0.9 and 1 with 0.1, and `y` must differ from `x`.
    fn skewed() -> Mdd {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], Some(vec![0.9, 0.1]));
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        compile(problem)
    }

    #[test]
    fn every_sample_of_an_exact_diagram_is_a_solution() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars);
        let mdd = compile(problem);
        let weights = vec![vec![1.0; 3]; 3];
        for _ in 0..200 {
            let path = sample_solution(&mdd, &weights);
            let mut solution = mdd.solution_from_path(&path);
            solution.sort();
            assert_eq!(solution, vec![0, 1, 2]);
        }
    }

    #[test]
    fn values_are_drawn_in_proportion_to_their_weight() {
        let mdd = skewed();
        let weights = problem_weights(&mdd);
        assert_eq!(weights.len(), 2);
        let draws = 4000;
        let zeros = (0..draws)
            .filter(|_| mdd.solution_from_path(&sample_solution(&mdd, &weights))[0] == 0)
            .count();
        let frequency = zeros as f64 / draws as f64;
        // 0.9 with a standard deviation of about 0.005: always taking the last edge would give 0.
        assert!((frequency - 0.9).abs() < 0.04, "frequency {frequency}");
    }

    #[test]
    fn the_weights_are_those_of_the_caller_not_those_of_the_problem() {
        let mdd = skewed();
        let (first, second) = (mdd.decision_at_layer(0), mdd.decision_at_layer(1));
        // Whichever variable comes first, force 1 on it: the other one is then forced to 0.
        let weights = vec![vec![0.0, 1.0], vec![0.5, 0.5]];
        for _ in 0..100 {
            let solution = mdd.solution_from_path(&sample_solution(&mdd, &weights));
            assert_eq!((solution[first.0], solution[second.0]), (1, 0));
        }
    }

    #[test]
    #[should_panic(expected = "cannot sample a solution from an unsatisfiable diagram")]
    fn an_unsatisfiable_diagram_cannot_be_sampled() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
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

    #[test]
    fn sample_categorical_respects_zero_weight_never_selected() {
        // Not a statistical test (no fixed seed guarantee here) -- just checks that a value with
        // zero weight is structurally unreachable given how `sample_categorical` walks the
        // cumulative distribution, run enough times to be confident about it.
        let weights = vec![0.0, 1.0, 0.0];
        for _ in 0..200 {
            assert_eq!(sample_categorical(&weights), 1);
        }
    }

    #[test]
    fn argmax_picks_the_largest_weight() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.9, 0.05, 0.05]), 0);
    }
}
