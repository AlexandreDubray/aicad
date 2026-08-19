use std::sync::{Arc, Mutex};

use burn::prelude::ElementConversion;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use rayon::prelude::*;

use crate::learning::consformer::MddCompilationConfig;
use crate::mdd::Mdd;
use crate::modelling::{ConstraintIndex, Problem, ValueIndex, VariableIndex};
use crate::nls::decode::DecodingOperator;
use crate::sampling::{DecodeMode, GibbsSampler};

struct Cache {
    problems: Vec<Arc<Problem>>,
    mdds: Vec<Vec<Mdd>>,
}

pub struct MddGibbsDecoding {
    compilation: MddCompilationConfig,
    domain_size: usize,
    rounds: usize,
    mode: DecodeMode,
    gibbs_cleanup: bool,
    cache: Mutex<Option<Cache>>,
}

impl MddGibbsDecoding {
    /// `gibbs_cleanup` toggles the second (clamped, sequential single-variable) stage of
    /// `GibbsSampler::resample_block` -- when `false`, the destroyed block is only ever resampled
    /// from `GibbsSampler::combined_marginal` (each variable independently), never from
    /// `combined_conditional`.
    pub fn new(
        compilation: MddCompilationConfig,
        domain_size: usize,
        rounds: usize,
        mode: DecodeMode,
        gibbs_cleanup: bool,
    ) -> Self {
        Self {
            compilation,
            domain_size,
            rounds,
            mode,
            gibbs_cleanup,
            cache: Mutex::new(None),
        }
    }

    fn with_mdds<R>(&self, problems: &[Arc<Problem>], f: impl FnOnce(&[Vec<Mdd>]) -> R) -> R {
        let mut cache = self.cache.lock().unwrap();
        let up_to_date = matches!(
            &*cache,
            Some(c) if c.problems.len() == problems.len()
                && c.problems.iter().zip(problems).all(|(a, b)| Arc::ptr_eq(a, b))
        );

        if !up_to_date {
            let mdds: Vec<Vec<Mdd>> = problems
                .par_iter()
                .map(|problem| compile_problem_mdds(problem, &self.compilation))
                .collect();
            *cache = Some(Cache {
                problems: problems.to_vec(),
                mdds,
            });
        }

        f(&cache.as_ref().unwrap().mdds)
    }
}

/// One exact MDD per constraint of `problem`, in `problem.iter_constraints()` order.
fn compile_problem_mdds(problem: &Arc<Problem>, compilation: &MddCompilationConfig) -> Vec<Mdd> {
    let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
    constraints
        .into_par_iter()
        .map(|constraint| {
            let mut mdd = Mdd::new(
                Arc::clone(problem),
                compilation.ordering.clone(),
                compilation.merge,
                compilation.select,
                &[constraint],
            );
            mdd.refine(usize::MAX);

            if mdd.is_unsat() {
                log::warn!(
                    "constraint `{}` (index {}) is unsatisfiable given its own scope's domains -- \
                     its compiled MDD has no accepting path, so MddGibbsDecoding will always \
                     abstain (uniform conditional) for it. This usually means a fixed/hint value \
                     already violates this constraint.",
                    problem[constraint].name(),
                    constraint.0,
                );
            }

            mdd
        })
        .collect()
}

fn value_to_index(problem: &Problem, variable: VariableIndex, value: isize) -> ValueIndex {
    let index = problem[variable]
        .iter_domain()
        .position(|v| v == value)
        .unwrap_or_else(|| {
            panic!(
                "value {value} is not in variable {}'s domain -- current assignment is \
                 inconsistent with the problem it's paired with",
                variable.0
            )
        });
    ValueIndex(index)
}

impl<B: Backend> DecodingOperator<B> for MddGibbsDecoding {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
        population_size: usize,
    ) -> Tensor<B, 2, Int> {
        let device = current.device();
        let [total_rows, n, domain_size] = logits.dims();
        assert_eq!(
            domain_size, self.domain_size,
            "MddGibbsDecoding was built with domain_size={}, but logits carry domain_size={}",
            self.domain_size, domain_size
        );

        let probs_flat: Vec<f32> = softmax(logits, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");
        let mask_flat: Vec<bool> = destroy_mask
            .into_data()
            .to_vec::<B::IntElem>()
            .expect("destroy mask should be int-convertible")
            .into_iter()
            .map(|v| v.elem::<i64>() != 0)
            .collect();
        let current_flat: Vec<i64> = current
            .into_data()
            .to_vec::<B::IntElem>()
            .expect("assignment tensor should be integer")
            .into_iter()
            .map(|v| v.elem::<i64>())
            .collect();

        let flat: Vec<i64> = self.with_mdds(problems, |mdds| {
            let samplers: Vec<GibbsSampler> = mdds.iter().map(|m| GibbsSampler::new(m)).collect();

            let new_rows: Vec<Vec<i64>> = (0..total_rows)
                .into_par_iter()
                .map(|row| {
                    let problem_idx = row / population_size;
                    let problem = &problems[problem_idx];
                    let sampler = &samplers[problem_idx];

                    let probs: Vec<Vec<f64>> = (0..n)
                        .map(|v| {
                            let variable = VariableIndex(v);
                            let ds = problem[variable].domain_size();
                            (0..ds)
                                .map(|d| {
                                    let value = problem[variable].value(ValueIndex(d));
                                    assert!(
                                        value >= 0 && (value as usize) < domain_size,
                                        "variable {v}'s domain value {value} is out of the \
                                         network's [0, {domain_size}) range -- MddGibbsDecoding \
                                         was built with a domain_size that doesn't match this \
                                         problem's own domains"
                                    );
                                    let offset =
                                        row * n * domain_size + v * domain_size + value as usize;
                                    probs_flat[offset] as f64
                                })
                                .collect()
                        })
                        .collect();

                    let mut assignment: Vec<ValueIndex> = (0..n)
                        .map(|v| {
                            let variable = VariableIndex(v);
                            let value = current_flat[row * n + v] as isize;
                            value_to_index(problem, variable, value)
                        })
                        .collect();

                    let order: Vec<VariableIndex> = (0..n)
                        .filter(|&v| mask_flat[row * n + v])
                        .map(VariableIndex)
                        .collect();

                    let cleanup_rounds = if self.gibbs_cleanup { self.rounds } else { 0 };
                    sampler.resample_block(
                        &probs,
                        &mut assignment,
                        &order,
                        self.mode,
                        cleanup_rounds,
                    );

                    (0..n)
                        .map(|v| problem[VariableIndex(v)].value(assignment[v]) as i64)
                        .collect()
                })
                .collect();

            new_rows.into_iter().flatten().collect()
        });

        Tensor::<B, 1, Int>::from_data(flat.as_slice(), &device).reshape([total_rows, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
    use crate::modelling::{gcc, not_equals};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Distribution, TensorData};

    fn compilation() -> MddCompilationConfig {
        MddCompilationConfig {
            ordering: OrderingHeuristic::MinDomMaxLinked,
            merge: MergeHeuristic::LessRelaxed,
            select: SelectHeuristic::Greedy,
        }
    }

    #[test]
    fn decode_only_changes_destroyed_positions_and_stays_in_domain() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let domain_size = 3;

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        gcc(&mut problem, vec![x, y], vec![]);
        let problem = Arc::new(problem);
        let problems = vec![problem.clone()];
        let n = problem.number_variables();

        let op = MddGibbsDecoding::new(compilation(), domain_size, 3, DecodeMode::Greedy, true);

        let logits = Tensor::<B, 1>::random(
            [1 * n * domain_size],
            Distribution::Uniform(0.0, 1.0),
            &device,
        )
        .reshape([1, n, domain_size]);
        // Only variable x is destroyed.
        let mask_data = [1i64, 0i64];
        let destroy_mask =
            Tensor::<B, 2, Int>::from_data(TensorData::new(mask_data.to_vec(), [1, n]), &device);
        let current =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i64, 1i64], [1, n]), &device);

        let next = DecodingOperator::<B>::decode(&op, logits, destroy_mask, current, &problems, 1);
        let next_data: Vec<i64> = next.into_data().to_vec::<i64>().unwrap();

        // y (not destroyed) must keep its value.
        assert_eq!(next_data[1], 1);
        // x (destroyed) must land in its own domain.
        assert!((0..3).contains(&next_data[0]));
    }

    #[test]
    fn greedy_decode_recovers_a_confident_and_correct_network_prediction_on_a_single_free_variable()
    {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let domain_size = 3;

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![1], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);
        gcc(&mut problem, vec![x, y, z], vec![]);
        let problem = Arc::new(problem);
        let problems = vec![problem.clone()];
        let n = problem.number_variables();

        let op = MddGibbsDecoding::new(compilation(), domain_size, 10, DecodeMode::Greedy, true);

        let mut logit_data = vec![0.0f32; n * domain_size];
        logit_data[z.0 * domain_size + 2] = 10.0;
        let logits =
            Tensor::<B, 1>::from_data(TensorData::new(logit_data, [n * domain_size]), &device)
                .reshape([1, n, domain_size]);

        let mask_data = [0i64, 0i64, 1i64];
        let destroy_mask =
            Tensor::<B, 2, Int>::from_data(TensorData::new(mask_data.to_vec(), [1, n]), &device);
        let current = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1i64, 0i64], [1, n]),
            &device,
        );

        let next = DecodingOperator::<B>::decode(&op, logits, destroy_mask, current, &problems, 1);
        let next_data: Vec<i64> = next.into_data().to_vec::<i64>().unwrap();

        assert_eq!(next_data, vec![0, 1, 2]);
    }

    #[test]
    fn gibbs_cleanup_recovers_feasibility_from_a_colliding_start() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let domain_size = 3;

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);
        let problem = Arc::new(problem);
        let problems = vec![problem.clone()];
        let n = problem.number_variables();

        let op = MddGibbsDecoding::new(compilation(), domain_size, 10, DecodeMode::Greedy, true);

        let logits = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.0f32; n * domain_size], [n * domain_size]),
            &device,
        )
        .reshape([1, n, domain_size]);
        let destroy_mask =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i64; n], [1, n]), &device);
        // Every variable starts pointing at the same (globally infeasible) domain position.
        let current = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0i64, 0i64, 1i64], [1, n]),
            &device,
        );

        let next = DecodingOperator::<B>::decode(&op, logits, destroy_mask, current, &problems, 1);
        let next_data: Vec<i64> = next.into_data().to_vec::<i64>().unwrap();

        assert_ne!(next_data[x.0], next_data[y.0], "{next_data:?}");
        assert_ne!(next_data[y.0], next_data[z.0], "{next_data:?}");
        assert_ne!(next_data[x.0], next_data[z.0], "{next_data:?}");
    }
}
