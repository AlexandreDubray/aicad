use std::collections::HashMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::learning::HasProblems;

#[derive(Default)]
struct ConstraintStats {
    count: usize,
    sum_rate: f64,
    sum_rate_sq: f64,
}

impl ConstraintStats {
    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_rate / self.count as f64
        }
    }

    fn stddev(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            let mean = self.mean();
            let variance = self.sum_rate_sq / self.count as f64 - mean * mean;
            variance.max(0.0).sqrt()
        }
    }
}

/// For each constraint, get a report on how it is satisfied by a neural-network prediction. This
/// is used to monitor training, regardless of the loss function.
/// The intuition is that, if given enough time, the neural network should learn to satisfy
/// constraints. We group the stat per constraint.
pub struct SatisfactionReport {
    by_constraint: HashMap<&'static str, ConstraintStats>,
}

impl SatisfactionReport {
    /// Builds a report for the current batch
    pub fn build<B: Backend, Ba: HasProblems<B>>(logits: Tensor<B, 3>, batch: &Ba) -> Self {
        let problems = batch.problems();
        let batch_size = problems.len();

        // Arg-max tensor of shape (batch_size, number_var, domain_size) over the last dimension
        // result in (batch_size, number_var, 1). Squeeze reduce all dimensions of size 1
        let assignment: Tensor<B, 2, Int> = logits.argmax(2).squeeze();
        // Get back the assignments as a flat tensor
        let assignment: Vec<i64> = assignment
            .into_data()
            .to_vec::<i64>()
            .expect("decoded assignment should be an integer tensor");

        let mut by_constraint: HashMap<&'static str, ConstraintStats> = HashMap::new();

        for (i, problem) in problems.iter().enumerate().take(batch_size) {
            let number_vars = problem.number_variables();
            let start = i * number_vars;
            // Get the assignment from this sample
            let sample_assignment: Vec<isize> = assignment[start..start + number_vars]
                .iter()
                .map(|&v| v as isize)
                .collect();

            for constraint in problem.iter_constraints() {
                let c = &problem[constraint];
                let satisfied = if c.is_satisfied(&sample_assignment) {
                    1.0
                } else {
                    0.0
                };

                let entry = by_constraint.entry(c.name()).or_default();
                entry.count += 1;
                entry.sum_rate += satisfied;
                entry.sum_rate_sq += satisfied * satisfied;
            }
        }

        SatisfactionReport { by_constraint }
    }

    /// Overall satisfaction rate across every constraint instance seen,
    /// pooled (not averaged-per-type -- a constraint type with 10x more
    /// instances counts 10x more), for use as a single model-selection score
    /// against a validation set.
    pub fn overall_rate(&self) -> f64 {
        let (total_count, total_rate) = self
            .by_constraint
            .values()
            .fold((0usize, 0.0), |(count, rate), stats| {
                (count + stats.count, rate + stats.sum_rate)
            });

        if total_count == 0 {
            0.0
        } else {
            total_rate / total_count as f64
        }
    }

    /// Merges the other satisfaction report into this one. Used for merging reports from multiple
    /// batch in a given epoch
    pub fn merge(&mut self, other: SatisfactionReport) {
        for (name, stats) in other.by_constraint {
            let entry = self.by_constraint.entry(name).or_default();
            entry.count += stats.count;
            entry.sum_rate += stats.sum_rate;
            entry.sum_rate_sq += stats.sum_rate_sq;
        }
    }

    pub fn print(&self, width: usize) {
        if self.by_constraint.is_empty() {
            println!("No constraints recorded.");
            return;
        }

        let mut rows: Vec<(&str, &ConstraintStats)> =
            self.by_constraint.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap());

        let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(4);

        println!(
            "{:<name_width$}  {:>8}  {:>8}  {:>8}",
            "TYPE", "AVG SAT", "STDDEV", "COUNT",
        );
        println!("{}", "-".repeat(name_width + width + 32));

        for (name, stats) in &rows {
            let mean = stats.mean();

            println!(
                "{:<name_width$}  {:>7.1}%  {:>7.1}%  {:>8}",
                name,
                mean * 100.0,
                stats.stddev() * 100.0,
                stats.count,
            );
        }
    }
}
