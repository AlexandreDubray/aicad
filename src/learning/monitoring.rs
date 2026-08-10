use std::collections::HashMap;

use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::learning::Batch;

struct ConstraintStats {
    count: usize,
    sum_rate: f64,
    sum_rate_sq: f64,
    min: f64,
    max: f64,
}

impl Default for ConstraintStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum_rate: 0.0,
            sum_rate_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
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

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

pub struct SatisfactionReport {
    by_constraint: HashMap<&'static str, ConstraintStats>,
}

impl SatisfactionReport {
    /// Builds a report for the current batch.
    pub fn build<B: Backend, Ba: Batch<B>>(logits: Tensor<B, 3>, batch: &Ba) -> Self {
        let problems = batch.problems();
        let batch_size = problems.len();

        let assignment: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);
        let assignment: Vec<i64> = assignment
            .into_data()
            .to_vec::<B::IntElem>()
            .expect("decoded assignment should be an integer tensor")
            .into_iter()
            .map(|v| v.elem::<i64>())
            .collect();

        let mut by_constraint: HashMap<&'static str, ConstraintStats> = HashMap::new();

        for (i, problem) in problems.iter().enumerate().take(batch_size) {
            let number_vars = problem.number_variables();
            let start = i * number_vars;
            let sample_assignment: Vec<isize> = assignment[start..start + number_vars]
                .iter()
                .map(|&v| v as isize)
                .collect();

            // First tally (satisfied, total) per constraint type, within
            // this problem only.
            let mut per_type: HashMap<&'static str, (usize, usize)> = HashMap::new();
            for constraint in problem.iter_constraints() {
                let c = &problem[constraint];
                let entry = per_type.entry(c.name()).or_insert((0, 0));
                entry.1 += 1;
                if c.is_satisfied(&sample_assignment) {
                    entry.0 += 1;
                }
            }

            // Then fold this problem's per-type rate in as a single sample
            // -- this is the step that makes it "per problem" rather than
            // "per instance".
            for (name, (satisfied, total)) in per_type {
                if total == 0 {
                    continue;
                }
                let rate = satisfied as f64 / total as f64;
                let entry = by_constraint.entry(name).or_default();
                entry.count += 1;
                entry.sum_rate += rate;
                entry.sum_rate_sq += rate * rate;
                entry.min = entry.min.min(rate);
                entry.max = entry.max.max(rate);
            }
        }

        SatisfactionReport { by_constraint }
    }

    /// Merges another report into this one, e.g. across multiple batches in
    /// an epoch.
    pub fn merge(&mut self, other: SatisfactionReport) {
        for (name, stats) in other.by_constraint {
            let entry = self.by_constraint.entry(name).or_default();
            entry.count += stats.count;
            entry.sum_rate += stats.sum_rate;
            entry.sum_rate_sq += stats.sum_rate_sq;
            entry.min = entry.min.min(stats.min);
            entry.max = entry.max.max(stats.max);
        }
    }

    pub fn print(&self, width: usize) {
        if self.by_constraint.is_empty() {
            log::warn!("Can not show constraint satisfaction: no constraints recorded.");
            return;
        }

        let mut rows: Vec<(&str, &ConstraintStats)> =
            self.by_constraint.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap());

        let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(4);

        log::info!(
            "{:<name_width$} {:>8} {:>8}  {:>8}  {:>8}",
            "TYPE",
            "MIN",
            "MAX",
            "AVG SAT",
            "STDDEV",
        );
        log::info!("{}", "-".repeat(name_width + width + 32));

        for (name, stats) in &rows {
            log::info!(
                "{:<name_width$} {:>8.1}% {:>8.1}%  {:>7.1}%  {:>7.1}%",
                name,
                stats.min() * 100.0,
                stats.max() * 100.0,
                stats.mean() * 100.0,
                stats.stddev() * 100.0,
            );
        }
    }
}
