use std::collections::HashMap;

use std::sync::Arc;

use crate::modelling::Problem;

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

pub(crate) struct SatisfactionReport {
    by_constraint: HashMap<&'static str, ConstraintStats>,
}

impl SatisfactionReport {
    /// Builds a report for `assignments`, one per problem of `problems`, in the same order. Each
    /// assignment holds one value per variable of its problem; assignments need not have the same
    /// length.
    ///
    /// # Panics
    ///
    /// If there are not as many assignments as problems.
    pub(crate) fn build(problems: &[Arc<Problem>], assignments: &[Vec<isize>]) -> Self {
        assert_eq!(
            problems.len(),
            assignments.len(),
            "a report needs one assignment per problem"
        );

        let mut by_constraint: HashMap<&'static str, ConstraintStats> = HashMap::new();

        for (problem, sample_assignment) in problems.iter().zip(assignments) {
            // First tally (satisfied, total) per constraint type, within
            // this problem only.
            let mut per_type: HashMap<&'static str, (usize, usize)> = HashMap::new();
            for constraint in problem.iter_constraints() {
                let c = &problem[constraint];
                let entry = per_type.entry(c.name()).or_insert((0, 0));
                entry.1 += 1;
                if c.is_satisfied(sample_assignment) {
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
    pub(crate) fn merge(&mut self, other: SatisfactionReport) {
        for (name, stats) in other.by_constraint {
            let entry = self.by_constraint.entry(name).or_default();
            entry.count += stats.count;
            entry.sum_rate += stats.sum_rate;
            entry.sum_rate_sq += stats.sum_rate_sq;
            entry.min = entry.min.min(stats.min);
            entry.max = entry.max.max(stats.max);
        }
    }

    pub(crate) fn print(&self, width: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::not_equals;

    fn chain(variables: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars: Vec<_> = (0..variables)
            .map(|_| problem.add_variable(vec![0, 1], None))
            .collect();
        for pair in vars.windows(2) {
            not_equals(&mut problem, pair[0], pair[1]);
        }
        Arc::new(problem)
    }

    #[test]
    fn problems_of_different_sizes_are_scored_each_on_its_own_assignment() {
        let problems = vec![chain(2), chain(4)];
        // The first satisfies its constraint, the second violates two of its three.
        let report = SatisfactionReport::build(&problems, &[vec![0, 1], vec![0, 0, 0, 1]]);
        let stats = &report.by_constraint["Not Equals"];
        assert_eq!(stats.count, 2);
        assert!((stats.max() - 1.0).abs() < 1e-12);
        assert!((stats.min() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "one assignment per problem")]
    fn every_problem_needs_an_assignment() {
        SatisfactionReport::build(&[chain(2)], &[]);
    }
}
