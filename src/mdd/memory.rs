//! Memory report of the constraints held by an MDD.
//!
//! # Purpose
//!
//! `MemoryReport` sums the heap size of the constraints of a problem by constraint type and prints
//! them as a sorted table with a bar per type, through the `log` crate at `info` level. It is a
//! diagnostic for finding which constraints dominate memory; it does not measure the diagram.
//!
//! # Design notes
//!
//! - **Printing goes through `log`.** The caller decides whether the table is shown by choosing its
//!   log level, as for every other message of the library.

use crate::modelling::Constraint;

use std::collections::HashMap;

/// Memory used by the constraints of one type.
#[derive(Default)]
struct TypeStats {
    /// Number of constraints of the type.
    count: usize,
    /// Heap size of all of them, in bytes.
    total_bytes: usize,
    /// Heap size of the largest one, in bytes.
    max_bytes: usize,
}

/// Memory used by a set of constraints, grouped by constraint type.
pub(super) struct MemoryReport {
    /// Statistics per constraint type name.
    by_type: HashMap<&'static str, TypeStats>,
    /// Heap size of all the constraints, in bytes.
    grand_total: usize,
}

impl MemoryReport {
    /// Measures `constraints` and groups the sizes by constraint type.
    pub(super) fn build<'a>(
        constraints: impl Iterator<Item = &'a Box<dyn Constraint + Send + Sync>>,
    ) -> Self {
        let mut by_type: HashMap<&'static str, TypeStats> = HashMap::new();
        let mut grand_total = 0usize;

        for c in constraints {
            let bytes = c.deep_size_of();
            let entry = by_type.entry(c.name()).or_default();
            entry.count += 1;
            entry.total_bytes += bytes;
            entry.max_bytes = entry.max_bytes.max(bytes);
            grand_total += bytes;
        }

        MemoryReport {
            by_type,
            grand_total,
        }
    }

    /// Logs the table, largest type first, with bars `width` characters wide.
    /// Logs a warning instead if there is nothing to report.
    pub(super) fn print(&self, width: usize) {
        if self.grand_total == 0 {
            log::warn!("Trying to show mdd memory report: no constraints recorded.");
            return;
        }

        // Sort by total_bytes descending.
        let mut rows: Vec<(&str, &TypeStats)> = self.by_type.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.1.total_bytes));

        let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(4);

        log::info!(
            "{:<name_width$}  {:>10}  {:>8}  {:>10}  {:>6}  {}",
            "TYPE",
            "TOTAL",
            "COUNT",
            "AVG",
            "%",
            "",
            name_width = name_width
        );
        log::info!("{}", "-".repeat(name_width + width + 45));

        for (name, stats) in &rows {
            let pct = stats.total_bytes as f64 / self.grand_total as f64;
            let bar_len = (pct * width as f64).round() as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(width - bar_len);
            let avg = stats.total_bytes / stats.count.max(1);

            log::info!(
                "{:<name_width$}  {:>10}  {:>8}  {:>10}  {:>5.1}%  {}",
                name,
                human_bytes(stats.total_bytes),
                stats.count,
                human_bytes(avg),
                pct * 100.0,
                bar,
                name_width = name_width
            );
        }

        log::info!("{}", "-".repeat(name_width + width + 45));
        log::info!(
            "{:<name_width$}  {:>10}",
            "TOTAL",
            human_bytes(self.grand_total),
            name_width = name_width
        );
    }
}

/// Formats a byte count with a binary unit (`B`, `KB`, `MB`, `GB`, `TB`).
fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.2} {}", size, UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, gcc};
    use crate::modelling::Problem;

    #[test]
    fn byte_counts_use_the_largest_unit_below_them() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.00 KB");
        assert_eq!(human_bytes(1536), "1.50 KB");
        assert_eq!(human_bytes(3 * 1024 * 1024), "3.00 MB");
        assert!(human_bytes(usize::MAX).ends_with(" TB"));
    }

    #[test]
    fn the_report_groups_by_type_and_its_totals_add_up() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        all_different(&mut problem, vars[0..2].to_vec());
        all_different(&mut problem, vars[2..4].to_vec());
        gcc(&mut problem, vars.clone(), vec![(0, 0, 2)]);

        let report = MemoryReport::build(problem.constraints().iter());

        assert_eq!(report.by_type.len(), 2);
        let counts: usize = report.by_type.values().map(|s| s.count).sum();
        assert_eq!(counts, 3);
        let bytes: usize = report.by_type.values().map(|s| s.total_bytes).sum();
        assert_eq!(bytes, report.grand_total);
        assert!(report.grand_total > 0);
        assert!(
            report
                .by_type
                .values()
                .all(|s| s.max_bytes <= s.total_bytes && s.max_bytes > 0)
        );
        report.print(20);
    }

    #[test]
    fn an_empty_report_prints_a_warning_instead_of_a_table() {
        let report = MemoryReport::build(Problem::default().constraints().iter());
        assert_eq!(report.grand_total, 0);
        report.print(20);
    }
}
