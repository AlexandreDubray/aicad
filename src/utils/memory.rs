use crate::constraints::Constraint;

use std::collections::HashMap;

// --- Data model -------------------------------------------------------

#[derive(Default)]
struct TypeStats {
    count: usize,
    total_bytes: usize,
    max_bytes: usize,
}

pub struct MemoryReport {
    by_type: HashMap<&'static str, TypeStats>,
    grand_total: usize,
}

impl MemoryReport {
    pub fn build<'a>(constraints: impl Iterator<Item = &'a Box<dyn Constraint + Send + Sync>>) -> Self {
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

        MemoryReport { by_type, grand_total }
    }

    /// Pretty-print a sorted bar chart to the terminal.
    /// `width` controls how wide the bar column is (in characters).
    pub fn print(&self, width: usize) {
        if self.grand_total == 0 {
            println!("No constraints recorded.");
            return;
        }

        // Sort by total_bytes descending.
        let mut rows: Vec<(&str, &TypeStats)> = self.by_type.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by(|a, b| b.1.total_bytes.cmp(&a.1.total_bytes));

        let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(4);

        println!(
            "{:<name_width$}  {:>10}  {:>8}  {:>10}  {:>6}  {}",
            "TYPE", "TOTAL", "COUNT", "AVG", "%", "",
            name_width = name_width
        );
        println!("{}", "-".repeat(name_width + width + 45));

        for (name, stats) in &rows {
            let pct = stats.total_bytes as f64 / self.grand_total as f64;
            let bar_len = (pct * width as f64).round() as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(width - bar_len);
            let avg = stats.total_bytes / stats.count.max(1);

            println!(
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

        println!("{}", "-".repeat(name_width + width + 45));
        println!(
            "{:<name_width$}  {:>10}",
            "TOTAL",
            human_bytes(self.grand_total),
            name_width = name_width
        );
    }
}

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
