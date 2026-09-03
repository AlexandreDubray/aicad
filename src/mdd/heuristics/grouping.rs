use crate::mdd::heuristics::EliminationOrdering;
use crate::modelling::{ConstraintIndex, Problem};

/// How to partition a problem's constraints into the groups that get compiled into (one MDD
/// each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintGrouping {
    RollingWindow {
        ordering: EliminationOrdering,
        window_size: usize,
    },
}

impl ConstraintGrouping {
    /// Returns the constraint-index groups to compile, one MDD per group.
    pub fn groups(&self, problem: &Problem) -> Vec<Vec<ConstraintIndex>> {
        match self {
            Self::RollingWindow {
                ordering,
                window_size,
            } => ordering.rolling_window_groups(problem, *window_size),
        }
    }

    pub fn new_rolling(window_size: usize) -> Self {
        Self::RollingWindow {
            ordering: EliminationOrdering::GreedyMinFill,
            window_size,
        }
    }
}

impl Default for ConstraintGrouping {
    fn default() -> Self {
        Self::RollingWindow {
            ordering: EliminationOrdering::GreedyMinFill,
            window_size: 1,
        }
    }
}
