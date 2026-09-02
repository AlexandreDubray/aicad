use crate::mdd::heuristics::EliminationOrdering;
use crate::modelling::{ConstraintIndex, Problem};

/// How to partition a problem's constraints into the groups that get compiled into (one MDD
/// each). `Buckets` gives disjoint groups (each constraint belongs to exactly one), `RollingWindow`
/// gives overlapping ones -- see each variant's field docs and `EliminationOrdering`'s methods for
/// the tradeoffs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintGrouping {
    /// Disjoint mini-bucket-style grouping -- see `EliminationOrdering::buckets`.
    Buckets {
        ordering: EliminationOrdering,
        size_bound: usize,
    },
    /// Overlapping sliding-window grouping over constraints sorted by elimination position -- see
    /// `EliminationOrdering::rolling_window_groups`. A constraint can end up compiled into more
    /// than one MDD; a consumer that treats every membership as independent evidence (e.g. belief
    /// propagation) will double-count it once per extra membership.
    RollingWindow {
        ordering: EliminationOrdering,
        window_size: usize,
    },
}

impl ConstraintGrouping {
    pub const PER_CONSTRAINT: Self = Self::Buckets {
        ordering: EliminationOrdering::GreedyMinFill,
        size_bound: 0,
    };

    /// Builds a grouping from the two knobs exposed to callers/Python: a non-zero `window_size`
    /// selects overlapping rolling-window grouping; otherwise `size_bound` selects (possibly
    /// disjoint) bucket grouping, with `size_bound == 0` meaning one MDD per constraint.
    pub fn from_config(size_bound: usize, window_size: usize) -> Self {
        if window_size > 0 {
            Self::RollingWindow {
                ordering: EliminationOrdering::GreedyMinFill,
                window_size,
            }
        } else {
            Self::Buckets {
                ordering: EliminationOrdering::GreedyMinFill,
                size_bound,
            }
        }
    }

    /// Returns the constraint-index groups to compile, one MDD per group.
    pub fn groups(&self, problem: &Problem) -> Vec<Vec<ConstraintIndex>> {
        match self {
            Self::Buckets {
                ordering,
                size_bound,
            } => ordering.buckets(problem, *size_bound),
            Self::RollingWindow {
                ordering,
                window_size,
            } => ordering.rolling_window_groups(problem, *window_size),
        }
    }
}

impl Default for ConstraintGrouping {
    fn default() -> Self {
        Self::PER_CONSTRAINT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::not_equals;

    #[test]
    fn per_constraint_yields_one_singleton_group_per_constraint() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);

        let groups = ConstraintGrouping::PER_CONSTRAINT.groups(&problem);
        assert_eq!(groups.len(), 2);
        for constraints in &groups {
            assert_eq!(constraints.len(), 1);
        }
    }

    #[test]
    fn default_is_per_constraint() {
        assert_eq!(
            ConstraintGrouping::default(),
            ConstraintGrouping::PER_CONSTRAINT
        );
    }

    #[test]
    fn a_wide_enough_bound_merges_a_triangles_constraints_and_still_covers_everything() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let grouping = ConstraintGrouping::Buckets {
            ordering: EliminationOrdering::GreedyMinFill,
            size_bound: 3,
        };
        let groups = grouping.groups(&problem);
        let total: usize = groups.iter().map(|cs| cs.len()).sum();
        assert_eq!(total, 3);
        assert!(groups.iter().any(|cs| cs.len() > 1));
    }

    #[test]
    fn rolling_window_can_merge_a_triangle_that_buckets_never_can() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let grouping = ConstraintGrouping::RollingWindow {
            ordering: EliminationOrdering::GreedyMinFill,
            window_size: 3,
        };
        let groups = grouping.groups(&problem);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn from_config_picks_rolling_window_only_when_window_size_is_non_zero() {
        assert_eq!(
            ConstraintGrouping::from_config(2, 0),
            ConstraintGrouping::Buckets {
                ordering: EliminationOrdering::GreedyMinFill,
                size_bound: 2,
            }
        );
        assert_eq!(
            ConstraintGrouping::from_config(0, 4),
            ConstraintGrouping::RollingWindow {
                ordering: EliminationOrdering::GreedyMinFill,
                window_size: 4,
            }
        );
        assert_eq!(
            ConstraintGrouping::from_config(0, 0),
            ConstraintGrouping::PER_CONSTRAINT
        );
    }
}
