use crate::mdd::heuristics::EliminationOrdering;
use crate::modelling::{ConstraintIndex, Problem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintGrouping {
    pub ordering: EliminationOrdering,
    pub size_bound: usize,
}

impl ConstraintGrouping {
    pub const PER_CONSTRAINT: Self = Self {
        ordering: EliminationOrdering::GreedyMinFill,
        size_bound: 0,
    };

    /// Returns the constraint-index groups to compile, one MDD per group.
    pub fn groups(&self, problem: &Problem) -> Vec<Vec<ConstraintIndex>> {
        self.ordering.buckets(problem, self.size_bound)
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

        let grouping = ConstraintGrouping {
            ordering: EliminationOrdering::GreedyMinFill,
            size_bound: 3,
        };
        let groups = grouping.groups(&problem);
        let total: usize = groups.iter().map(|cs| cs.len()).sum();
        assert_eq!(total, 3);
        assert!(groups.iter().any(|cs| cs.len() > 1));
    }
}
