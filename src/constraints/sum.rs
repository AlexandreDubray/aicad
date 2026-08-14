use super::*;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::Hasher;

pub struct SumProperty {
    min: isize,
    max: isize,
}

#[derive(deepsize::DeepSizeOf)]
pub struct Sum {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Target value the sum of the scope's variables must equal
    target: isize,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Sum {

    /// Creates a new Sum constraint: the sum of the given variables must equal `target`.
    pub fn new(variables: Vec<VariableIndex>, target: isize) -> Self {
        let layer_in_scope = (0..(variables.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
        Self {
            variables,
            target,
            layer_in_scope,
        }
    }

}

impl Constraint for Sum {

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        // The layers in the scope of the variable are indicated using a bitvector of 64-bit words.
        // For each layer l its word index is given by l / 64 and the bit index by l % 64
        for variable in self.variables.iter() {
            let layer = ordering[variable.0];
            // Sets the bit of the layer to 1
            self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        assignment: isize,
    ) -> bool {
        let parent = parent.as_any().downcast_ref::<SumProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of SumProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<SumProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of SumProperty",
                    child.name()
                );
        });

        let local_min = parent.min + child.min + assignment;
        let local_max = parent.max + child.max + assignment;
        // TODO: check if this work with negative values
        local_min > self.target || local_max < self.target
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut total: isize = 0;
        for variable in self.variables.iter().copied() {
            total += assignment[variable.0];
        }
        total == self.target
    }

    fn name(&self) -> &'static str {
        "Sum"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn identity_property(&self) -> SumProperty {
        SumProperty { min: 0, max: 0 }
    }
}

impl ConstraintProperty for SumProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<SumProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        let delta = if in_scope && self.values.contains(&assignment) {
            assignment
        } else {
            0
        };
        let candidate_lb = other.min + delta;
        let candidate_ub = other.max + delta;

        self.min = self.min.min(other.min + delta);
        self.max = self.max.max(other.max + delta);
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_isize(self.min);
        hasher.write_isize(self.max);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<SumProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling eq on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        self.min == other.min && self.max == other.max
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "SumProperty"
    }

}

#[cfg(test)]
mod test_sum {

    use crate::modelling::*;
    use crate::constraints::{Sum, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_matches_target() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let sum = Sum::new(vars, 6);
        assert!(sum.is_satisfied(&[1, 2, 3]));
    }

    #[test]
    pub fn test_is_satisfied_does_not_match_target() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let sum = Sum::new(vars, 6);
        assert!(!sum.is_satisfied(&[1, 2, 2]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_zero_target() {
        let sum = Sum::new(vec![], 0);
        assert!(sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_nonzero_target() {
        let sum = Sum::new(vec![], 1);
        assert!(!sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_negative_values() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        let sum = Sum::new(vars, -3);
        assert!(sum.is_satisfied(&[-5, 2]));
        assert!(!sum.is_satisfied(&[5, -2]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_sum() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        sum(&mut problem, vec![x, y], 3);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![1, 2], &solutions));
        assert!(is_solution(vec![2, 1], &solutions));
    }

    #[test]
    pub fn test_sum_brute_force() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2, 3], None);
        let y = problem.add_variable(vec![0, 1, 2, 3], None);
        let z = problem.add_variable(vec![0, 1, 2, 3], None);
        sum(&mut problem, vec![x, y, z], 5);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    if a + b + c == 5 {
                        expected.push(vec![a, b, c]);
                    }
                }
            }
        }

        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }

    #[test]
    pub fn test_unsat_forced_mismatch() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
        // Both variables are forced to 0, so the sum is always 0, never 5.
        sum(&mut problem, vec![x, y], 5);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_unsat_target_above_reachable_range() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        // The maximum reachable sum is 2, so a target of 10 is unreachable.
        sum(&mut problem, vec![x, y], 10);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_unsat_target_below_reachable_range() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        // The minimum reachable sum is 0, so a target of -10 is unreachable.
        sum(&mut problem, vec![x, y], -10);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        sum(&mut problem, vec![x, y], 3);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![1, 2], &solutions));
        assert!(is_solution(vec![2, 1], &solutions));
    }

    #[test]
    pub fn test_negative_domain_values() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![-1, 0, 1], None);
        let y = problem.add_variable(vec![-1, 0, 1], None);
        sum(&mut problem, vec![x, y], 0);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 3);
        assert!(is_solution(vec![-1, 1], &solutions));
        assert!(is_solution(vec![0, 0], &solutions));
        assert!(is_solution(vec![1, -1], &solutions));
    }

    #[test]
    pub fn test_combined_with_all_different() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2, 3], None);
        let y = problem.add_variable(vec![0, 1, 2, 3], None);
        let z = problem.add_variable(vec![0, 1, 2, 3], None);
        all_different(&mut problem, vec![x, y, z]);
        sum(&mut problem, vec![x, y, z], 6);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed, SelectHeuristic::Greedy);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..4 {
            for b in 0..4 {
                if a == b {
                    continue;
                }
                for c in 0..4 {
                    if c == a || c == b {
                        continue;
                    }
                    if a + b + c == 6 {
                        expected.push(vec![a, b, c]);
                    }
                }
            }
        }
        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }
}
