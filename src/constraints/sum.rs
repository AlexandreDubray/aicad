use super::*;
use crate::mdd::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

#[derive(Clone, deepsize::DeepSizeOf)]
pub struct SumProperty {
    min: isize,
    max: isize,
    values: Arc<FxHashSet<isize>>,
}

#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Sum {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Union of the domains of the variables
    values: Arc<FxHashSet<isize>>,
    /// Target value the sum of the scope's variables must equal
    target: isize,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Sum {
    /// Creates a new Sum constraint: the sum of the given variables must equal `target`.
    pub fn new(variables: Vec<VariableIndex>, target: isize, problem: &Problem) -> Self {
        let mut values = FxHashSet::<isize>::default();
        for variable in variables.iter().copied() {
            values.extend(problem[variable].iter_domain());
        }
        Self {
            variables,
            values: Arc::new(values),
            target,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Sum {
    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        self.layer_in_scope = (0..(order.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
        for (layer, variable) in order.iter().enumerate() {
            if scope.contains(variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
            }
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        _layer: usize,
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

    fn rank_nodes(&self, _nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(SumProperty {
            min: isize::MAX,
            max: isize::MIN,
            values: Arc::clone(&self.values),
        })
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(SumProperty {
            min: 0,
            max: 0,
            values: Arc::clone(&self.values),
        })
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

    use crate::constraints::{Constraint, Sum};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use std::sync::Arc;

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_matches_target() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2, 3], None);
        let sum = Sum::new(vars, 6, &problem);
        assert!(sum.is_satisfied(&[1, 2, 3]));
    }

    #[test]
    pub fn test_is_satisfied_does_not_match_target() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2, 3], None);
        let sum = Sum::new(vars, 6, &problem);
        assert!(!sum.is_satisfied(&[1, 2, 2]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_zero_target() {
        let problem = Problem::default();
        let sum = Sum::new(vec![], 0, &problem);
        assert!(sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope_nonzero_target() {
        let problem = Problem::default();
        let sum = Sum::new(vec![], 1, &problem);
        assert!(!sum.is_satisfied(&[]));
    }

    #[test]
    pub fn test_is_satisfied_negative_values() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![-5, -2, 2, 5], None);
        let sum = Sum::new(vars, -3, &problem);
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With no refine step, the freshly-built MDD is already the width-1 relaxation: it
        // must not exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        sum(&mut problem, vec![x, y], 3);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
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

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
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
