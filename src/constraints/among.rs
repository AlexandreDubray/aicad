use super::*;
use crate::mdd::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

pub struct AmongProperty {
    values: Arc<FxHashSet<isize>>,
    min: usize,
    max: usize,
}

impl AmongProperty {
    fn new(values: Arc<FxHashSet<isize>>, min: usize, max: usize) -> Self {
        Self { values, min, max }
    }
}

#[derive(deepsize::DeepSizeOf)]
pub struct Among {
    variables: Vec<VariableIndex>,
    values: Arc<FxHashSet<isize>>,
    lb: usize,
    ub: usize,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Among {
    pub fn new(
        variables: Vec<VariableIndex>,
        values: FxHashSet<isize>,
        lb: usize,
        ub: usize,
    ) -> Self {
        Self {
            variables,
            values: Arc::new(values),
            lb,
            ub,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Among {
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
        let parent = parent.as_any().downcast_ref::<AmongProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of AmongProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<AmongProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of AmongProperty",
                    child.name()
                );
        });

        let mut local_lb = parent.min + child.min;
        if self.values.contains(&assignment) {
            local_lb += 1;
        }
        let mut local_ub = parent.max + child.max;
        if self.values.contains(&assignment) {
            local_ub += 1;
        }
        local_lb > self.ub || local_ub < self.lb
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut count = 0;
        for variable in self.variables.iter().copied() {
            let value = assignment[variable.0];
            if self.values.contains(&value) {
                count += 1;
            }
        }
        self.lb <= count && count <= self.ub
    }

    fn name(&self) -> &'static str {
        "Among"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn rank_nodes(&self, nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        AmongProperty::new(self.values.clone(), usize::MAX, 0);
    }
}

impl ConstraintProperty for AmongProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<AmongProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        let delta = if in_scope && self.values.contains(&assignment) {
            1
        } else {
            0
        };
        let candidate_lb = other.min + delta;
        let candidate_ub = other.max + delta;

        self.min = self.min.min(other.min + delta);
        self.max = self.max.max(other.max + delta);
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for word in self.value_all_path.iter() {
            hasher.write_u64(word);
        }
        for word in self.value_some_path.iter() {
            hasher.write_u64(word);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<AmongProperty>()
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
        "AmongProperty"
    }
}

#[cfg(test)]
mod test_among {

    use crate::constraints::{Among, Constraint};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHashSet;

    fn values(vals: &[isize]) -> FxHashSet<isize> {
        FxHashSet::from_iter(vals.iter().copied())
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_within_bounds() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 1, 2);
        // value 1 (in set), 0 (not in set), 2 (in set) -> count = 2, within [1, 2]
        assert!(among.is_satisfied(&[1, 0, 2]));
    }

    #[test]
    pub fn test_is_satisfied_lower_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 1, 2);
        // no variable takes a value in {1, 2} -> count = 0 < lb = 1
        assert!(!among.is_satisfied(&[0, 0, 0]));
    }

    #[test]
    pub fn test_is_satisfied_upper_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let among = Among::new(vars, values(&[1, 2]), 0, 1);
        // three variables take a value in {1, 2} -> count = 3 > ub = 1
        assert!(!among.is_satisfied(&[1, 2, 1]));
    }

    #[test]
    pub fn test_is_satisfied_boundary_lb_eq_ub() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        let among = Among::new(vars, values(&[5]), 1, 1);
        assert!(among.is_satisfied(&[5, 0]));
        assert!(!among.is_satisfied(&[5, 5]));
        assert!(!among.is_satisfied(&[0, 0]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let among = Among::new(vec![], values(&[1]), 0, 0);
        assert!(among.is_satisfied(&[]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_exactly_one() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 1, 1);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_range_bounds() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        among(&mut problem, vec![x, y, z], vec![0, 1], 1, 2);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        // Brute-force the expected solution set: 3^3 assignments, keep those where the
        // number of variables taking a value in {0, 1} is between 1 and 2 (inclusive).
        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let count = [a, b, c].iter().filter(|v| **v == 0 || **v == 1).count();
                    if (1..=2).contains(&count) {
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
    pub fn test_lower_bound_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
        // Neither variable can ever take value 1, so the count is always 0 < lb = 1.
        among(&mut problem, vec![x, y], vec![1], 1, 2);

        let mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_upper_bound_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![1], None);
        let y = problem.add_variable(vec![1], None);
        // Both variables are forced to 1, so the count is always 2 > ub = 0.
        among(&mut problem, vec![x, y], vec![1], 0, 0);

        let mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution (though it may also keep invalid ones).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 1, 1);

        let mdd = Mdd::new(
            problem,
            1,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_all_variables_must_take_value() {
        // lb = ub = number of variables: every variable must take a value in the set.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        let z = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y, z], vec![1], 3, 3);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 1);
        assert!(is_solution(vec![1, 1, 1], &solutions));
    }

    #[test]
    pub fn test_no_bound_restriction() {
        // lb = 0 and ub = number of variables imposes no real restriction: every
        // combination is a valid solution.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        among(&mut problem, vec![x, y], vec![1], 0, 2);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::MinDomMaxLinked,
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 4);
        assert!(is_solution(vec![0, 0], &solutions));
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
        assert!(is_solution(vec![1, 1], &solutions));
    }

    #[test]
    pub fn test_combined_with_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        among(&mut problem, vec![x, y, z], vec![2], 1, 1);

        let mut mdd = Mdd::new(
            problem,
            usize::MAX,
            OrderingHeuristic::Custom(vec![0, 1, 2]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
        );
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                if a == b {
                    continue;
                }
                for c in 0..3 {
                    let count = [a, b, c].iter().filter(|v| **v == 2).count();
                    if count == 1 {
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
