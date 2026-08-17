use super::*;
use crate::mdd::*;
use crate::modelling::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

#[derive(Clone, deepsize::DeepSizeOf)]
struct GccProperty {
    map: Arc<FxHashMap<isize, usize>>,
    min: Vec<usize>,
    max: Vec<usize>,
}

impl GccProperty {
    pub fn new(n: usize, map: Arc<FxHashMap<isize, usize>>, min_seed: usize) -> Self {
        Self {
            map,
            min: vec![min_seed; n],
            max: vec![0; n],
        }
    }
}

#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Gcc {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Required [lo, hi] occurrence range for each explicitly-bounded value. Values of the
    /// joint domain that are not keys of this map are unconstrained.
    bounds: Vec<(isize, usize, usize)>,
    /// Map each value of `domain` to a slot in the properties' vectors
    val_to_bit: Arc<FxHashMap<isize, usize>>,
    /// For each value (by its slot), the required lower/upper occurrence bound
    lo: Vec<usize>,
    hi: Vec<usize>,
    /// Bitvector to indicate if a layer is in the scope of the constraint or not
    layer_in_scope: Vec<u64>,
}

impl Gcc {
    /// Creates a new GCC constraint over `variables`. `bounds` maps each cardinality
    /// constrained value to its required (lo, hi) occurrence range. Any value of the joint
    /// domain absent from `bounds` is implicitly unconstrained (range [0, |variables|]).
    pub fn new(variables: Vec<VariableIndex>, bounds: Vec<(isize, usize, usize)>) -> Self {
        let mut check = FxHashMap::<isize, (usize, usize)>::default();
        for (value, lb, ub) in bounds.iter().copied() {
            match check.get(&value) {
                None => {
                    let _ = check.insert(value, (lb, ub));
                }
                Some(&(l, u)) => {
                    if l != lb || u != ub {
                        log::warn!("GCC constraint has multiple bounds for value {}: First bound ({}, {}), second bound ({}, {}). Last bound is kept.", value, l, u, lb, ub);
                    }
                }
            };
        }
        let val_to_bit = Arc::new(
            bounds
                .iter()
                .copied()
                .enumerate()
                .map(|(bit, (value, _, _))| (value, bit))
                .collect(),
        );
        let lo = bounds.iter().copied().map(|(_, lo, _)| lo).collect();
        let hi = bounds.iter().copied().map(|(_, _, hi)| hi).collect();
        Self {
            variables,
            bounds,
            val_to_bit,
            lo,
            hi,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Gcc {
    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        self.layer_in_scope = (0..(order.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
        for (layer, variable) in order.iter().enumerate() {
            if scope.contains(variable) {
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
        let parent = parent.as_any().downcast_ref::<GccProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of GccProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<GccProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of GccProperty",
                    child.name()
                );
        });

        // `bit` is `None` when `assignment` isn't itself one of the bounded values - in that
        // case this edge contributes `delta = 0` to every bounded value's count, but the bound
        // check below must still run: an edge to an *unbounded* value can still be the one that
        // makes some other bounded value's count infeasible to complete (not enough variables
        // left to reach its lower bound, or already past its upper bound).
        let bit = self.val_to_bit.get(&assignment).copied();
        for (v, lb, ub) in self.bounds.iter().copied() {
            let v_bit = *self.val_to_bit.get(&v).unwrap();
            let delta = if bit == Some(v_bit) { 1 } else { 0 };
            let min = parent.min[v_bit] + child.min[v_bit] + delta;

            if min > ub {
                return true;
            }
            let max = parent.max[v_bit] + child.max[v_bit] + delta;
            if max < lb {
                return true;
            }
        }
        false
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut counts: FxHashMap<isize, usize> = FxHashMap::default();
        for variable in self.variables.iter().copied() {
            let value = assignment[*variable];
            *counts.entry(value).or_insert(0) += 1;
        }
        for (value, lo, hi) in self.bounds.iter().copied() {
            let count = counts.get(&value).copied().unwrap_or(0);
            if count < lo || count > hi {
                return false;
            }
        }
        true
    }

    fn name(&self) -> &'static str {
        "GCC"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn rank_nodes(&self, _nodes: &[NodeIndex]) -> Vec<f64> {
        vec![]
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(GccProperty::new(
            self.bounds.len(),
            self.val_to_bit.clone(),
            usize::MAX,
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(GccProperty::new(
            self.bounds.len(),
            self.val_to_bit.clone(),
            0,
        ))
    }
}

impl ConstraintProperty for GccProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<GccProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        let target_bit = if in_scope {
            match self.map.get(&assignment) {
                None => self.min.len(),
                Some(&bit) => bit,
            }
        } else {
            self.min.len()
        };

        // Then, we integrate the min-max values for each bounded value from the other property
        for bit in 0..self.min.len() {
            if bit == target_bit {
                self.min[bit] = self.min[bit].min(other.min[bit] + 1);
                self.max[bit] = self.max[bit].max(other.max[bit] + 1);
            } else {
                self.min[bit] = self.min[bit].min(other.min[bit]);
                self.max[bit] = self.max[bit].max(other.max[bit]);
            }
        }
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for &bound in self.min.iter() {
            hasher.write_usize(bound);
        }
        for &bound in self.max.iter() {
            hasher.write_usize(bound);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<GccProperty>()
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
        "GccProperty"
    }
}

#[cfg(test)]
mod test_gcc {

    use crate::constraints::{Constraint, Gcc};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use std::sync::Arc;

    #[test]
    pub fn test_is_satisfied_within_bounds() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        // value 0 must appear exactly once, value 1 between 1 and 2 times.
        let gcc = Gcc::new(vars, vec![(0, 1, 1), (1, 1, 2)]);
        assert!(gcc.is_satisfied(&[0, 1, 1]));
    }

    #[test]
    pub fn test_is_satisfied_lower_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let gcc = Gcc::new(vars, vec![(2, 1, 3)]);
        // value 2 never appears, but requires at least 1 occurrence.
        assert!(!gcc.is_satisfied(&[0, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_upper_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let gcc = Gcc::new(vars, vec![(1, 0, 1)]);
        // value 1 appears twice, more than the allowed maximum of 1.
        assert!(!gcc.is_satisfied(&[1, 1, 0]));
    }

    #[test]
    pub fn test_is_satisfied_unbounded_value_is_free() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        // Only value 0 is constrained; value 3 can occur any number of times.
        let gcc = Gcc::new(vars, vec![(0, 1, 1)]);
        assert!(gcc.is_satisfied(&[0, 3]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let gcc = Gcc::new(vec![], vec![]);
        assert!(gcc.is_satisfied(&[]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_all_different_via_gcc() {
        // Each of the 3 values must appear at most once among 3 variables: equivalent to
        // all_different.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        gcc(&mut problem, vars, vec![(0, 0, 1), (1, 0, 1), (2, 0, 1)]);

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
        assert_eq!(solutions.len(), 6);
        for sol in solutions.iter() {
            assert_ne!(sol[0], sol[1]);
            assert_ne!(sol[1], sol[2]);
            assert_ne!(sol[0], sol[2]);
        }
    }

    #[test]
    pub fn test_exact_multiplicity() {
        // 4 binary variables, value 1 must appear exactly twice.
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1], None);
        gcc(&mut problem, vars, vec![(1, 2, 2)]);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vec![0, 1, 2, 3]),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        let count = [a, b, c, d].iter().filter(|v| **v == 1).count();
                        if count == 2 {
                            expected.push(vec![a, b, c, d]);
                        }
                    }
                }
            }
        }
        assert_eq!(expected.len(), 6);
        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }

    #[test]
    pub fn test_range_bounds_per_value() {
        // 3 ternary variables: value 0 between 1 and 2 times, value 2 at most 1 time.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        gcc(&mut problem, vars, vec![(0, 1, 2), (2, 0, 1)]);

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
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let count0 = [a, b, c].iter().filter(|v| **v == 0).count();
                    let count2 = [a, b, c].iter().filter(|v| **v == 2).count();
                    if (1..=2).contains(&count0) && count2 <= 1 {
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
    pub fn test_lower_bound_unsat_unreachable_value() {
        // Value 1 requires at least one occurrence, but no variable can ever be assigned it.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0], None);
        gcc(&mut problem, vars, vec![(1, 1, 2)]);

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
    pub fn test_upper_bound_unsat() {
        // Both variables forced to 1, but value 1 is capped at 1 occurrence.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![1], None);
        gcc(&mut problem, vars, vec![(1, 0, 1)]);

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
        let vars = problem.add_variables(2, vec![0, 1], None);
        gcc(&mut problem, vars, vec![(1, 1, 1)]);

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
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_no_bound_restriction() {
        // Unbounded values impose no restriction: every combination is a valid solution.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1], None);
        gcc(&mut problem, vars, vec![]);

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
        gcc(&mut problem, vec![x, y, z], vec![(2, 1, 1)]);

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
