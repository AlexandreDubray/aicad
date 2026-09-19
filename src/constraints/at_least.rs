use super::*;
use crate::modelling::*;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// `AtLeastProperty` mirrors `AmongProperty`'s (min, max) envelope exactly -- `min` is a
/// guaranteed-achieved count, `max` an achievable-at-best count, folded across possibly
/// multiple incoming edges the same way `Among`/`Gcc` do. The only difference from `Among`
/// is that both fields are capped at `lb`: since `AtLeast` has no upper bound, once a count
/// has already reached `lb` there is nothing left that distinguishing it from any higher
/// count could ever affect -- every count >= lb behaves identically for every future
/// decision. Capping is therefore an exact state-space reduction, not a relaxation: see
/// `Gcc`'s equivalent saturation for values whose upper bound can never bind.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AtLeastProperty {
    values: Arc<FxHashSet<isize>>,
    lb: usize,
    min: usize,
    max: usize,
}

impl AtLeastProperty {
    fn new(values: Arc<FxHashSet<isize>>, lb: usize, min: usize, max: usize) -> Self {
        Self {
            values,
            lb,
            min,
            max,
        }
    }
}

/// At least `lb` of `variables` must take a value in `values`. This is exactly `Among`
/// without an upper bound -- the common "at least K of these" cardinality pattern (e.g. NSPLib
/// shift coverage: at least K nurses on a given shift, with no ceiling on how many more could
/// be). Modelling it directly, instead of via `Among`/`Gcc` with an upper bound pinned to
/// `variables.len()`, lets the compiled state saturate at `lb` instead of tracking every
/// count up to `variables.len()` -- which is what actually inflates the diagram for a bound
/// that can never bind, regardless of how many variables are in scope.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct AtLeast {
    variables: Vec<VariableIndex>,
    values: Arc<FxHashSet<isize>>,
    lb: usize,
    layer_in_scope: Vec<u64>,
}

impl AtLeast {
    pub fn new(variables: Vec<VariableIndex>, values: FxHashSet<isize>, lb: usize) -> Self {
        Self {
            variables,
            values: Arc::new(values),
            lb,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for AtLeast {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let mut values: Vec<isize> = self.values.iter().copied().collect();
        values.sort_unstable();
        ConstraintShapeKey::AtLeast {
            arity: self.variables.len(),
            values,
            lb: self.lb,
        }
    }

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
        let parent = parent.as_any().downcast_ref::<AtLeastProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on parent property of type {} instead of AtLeastProperty",
                    parent.name()
                );
        });
        let child = child.as_any().downcast_ref::<AtLeastProperty>().unwrap_or_else(|| {
                panic!(
                    "Calling is_assignment_invalid on child property of type {} instead of AtLeastProperty",
                    child.name()
                );
        });

        // No upper bound exists, so the only way this edge can be pruned is if even the best
        // achievable completion (parent's guaranteed-so-far max + child's achievable-suffix
        // max + this edge's own contribution) could never reach `lb`.
        let mut achievable_max = parent.max + child.max;
        if self.values.contains(&assignment) {
            achievable_max += 1;
        }
        achievable_max < self.lb
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
        count >= self.lb
    }

    fn name(&self) -> &'static str {
        "AtLeast"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AtLeastProperty::new(
            self.values.clone(),
            self.lb,
            usize::MAX,
            0,
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(AtLeastProperty::new(self.values.clone(), self.lb, 0, 0))
    }
}

impl ConstraintProperty for AtLeastProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
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

        // Cap at `lb`: once a count is already >= lb, every further increment is provably
        // interchangeable for every future decision (there is no upper bound left to violate).
        self.min = self.min.min(other.min + delta).min(self.lb);
        self.max = self.max.max(other.max + delta).min(self.lb);
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        // Both operands are already capped at `lb` by `update`, so a plain min/max fold
        // preserves that invariant without needing to re-cap here.
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn order_key(&self) -> Vec<f64> {
        vec![self.min as f64, self.max as f64]
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.min);
        hasher.write_usize(self.max);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<AtLeastProperty>()
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
        "AtLeastProperty"
    }
}

#[cfg(test)]
mod test_at_least {

    use crate::constraints::{AtLeast, Constraint};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHashSet;
    use std::sync::Arc;

    fn values(vals: &[isize]) -> FxHashSet<isize> {
        FxHashSet::from_iter(vals.iter().copied())
    }

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_bound_met() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let c = AtLeast::new(vars, values(&[1, 2]), 2);
        // value 1 (in set), 0 (not), 2 (in set) -> count = 2, meets lb = 2.
        assert!(c.is_satisfied(&[1, 0, 2]));
    }

    #[test]
    pub fn test_is_satisfied_bound_violated() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let c = AtLeast::new(vars, values(&[1, 2]), 2);
        // only one variable takes a value in {1, 2} -> count = 1 < lb = 2.
        assert!(!c.is_satisfied(&[1, 0, 0]));
    }

    #[test]
    pub fn test_is_satisfied_more_than_bound_is_fine() {
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let c = AtLeast::new(vars, values(&[1]), 1);
        // every variable takes value 1 -> count = 3, no upper bound to violate.
        assert!(c.is_satisfied(&[1, 1, 1]));
    }

    #[test]
    pub fn test_is_satisfied_zero_bound_always_holds() {
        let vars = vec![VariableIndex(0), VariableIndex(1)];
        let c = AtLeast::new(vars, values(&[1]), 0);
        assert!(c.is_satisfied(&[0, 0]));
        assert!(c.is_satisfied(&[1, 1]));
    }

    #[test]
    pub fn test_is_satisfied_empty_scope() {
        let c = AtLeast::new(vec![], values(&[1]), 0);
        assert!(c.is_satisfied(&[]));
    }

    // --- MDD construction / propagation / split-refine tests --- //

    #[test]
    pub fn test_basic_at_least_one() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        at_least(&mut problem, vec![x, y], vec![1], 1);

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
        assert!(is_solution(vec![1, 0], &solutions));
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 1], &solutions));
    }

    #[test]
    pub fn test_lower_bound_unsat() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![0], None);
        // Neither variable can ever take value 1, so the count is always 0 < lb = 1.
        at_least(&mut problem, vec![x, y], vec![1], 1);

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
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        at_least(&mut problem, vec![x, y], vec![1], 1);

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
        assert!(is_solution(vec![1, 1], &solutions));
    }

    #[test]
    pub fn test_no_bound_restriction_when_lb_zero() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        at_least(&mut problem, vec![x, y], vec![1], 0);

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
    }

    #[test]
    pub fn test_combined_with_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        at_least(&mut problem, vec![x, y, z], vec![2], 1);

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
                    if count >= 1 {
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
    pub fn test_saturates_state_space_at_lb_plus_one() {
        // 6 variables, each in {0, 1}: AtLeast(2) over value 1 should compile to at most
        // lb + 1 = 3 distinct exact states per layer (counts 0, 1, "2 or more"), regardless
        // of how many variables are in scope -- unlike a plain count that would need up to
        // 7 states (0..=6) if tracked unsaturated.
        let mut problem = Problem::default();
        let vars = problem.add_variables(6, vec![0, 1], None);
        at_least(&mut problem, vars.clone(), vec![1], 2);

        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(vars.iter().map(|v| v.0).collect()),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);
        for layer in 0..=6 {
            assert!(
                mdd.number_nodes_in_layer(layer) <= 3,
                "layer {} has {} nodes, expected at most 3 (saturated at lb=2)",
                layer,
                mdd.number_nodes_in_layer(layer)
            );
        }
    }
}
