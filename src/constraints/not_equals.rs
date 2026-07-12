use super::*;
use crate::utils::SparseBitset;
use crate::modelling::*;
use crate::mdd::*;
use std::hash::{Hash, Hasher};
use rustc_hash::FxHashSet;

pub struct NotEquals {
    x: VariableIndex,
    y: VariableIndex,
    domains: FxHashSet<isize>,
    top_down_properties: Vec<Vec<SparseBitset<isize>>>,
    bottom_up_properties: Vec<Vec<SparseBitset<isize>>>,
    layer_x: usize,
    layer_y: usize,
}

impl NotEquals {

    pub fn new(x: VariableIndex, y: VariableIndex) -> Self {
        Self {
            x,
            y,
            domains: FxHashSet::<isize>::default(),
            top_down_properties: vec![],
            bottom_up_properties: vec![],
            layer_x: 0,
            layer_y: 0,
        }
    }

}

impl Constraint for NotEquals {

    fn init(&mut self, vars: &[Variable]) {
        for value in vars[*self.x].iter_domain() {
            self.domains.insert(value);
        }
        for value in vars[*self.y].iter_domain() {
            self.domains.insert(value);
        }
        self.top_down_properties = (0..vars.len() + 1).map(|_| {
            vec![SparseBitset::new(self.domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
        self.bottom_up_properties = (0..vars.len() + 1).map(|_| {
            vec![SparseBitset::new(self.domains.iter().copied())]
        }).collect::<Vec<Vec<SparseBitset<isize>>>>();
    }

    fn update_variable_ordering(&mut self, ordering: &[usize]) {
        self.layer_x = ordering[self.x.0];
        self.layer_y = ordering[self.y.0];
    }

    fn reset_property_top_down(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.top_down_properties[layer][index].reset(0);
    }

    fn update_property_top_down(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize)  {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        if self.is_layer_in_scope(source_layer) {
            self.top_down_properties[target_layer][target_index].insert(assignment);
        }
        let (td_properties_above, td_properties_below) = self.top_down_properties.split_at_mut(target_layer);
        td_properties_below[0][target_index].union(&td_properties_above[source_layer][source_index]);
    }

    fn reset_property_bottom_up(&mut self, node: NodeIndex) {
        let NodeIndex(layer, index) = node;
        self.bottom_up_properties[layer][index].reset(0);
    }

    fn update_property_bottom_up(&mut self, source: NodeIndex, target: NodeIndex, assignment: isize) {
        let NodeIndex(source_layer, source_index) = source;
        let NodeIndex(target_layer, target_index) = target;
        if self.is_layer_in_scope(target_layer) {
            self.bottom_up_properties[target_layer][target_index].insert(assignment);
        }
        let (bu_properties_above, bu_properties_below) = self.bottom_up_properties.split_at_mut(source_layer);
        bu_properties_above[target_layer][target_index].union(&bu_properties_below[0][source_index]);
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        layer == self.layer_x || layer == self.layer_y
    }

    fn is_assignment_invalid(&self, source: NodeIndex, _target: NodeIndex, decision: VariableIndex, assignment: isize) -> bool {
        let NodeIndex(source_layer, source_index) = source;

        if decision == self.x {
            if self.layer_x < self.layer_y {
                self.bottom_up_properties[source_layer][source_index].contains(assignment) && self.bottom_up_properties[source_layer][source_index].size() == 1
            } else {
                self.top_down_properties[source_layer][source_index].contains(assignment) && self.top_down_properties[source_layer][source_index].size() == 1
            }
        } else if self.layer_x > self.layer_y {
            self.bottom_up_properties[source_layer][source_index].contains(assignment) && self.bottom_up_properties[source_layer][source_index].size() == 1
        } else {
            self.top_down_properties[source_layer][source_index].contains(assignment) && self.top_down_properties[source_layer][source_index].size() == 1
        }
    }

    fn add_node_in_layer(&mut self, layer: usize) {
        let top_down_property = SparseBitset::new(self.domains.iter().copied());
        let bottom_up_property = SparseBitset::new(self.domains.iter().copied());
        self.top_down_properties[layer].push(top_down_property);
        self.bottom_up_properties[layer].push(bottom_up_property);
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new([self.x, self.y].into_iter())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        assignment[*self.x] != assignment[*self.y]
    }

    fn hash_node_state(&self, node: NodeIndex, state: &mut dyn Hasher) {
        let NodeIndex(layer, index) = node;
        for word in self.top_down_properties[layer][index].words().iter().copied() {
            state.write_u64(word);
        }
        for word in self.bottom_up_properties[layer][index].words().iter().copied() {
            state.write_u64(word);
        }
    }

    fn eq_node_state(&self, node: NodeIndex, other: NodeIndex) -> bool {
        let NodeIndex(layer, index) = node;
        let NodeIndex(olayer, oindex) = other;
        self.top_down_properties[layer][index] == self.top_down_properties[olayer][oindex] &&
        self.bottom_up_properties[layer][index] == self.bottom_up_properties[olayer][oindex]
    }
}

#[cfg(test)]
mod test_not_equals {

    use crate::modelling::*;
    use crate::constraints::{NotEquals, Constraint};
    use crate::mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;

    // --- is_satisfied: pure logic, no MDD involved --- //

    #[test]
    pub fn test_is_satisfied_different() {
        let ne = NotEquals::new(VariableIndex(0), VariableIndex(1));
        assert!(ne.is_satisfied(&[0, 1]));
    }

    #[test]
    pub fn test_is_satisfied_equal() {
        let ne = NotEquals::new(VariableIndex(0), VariableIndex(1));
        assert!(!ne.is_satisfied(&[3, 3]));
    }

    // --- MDD construction / propagation tests --- //

    #[test]
    pub fn test_basic_filtering() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 2);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_larger_domain() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    expected.push(vec![a, b]);
                }
            }
        }
        assert_eq!(solutions.len(), expected.len());
        for sol in expected {
            assert!(is_solution(sol, &solutions));
        }
    }

    #[test]
    pub fn test_bottom_up_disjoint_singletons_is_sat() {
        // Regression test for a bug in `update_property_bottom_up`: it checked
        // `is_layer_in_scope` on the *child's* layer instead of the current node's layer, and
        // mutated the child's own already-computed property in place instead of the current
        // node's. With x and y forced to *different* singleton values, the constraint is
        // trivially satisfiable (x != y already holds) - but the buggy state folded x's value
        // into what should have been y's downstream property, making the root's aggregated
        // bottom-up set look like a singleton {0}. That incorrectly triggered removal of the
        // only edge out of the root, reporting the whole problem as UNSAT.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![1], None);
        not_equals(&mut problem, x, y);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1]), MergeHeuristic::LessRelaxed);
        assert!(!mdd.is_unsat());
        assert_eq!(mdd.get_solution(), Some(vec![0, 1]));
    }

    #[test]
    pub fn test_bottom_up_disjoint_singletons_is_sat_reversed_order() {
        // Same as above but with y decided before x, exercising the symmetric code path.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0], None);
        let y = problem.add_variable(vec![1], None);
        not_equals(&mut problem, x, y);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![1, 0]), MergeHeuristic::LessRelaxed);
        assert!(!mdd.is_unsat());
        assert_eq!(mdd.get_solution(), Some(vec![0, 1]));
    }

    #[test]
    pub fn test_bottom_up_matching_singletons_is_unsat() {
        // Sanity check in the other direction: if x and y are both forced to the *same*
        // singleton value, the problem genuinely is unsat. Guards against overcorrecting the
        // fix above into never detecting a real conflict.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![5], None);
        let y = problem.add_variable(vec![5], None);
        not_equals(&mut problem, x, y);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1]), MergeHeuristic::LessRelaxed);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_unsat_singleton_conflict() {
        // Both variables are forced to the same single value.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![5], None);
        let y = problem.add_variable(vec![5], None);
        not_equals(&mut problem, x, y);

        let mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        assert!(mdd.is_unsat());
        assert_eq!(mdd.get_solution(), None);
    }

    #[test]
    pub fn test_order_x_before_y() {
        // Forces layer_x < layer_y, exercising the "decision == x, layer_x < layer_y" and
        // the final-else "decision == y, layer_x < layer_y" branches of is_assignment_invalid.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1]), MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 6);
        for sol in solutions.iter() {
            assert_ne!(sol[0], sol[1]);
        }
    }

    #[test]
    pub fn test_order_y_before_x() {
        // Forces layer_x > layer_y, exercising the other two branches of
        // is_assignment_invalid (the ones guarded by `layer_x > layer_y`).
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![1, 0]), MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);
        assert_eq!(solutions.len(), 6);
        for sol in solutions.iter() {
            assert_ne!(sol[0], sol[1]);
        }
    }

    #[test]
    pub fn test_relaxed_width_is_superset() {
        // With a max width of 1 and no refine step, the MDD is a relaxation: it must not
        // exclude any valid solution.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);

        let mdd = Mdd::new(problem, 1, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
        let solutions = get_all_solutions(&mdd);
        assert!(is_solution(vec![0, 1], &solutions));
        assert!(is_solution(vec![1, 0], &solutions));
    }

    #[test]
    pub fn test_chained_not_equals_matches_all_different() {
        // x != y, y != z, x != z on a 3-value domain is equivalent to all_different(x, y, z):
        // only the 3! = 6 permutations of {0, 1, 2} should remain.
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        let z = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        not_equals(&mut problem, y, z);
        not_equals(&mut problem, x, z);

        let mut mdd = Mdd::new(problem, usize::MAX, OrderingHeuristic::Custom(vec![0, 1, 2]), MergeHeuristic::LessRelaxed);
        mdd.refine();
        let solutions = get_all_solutions(&mdd);

        let mut expected: Vec<Vec<isize>> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    if a != b && b != c && a != c {
                        expected.push(vec![a, b, c]);
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
}
