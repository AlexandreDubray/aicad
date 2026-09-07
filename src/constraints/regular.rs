use super::*;
use crate::modelling::VariableIndex;
use crate::utils::Bitset;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, deepsize::DeepSizeOf)]
struct RegularProperty {
    val_to_symbol: Arc<FxHashMap<isize, usize>>,
    transitions: Arc<Vec<Vec<Option<usize>>>>,
    num_states: usize,
    states: Bitset,
}

impl RegularProperty {
    fn new(
        num_states: usize,
        val_to_symbol: Arc<FxHashMap<isize, usize>>,
        transitions: Arc<Vec<Vec<Option<usize>>>>,
        seed_states: &[usize],
    ) -> Self {
        let mut states = Bitset::new(num_states);
        for &state in seed_states {
            states.insert(state);
        }
        Self {
            val_to_symbol,
            transitions,
            num_states,
            states,
        }
    }
}

/// A regular constraint on the variables in its scope. a regular constraint imposes that the
/// assignments must be accepted by the automaton defined in the constraint. This is per nature a
/// sequence variable; hence, when imposed, the variables *MUST* appear in the same order in the
/// compiled MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Regular {
    /// Scope of the constraint
    variables: Vec<VariableIndex>,
    /// Maps value in the variables' domain to their bit representation
    val_to_symbol: Arc<FxHashMap<isize, usize>>,
    /// Transition matrix of the automata
    transitions: Arc<Vec<Vec<Option<usize>>>>,
    /// Number of states. First dimension of transitions
    num_states: usize,
    /// Initial state (0 <= initial_state < num_states)
    initial_state: usize,
    /// Set of accepting states (each state s 0 <= s < num_states)
    accepting_states: FxHashSet<usize>,
    /// Indicate which layers are in the constraint scope
    layer_in_scope: Vec<u64>,
}

impl Regular {
    pub fn new(
        variables: Vec<VariableIndex>,
        transitions: Vec<Vec<Option<usize>>>,
        initial_state: usize,
        accepting_states: FxHashSet<usize>,
        problem: &Problem,
    ) -> Self {
        let mut alphabet_set = FxHashSet::<isize>::default();
        for &variable in variables.iter() {
            alphabet_set.extend(problem[variable].iter_domain());
        }
        let mut alphabet: Vec<isize> = alphabet_set.into_iter().collect();
        alphabet.sort_unstable();
        let num_states = transitions.len();
        let val_to_symbol = Arc::new(
            alphabet
                .into_iter()
                .enumerate()
                .map(|(symbol, value)| (value, symbol))
                .collect(),
        );
        Self {
            variables,
            val_to_symbol,
            transitions: Arc::new(transitions),
            num_states,
            initial_state,
            accepting_states,
            layer_in_scope: vec![],
        }
    }
}

impl Constraint for Regular {
    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        self.layer_in_scope = (0..(order.len() / 64 + 1)).map(|_| 0).collect::<Vec<u64>>();
        let mut scope_layers: Vec<(usize, VariableIndex)> =
            Vec::with_capacity(self.variables.len());
        for (layer, &variable) in order.iter().enumerate() {
            if scope.contains(&variable) {
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                scope_layers.push((layer, variable));
            }
        }
        scope_layers.sort_by_key(|&(layer, _)| layer);
        let observed_order: Vec<VariableIndex> = scope_layers
            .into_iter()
            .map(|(_, variable)| variable)
            .collect();
        if observed_order != self.variables {
            panic!("Regular constraint's variables must keep their declared relative order in the chosen variable ordering");
        }
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        let mut state = self.initial_state;
        for &variable in self.variables.iter() {
            let value = assignment[*variable];
            let symbol = match self.val_to_symbol.get(&value) {
                Some(&symbol) => symbol,
                None => return false,
            };
            match self.transitions[state][symbol] {
                Some(next) => state = next,
                None => return false,
            }
        }
        self.accepting_states.contains(&state)
    }

    fn name(&self) -> &'static str {
        "Regular"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        _layer: usize,
        assignment: isize,
    ) -> bool {
        let parent = parent.as_any().downcast_ref::<RegularProperty>().unwrap_or_else(|| {
            panic!(
                "Calling is_assignment_invalid on parent property of type {} instead of RegularProperty",
                parent.name()
            );
        });
        let child = child.as_any().downcast_ref::<RegularProperty>().unwrap_or_else(|| {
            panic!(
                "Calling is_assignment_invalid on child property of type {} instead of RegularProperty",
                child.name()
            );
        });

        let symbol = *self.val_to_symbol.get(&assignment).unwrap();
        for state in 0..self.num_states {
            if !parent.states.contains(state) {
                continue;
            }
            if let Some(next) = self.transitions[state][symbol] {
                if child.states.contains(next) {
                    return false;
                }
            }
        }
        true
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(RegularProperty::new(
            self.num_states,
            self.val_to_symbol.clone(),
            self.transitions.clone(),
            &[],
        ))
    }

    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(RegularProperty::new(
            self.num_states,
            self.val_to_symbol.clone(),
            self.transitions.clone(),
            &[self.initial_state],
        ))
    }

    fn empty_property_backward(&self) -> Box<dyn ConstraintProperty> {
        let accepting: Vec<usize> = self.accepting_states.iter().copied().collect();
        Box::new(RegularProperty::new(
            self.num_states,
            self.val_to_symbol.clone(),
            self.transitions.clone(),
            &accepting,
        ))
    }
}

impl ConstraintProperty for RegularProperty {
    fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let other = other
            .as_any()
            .downcast_ref::<RegularProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        if in_scope {
            let symbol = *self.val_to_symbol.get(&assignment).unwrap();
            for state in 0..self.num_states {
                if !other.states.contains(state) {
                    continue;
                }
                if let Some(next) = self.transitions[state][symbol] {
                    self.states.insert(next);
                }
            }
        } else {
            self.states.union(&other.states);
        }
    }

    fn update_backward(
        &mut self,
        other: &dyn ConstraintProperty,
        assignment: isize,
        in_scope: bool,
    ) {
        let other = other
            .as_any()
            .downcast_ref::<RegularProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling update_backward on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        if in_scope {
            let symbol = *self.val_to_symbol.get(&assignment).unwrap();
            for state in 0..self.num_states {
                if let Some(next) = self.transitions[state][symbol] {
                    if other.states.contains(next) {
                        self.states.insert(state);
                    }
                }
            }
        } else {
            self.states.union(&other.states);
        }
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = other
            .as_any()
            .downcast_ref::<RegularProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling merge on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });

        self.states.union(&other.states);
    }

    fn order_key(&self) -> Vec<f64> {
        vec![self.states.size() as f64]
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        for word in self.states.iter() {
            hasher.write_u64(word);
        }
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = other
            .as_any()
            .downcast_ref::<RegularProperty>()
            .unwrap_or_else(|| {
                panic!(
                    "Calling eq on property {} with other property of type {}",
                    self.name(),
                    other.name()
                );
            });
        self.states == other.states
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "RegularProperty"
    }
}

#[cfg(test)]
mod test_regular {

    use crate::constraints::{Constraint, Regular};
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHashSet;
    use std::sync::Arc;

    fn no_one_then_zero_automaton() -> (Vec<Vec<Option<usize>>>, usize, FxHashSet<usize>) {
        let transitions = vec![
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
            vec![None, Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
        ];
        let accepting = FxHashSet::from_iter([0, 1, 2, 3]);
        (transitions, 0, accepting)
    }

    fn brute_force_no_one_then_zero(d: usize) -> Vec<Vec<isize>> {
        fn go(d: usize, current: &mut Vec<isize>, all: &mut Vec<Vec<isize>>) {
            if current.len() == d {
                all.push(current.clone());
                return;
            }
            for value in 0..3isize {
                if current.last() == Some(&1) && value == 0 {
                    continue;
                }
                current.push(value);
                go(d, current, all);
                current.pop();
            }
        }
        let mut all = vec![];
        go(d, &mut vec![], &mut all);
        all
    }

    #[test]
    pub fn is_satisfied_matches_the_automaton() {
        let (transitions, initial, accepting) = no_one_then_zero_automaton();
        let vars = vec![VariableIndex(0), VariableIndex(1), VariableIndex(2)];
        let mut problem = Problem::default();
        for _ in 0..3 {
            problem.add_variable((0..3isize).collect(), None);
        }
        let constraint = Regular::new(vars, transitions, initial, accepting, &problem);
        assert!(constraint.is_satisfied(&[0, 1, 2]));
        assert!(!constraint.is_satisfied(&[1, 0, 2]));
        assert!(constraint.is_satisfied(&[2, 2, 2]));
    }

    #[test]
    pub fn exact_compilation_matches_the_automatons_language() {
        let d = 5;
        let (transitions, initial, accepting) = no_one_then_zero_automaton();
        let vars: Vec<_> = (0..d).map(VariableIndex).collect();
        let mut problem = Problem::default();
        for _ in 0..d {
            problem.add_variable((0..3isize).collect(), None);
        }
        regular(
            &mut problem,
            vars,
            transitions,
            initial,
            accepting.into_iter().collect(),
        );
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom((0..d).collect()),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        mdd.refine(usize::MAX);

        let mdd_solutions = get_all_solutions(&mdd);
        let true_solutions = brute_force_no_one_then_zero(d);

        assert_eq!(mdd_solutions.len(), true_solutions.len());
        for solution in true_solutions {
            assert!(
                is_solution(solution.clone(), &mdd_solutions),
                "missing true solution {:?}",
                solution
            );
        }
        for solution in mdd_solutions {
            assert!(
                is_solution(solution.clone(), &brute_force_no_one_then_zero(d)),
                "mdd accepts non-solution {:?}",
                solution
            );
        }
    }

    #[test]
    pub fn merged_relaxed_mdd_still_contains_every_true_regular_solution() {
        let d = 5;
        for max_width in [1usize, 2, 3] {
            let (transitions, initial, accepting) = no_one_then_zero_automaton();
            let vars: Vec<_> = (0..d).map(VariableIndex).collect();
            let mut problem = Problem::default();
            for _ in 0..d {
                problem.add_variable((0..3isize).collect(), None);
            }
            regular(
                &mut problem,
                vars,
                transitions,
                initial,
                accepting.into_iter().collect(),
            );
            let problem = Arc::new(problem);
            let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
            let mut mdd = Mdd::new(
                problem,
                OrderingHeuristic::Custom((0..d).collect()),
                MergeHeuristic::LessRelaxed,
                SelectHeuristic::Greedy,
                &constraints,
            );
            mdd.refine(max_width);
            let candidate_solutions = get_all_solutions(&mdd);
            for solution in brute_force_no_one_then_zero(d) {
                assert!(
                    is_solution(solution.clone(), &candidate_solutions),
                    "max_width={} true regular solution {:?} missing from relaxed mdd",
                    max_width,
                    solution
                );
            }
        }
    }
}
