//! `Regular`: the sequence of values taken by the scope is a word accepted by a DFA.
//!
//! # Meaning
//!
//! Given a deterministic automaton with states $1, \dots, q$, an initial state $s_1$, a set of
//! accepting states $F$ and a transition function $\delta : \{1..q\} \times \Sigma \rightarrow
//! \{1..q\} \cup \{\bot\}$, constrains $x_1, \dots, x_n$ so that the word $x_1 \cdots x_n$ is
//! accepted, that is so that
//!
//! $$\delta(\dots\delta(\delta(s_1, x_1), x_2) \dots, x_n) \in F,$$
//!
//! no transition on the way being $\bot$.
//!
//! The automaton is given as a table `transitions[state][symbol]`, with `None` standing for
//! $\bot$. **The columns are the symbols of the alphabet in increasing order of value**: the
//! alphabet $\Sigma$ is the set of values found in the domains of the scope when
//! [`Regular::new`] is called, sorted, and column $j$ is its $j$-th smallest value. With the
//! domain $\{0, 1, 2\}$ column $j$ is simply the value $j$; with $\{-1, 4, 7\}$ column 0 is $-1$,
//! column 1 is 4 and column 2 is 7. A value met later that is not in $\Sigma$ (a domain that grew
//! after construction) has no column: it is rejected by every state, exactly as
//! [`is_satisfied`](Constraint::is_satisfied) rejects it.
//!
//! The word is read in the order the scope was **declared**, not in the order of the layers of the
//! MDD. The constraint therefore asks the compilation to branch the variables of the scope in the
//! declared order (see [`precedence_edges`](Constraint::precedence_edges)); other variables may be
//! interleaved freely.
//!
//! With an empty scope the word is empty, so the constraint holds exactly when the initial state
//! is accepting. As for [`Sum`], the MDD has no layer at which to filter anything, so an empty
//! `Regular` is not seen by the compilation.
//!
//! A variable may appear only once in the scope, and [`Regular::new`] panics otherwise; it also
//! panics on an automaton that is not well formed (see its documentation).
//!
//! # How it compiles
//!
//! A node carries a set of automaton states.
//!
//! - **Top-down**, the set $T$ of the states in which the automaton can be after reading the
//!   scope variables of some path from the root to the node. The root holds $\{s_1\}$.
//! - **Bottom-up**, the set $B$ of the states from which the automaton can reach an accepting
//!   state by reading the scope variables of some path from the node to the sink. The sink holds
//!   $F$.
//!
//! A node with no path holds the empty set, so there is no sentinel to recognise.
//!
//! ## Property update
//!
//! On an edge of the scope with value $v$, the top-down set of the child gets
//! $\{\delta(s, v) \mid s \in T_{parent}\}$ and the bottom-up set of the parent gets
//! $\{s \mid \delta(s, v) \in B_{child}\}$. Out of scope layers copy the set. When a node has
//! several parents (or children) the results are united.
//!
//! ## Node merging
//!
//! Merging two nodes takes the union of their sets. The merged node accepts the paths of both, and
//! also the paths that mix a prefix of one with a suffix of the other when the sets overlap, so it
//! over-approximates them.
//!
//! ## Edge filtering
//!
//! An edge with value $v$ from a parent $T$ to a child $B$ is removed when
//! no state $s \in T$ has $\delta(s, v) \in B$. In an MDD whose nodes are not merged, $T$ and $B$
//! are exact, and the converse holds: the automaton is deterministic, so a prefix reaching $s$
//! and a suffix leaving $\delta(s, v)$ can always be glued through the edge. The bottom-up sets
//! are computed before the edges are filtered, so one pass can leave sets slightly larger than
//! the fixpoint; the next pass removes what is left.
//!
//! # Design notes
//!
//! - **The automaton is shared**: the transition table and the value to symbol map are in
//!   [`Arc`]s, so the properties of all the nodes point to the same copy.
//! - **The alphabet is frozen at construction.** Domains that shrink afterwards are fine;
//!   a value that appears afterwards is rejected, and does not cause a panic.
//! - **The structural key does not hold the alphabet**, only the table, the initial state and the
//!   accepting states (sorted). The shared templates compiled by the arena assume that the value
//!   at index $j$ of a domain is the symbol $j$, which holds for domains sorted in increasing
//!   order.
//! - **`update_variable_ordering` panics**, in release mode too, if the ordering does not keep
//!   the declared order of the scope. `Mdd::new` repairs the heuristic's order before calling it,
//!   so this is a safety net.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, Regular};
//! use aicad::modelling::*;
//! use rustc_hash::FxHashSet;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1], None);
//! // "No two consecutive 1": state 1 means that the last value was a 1.
//! let transitions = vec![
//!     vec![Some(0), Some(1)],
//!     vec![Some(0), None],
//! ];
//! let regular = Regular::new(vars, transitions, 0, FxHashSet::from_iter([0, 1]), &problem);
//!
//! assert!(regular.is_satisfied(&[1, 0, 1]));
//! assert!(!regular.is_satisfied(&[0, 1, 1]));
//! ```
use super::*;
use crate::modelling::VariableIndex;
use crate::utils::Bitset;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hasher;
use std::sync::Arc;

/// Per-node state of [`Regular`]: a set of states of the automaton. Top-down, the states reachable
/// by the scope variables of some path from the root; bottom-up, the states from which an
/// accepting state is reachable by the scope variables of some path to the sink. The empty set
/// means that no path has been folded in, which is the identity of `merge`. See the
/// [module documentation](self).
#[derive(Clone, deepsize::DeepSizeOf)]
struct RegularProperty {
    /// Maps a value to the column of the transition table. Shared with the constraint.
    val_to_symbol: Arc<FxHashMap<isize, usize>>,
    /// The transition table `[state][symbol]`. Shared with the constraint.
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

/// The constraint that the scope, read in its declared order, is a word accepted by a DFA. See
/// the [module documentation](self) for the semantics and for how it is compiled into an MDD.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Regular {
    /// Scope of the constraint, without repetition, in the order the automaton reads it.
    variables: Vec<VariableIndex>,
    /// Maps a value of the alphabet to its column in `transitions`
    val_to_symbol: Arc<FxHashMap<isize, usize>>,
    /// Transition table of the automaton: `transitions[state][symbol]`
    transitions: Arc<Vec<Vec<Option<usize>>>>,
    /// Number of states. First dimension of transitions
    num_states: usize,
    /// Initial state (0 <= initial_state < num_states)
    initial_state: usize,
    /// Set of accepting states (each state s 0 <= s < num_states)
    accepting_states: FxHashSet<usize>,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
}

impl Regular {
    /// Builds the constraint that `variables`, read in this order, form a word accepted by the
    /// automaton. `transitions[state][symbol]` is the state reached from `state` on the symbol
    /// of column `symbol` (`None` if there is no transition), where the columns are the sorted
    /// values found in the domains of `variables` at this point.
    ///
    /// # Panics
    ///
    /// - If a variable appears more than once in `variables`.
    /// - If `initial_state` or an accepting state is not a state of the table, or a transition
    ///   leads to a state that is not.
    /// - If a row of the table has fewer columns than there are values in the alphabet. Extra
    ///   columns are ignored.
    pub fn new(
        variables: Vec<VariableIndex>,
        transitions: Vec<Vec<Option<usize>>>,
        initial_state: usize,
        accepting_states: FxHashSet<usize>,
        problem: &Problem,
    ) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        if distinct.len() != variables.len() {
            panic!("Regular does not support a variable repeated in its scope");
        }
        let mut alphabet_set = FxHashSet::<isize>::default();
        for &variable in variables.iter() {
            alphabet_set.extend(problem[variable].iter_domain());
        }
        let mut alphabet: Vec<isize> = alphabet_set.into_iter().collect();
        alphabet.sort_unstable();
        let num_states = transitions.len();
        if initial_state >= num_states {
            panic!(
                "Regular: the initial state {initial_state} is not a state of the automaton ({num_states} states)"
            );
        }
        if !accepting_states.iter().all(|&state| state < num_states) {
            panic!(
                "Regular: an accepting state is not a state of the automaton ({num_states} states)"
            );
        }
        for (state, row) in transitions.iter().enumerate() {
            if row.len() < alphabet.len() {
                panic!(
                    "Regular: state {state} has {} transitions for an alphabet of {} values",
                    row.len(),
                    alphabet.len()
                );
            }
            if !row.iter().flatten().all(|&next| next < num_states) {
                panic!(
                    "Regular: a transition of state {state} leads outside the automaton ({num_states} states)"
                );
            }
        }
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
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        let mut accepting_states: Vec<usize> = self.accepting_states.iter().copied().collect();
        accepting_states.sort_unstable();
        ConstraintShapeKey::Regular {
            arity: self.variables.len(),
            transitions: (*self.transitions).clone(),
            initial_state: self.initial_state,
            accepting_states,
        }
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        self.layer_in_scope = vec![0; order.len() / 64 + 1];
        let mut observed_order: Vec<VariableIndex> = Vec::with_capacity(self.variables.len());
        for (layer, &variable) in order.iter().enumerate() {
            if scope.contains(&variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                observed_order.push(variable);
            }
        }
        // `Mdd::new` repairs the heuristic's order to satisfy `precedence_edges` (below) before
        // calling this, so this should be unreachable in practice -- kept as a safety net (e.g.
        // for a `Regular` compiled outside `Mdd::new`'s repair pass) rather than silently
        // building a `layer_in_scope` bitset that doesn't match the automaton's actual sequence.
        if observed_order != self.variables {
            panic!(
                "Regular constraint's variables must keep their declared relative order in the chosen variable ordering"
            );
        }
    }

    /// The automaton is walked in the exact sequence `self.variables` was declared in, so each
    /// consecutive pair must be branched in that order: variable `k` before variable `k+1`.
    fn precedence_edges(&self) -> Vec<(VariableIndex, VariableIndex)> {
        self.variables
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .collect()
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(
            !self.layer_in_scope.is_empty(),
            "update_variable_ordering has not been called"
        );
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

        // A value outside the alphabet is read by no transition.
        let Some(&symbol) = self.val_to_symbol.get(&assignment) else {
            return true;
        };
        for state in 0..self.num_states {
            if !parent.states.contains(state) {
                continue;
            }
            if let Some(next) = self.transitions[state][symbol]
                && child.states.contains(next)
            {
                return false;
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
            // A value outside the alphabet is read by no transition: nothing is reached.
            let Some(&symbol) = self.val_to_symbol.get(&assignment) else {
                return;
            };
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
            // A value outside the alphabet is read by no transition: no state leads anywhere.
            let Some(&symbol) = self.val_to_symbol.get(&assignment) else {
                return;
            };
            for state in 0..self.num_states {
                if let Some(next) = self.transitions[state][symbol]
                    && other.states.contains(next)
                {
                    self.states.insert(state);
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
    use super::RegularProperty;
    use crate::constraints::{
        AllDifferent, Constraint, ConstraintProperty, ConstraintShapeKey, Regular,
    };
    use crate::mdd::heuristics::*;
    use crate::mdd::mdd::test_mdd::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::{FxHashSet, FxHasher};
    use std::hash::Hasher;
    use std::sync::Arc;

    // ----------------------------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------------------------

    type Table = Vec<Vec<Option<usize>>>;

    fn dom(n: isize) -> Vec<isize> {
        (0..n).collect()
    }

    fn set(states: &[usize]) -> FxHashSet<usize> {
        states.iter().copied().collect()
    }

    /// "No two consecutive 1" over {0, 1}: state 1 means that the last value was a 1.
    fn no_two_ones() -> Table {
        vec![vec![Some(0), Some(1)], vec![Some(0), None]]
    }

    /// "A 1 is never followed by a 0" over {0, 1, 2}. State 2 means that the last value was a 1.
    /// Every state accepts.
    fn no_one_then_zero() -> Table {
        vec![
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
            vec![None, Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
        ]
    }

    /// A problem with one variable per domain, and the constraint over the variables of `scope`
    /// (indices into `domains`, in the order the automaton reads them).
    fn build(
        domains: &[Vec<isize>],
        scope: &[usize],
        table: &Table,
        initial: usize,
        accepting: &[usize],
    ) -> (Problem, Vec<VariableIndex>, Regular) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        let constraint = Regular::new(
            scope.iter().map(|&i| vars[i]).collect(),
            table.clone(),
            initial,
            set(accepting),
            &problem,
        );
        (problem, vars, constraint)
    }

    /// The constraint with the identity ordering already set, over `n` variables whose domain
    /// has one value per column of the table.
    fn over(n: usize, table: &Table, initial: usize, accepting: &[usize]) -> Regular {
        let domains = vec![dom(table[0].len() as isize); n];
        let scope: Vec<usize> = (0..n).collect();
        let (_, vars, mut constraint) = build(&domains, &scope, table, initial, accepting);
        constraint.update_variable_ordering(&vars);
        constraint
    }

    /// Adds the constraint to the problem, and compiles with the given variable order. `width`
    /// is the refinement budget: `1` keeps the initial relaxation.
    fn compile(problem: Problem, order: Vec<usize>, width: usize) -> Mdd {
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut mdd = Mdd::new(
            problem,
            OrderingHeuristic::Custom(order),
            MergeHeuristic::LessRelaxed,
            SelectHeuristic::Greedy,
            &constraints,
        );
        if width > 1 {
            mdd.refine(width);
        }
        mdd
    }

    /// Like `compile` with `usize::MAX`, then runs the propagation again until nothing changes.
    ///
    /// `Mdd::refine` propagates only one pass after its last split, so an MDD can keep paths that
    /// a second pass would remove (see the task "MDD propagation is not iterated to a
    /// fixpoint"). The exactness tests need the fixpoint, because they check what the constraint
    /// rules out, not how many passes the engine runs.
    fn settled(problem: Problem, order: Vec<usize>) -> Mdd {
        let mut mdd = compile(problem, order, usize::MAX);
        for _ in 0..10 {
            if mdd.is_unsat() {
                break;
            }
            mdd.propagate_constraints();
        }
        mdd
    }

    fn accepted(mdd: &Mdd) -> Vec<Vec<isize>> {
        if mdd.is_unsat() {
            return vec![];
        }
        let mut solutions = get_all_solutions(mdd);
        solutions.sort();
        solutions
    }

    /// All tuples of the cartesian product of the domains, sorted.
    fn product(domains: &[Vec<isize>]) -> Vec<Vec<isize>> {
        let mut tuples: Vec<Vec<isize>> = vec![vec![]];
        for domain in domains {
            let mut next = vec![];
            for prefix in &tuples {
                for &value in domain {
                    let mut tuple = prefix.clone();
                    tuple.push(value);
                    next.push(tuple);
                }
            }
            tuples = next;
        }
        tuples
    }

    /// Whether the DFA accepts the word `scope.map(|i| tuple[i])`, written independently of
    /// `Regular`: the alphabet is passed explicitly and a symbol is looked up by position.
    fn runs(
        table: &Table,
        initial: usize,
        accepting: &[usize],
        alphabet: &[isize],
        word: &[isize],
    ) -> bool {
        let mut state = initial;
        for value in word {
            let Some(symbol) = alphabet.iter().position(|a| a == value) else {
                return false;
            };
            match table[state][symbol] {
                Some(next) => state = next,
                None => return false,
            }
        }
        accepting.contains(&state)
    }

    /// The sorted union of the domains of the variables of `scope`.
    fn alphabet_of(domains: &[Vec<isize>], scope: &[usize]) -> Vec<isize> {
        let mut alphabet: Vec<isize> = scope.iter().flat_map(|&i| domains[i].clone()).collect();
        alphabet.sort_unstable();
        alphabet.dedup();
        alphabet
    }

    /// The expected solutions: the tuples whose scope word is accepted by the DFA.
    fn expected(
        domains: &[Vec<isize>],
        scope: &[usize],
        table: &Table,
        initial: usize,
        accepting: &[usize],
    ) -> Vec<Vec<isize>> {
        let alphabet = alphabet_of(domains, scope);
        let mut tuples = product(domains);
        tuples.retain(|t| {
            let word: Vec<isize> = scope.iter().map(|&i| t[i]).collect();
            runs(table, initial, accepting, &alphabet, &word)
        });
        tuples.sort();
        tuples
    }

    /// The states in the set of a property, in increasing order.
    fn states_of(property: &dyn ConstraintProperty) -> Vec<usize> {
        let property = property.as_any().downcast_ref::<RegularProperty>().unwrap();
        (0..property.num_states)
            .filter(|&s| property.states.contains(s))
            .collect()
    }

    /// A property holding exactly `states`, obtained from a constraint with enough states.
    fn with_states(constraint: &Regular, states: &[usize]) -> Box<dyn ConstraintProperty> {
        let mut property = RegularProperty::new(
            constraint.num_states,
            constraint.val_to_symbol.clone(),
            constraint.transitions.clone(),
            states,
        );
        property.states = {
            let mut s = crate::utils::Bitset::new(constraint.num_states);
            for &state in states {
                s.insert(state);
            }
            s
        };
        Box::new(property)
    }

    fn same(a: &dyn ConstraintProperty, b: &dyn ConstraintProperty) -> bool {
        ConstraintProperty::eq(a, b)
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    fn foreign() -> Box<dyn ConstraintProperty> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        AllDifferent::new(vars, &problem).identity_property()
    }

    /// The tiny deterministic generator used for the random instances below.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
    }

    /// One random instance.
    struct Instance {
        domains: Vec<Vec<isize>>,
        scope: Vec<usize>,
        table: Table,
        initial: usize,
        accepting: Vec<usize>,
    }

    /// A random instance: up to 5 variables with domains over {-1, 0, 1, 2}, a scope made of some
    /// of them in a random reading order, and a random partial DFA of 1 to 4 states.
    fn random_instance(rng: &mut Lcg) -> Instance {
        let n = 1 + rng.below(5) as usize;
        let domains: Vec<Vec<isize>> = (0..n)
            .map(|_| {
                let mut d: Vec<isize> = (-1..3).filter(|_| rng.below(2) == 1).collect();
                if d.is_empty() {
                    d.push(rng.below(4) as isize - 1);
                }
                d
            })
            .collect();
        let mut scope: Vec<usize> = (0..n).filter(|_| rng.below(4) != 0).collect();
        if scope.is_empty() {
            scope.push(rng.below(n as u64) as usize);
        }
        for i in (1..scope.len()).rev() {
            scope.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let q = 1 + rng.below(4) as usize;
        let columns = alphabet_of(&domains, &scope).len();
        let table: Table = (0..q)
            .map(|_| {
                (0..columns)
                    .map(|_| {
                        if rng.below(5) == 0 {
                            None
                        } else {
                            Some(rng.below(q as u64) as usize)
                        }
                    })
                    .collect()
            })
            .collect();
        let accepting: Vec<usize> = (0..q).filter(|_| rng.below(2) == 1).collect();
        Instance {
            domains,
            scope,
            table,
            initial: rng.below(q as u64) as usize,
            accepting,
        }
    }

    /// A trivially satisfied automaton over `vars`: one state that loops on every value. It only
    /// pulls the variables into the MDD, which otherwise holds the variables of the constraints
    /// it compiles and nothing else.
    fn add_free(problem: &mut Problem, vars: Vec<VariableIndex>) {
        if vars.is_empty() {
            return;
        }
        let mut alphabet: Vec<isize> = vars
            .iter()
            .flat_map(|&v| problem[v].iter_domain().collect::<Vec<_>>())
            .collect();
        alphabet.sort_unstable();
        alphabet.dedup();
        regular(
            problem,
            vars,
            vec![vec![Some(0); alphabet.len()]],
            0,
            vec![0],
        );
    }

    /// The problem of an instance: its variables, the automaton on its scope, and a trivial
    /// automaton on the variables that are not in the scope so that they are part of the MDD.
    fn problem_of(i: &Instance) -> (Problem, Vec<VariableIndex>) {
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = i
            .domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        regular(
            &mut problem,
            i.scope.iter().map(|&k| vars[k]).collect(),
            i.table.clone(),
            i.initial,
            i.accepting.clone(),
        );
        let free: Vec<VariableIndex> = (0..vars.len())
            .filter(|k| !i.scope.contains(k))
            .map(|k| vars[k])
            .collect();
        add_free(&mut problem, free);
        (problem, vars)
    }

    /// A random order in which to branch the variables: the scope in its declared order and the
    /// other variables in increasing order (the order of their trivial automaton), interleaved.
    fn order_for(i: &Instance, rng: &mut Lcg) -> Vec<usize> {
        let mut scope = i.scope.clone().into_iter().peekable();
        let free: Vec<usize> = (0..i.domains.len())
            .filter(|k| !i.scope.contains(k))
            .collect();
        let mut free = free.into_iter().peekable();
        let mut order = vec![];
        while scope.peek().is_some() || free.peek().is_some() {
            let take_scope = match (scope.peek(), free.peek()) {
                (Some(_), Some(_)) => rng.below(2) == 0,
                (Some(_), None) => true,
                _ => false,
            };
            order.push(if take_scope {
                scope.next().unwrap()
            } else {
                free.next().unwrap()
            });
        }
        order
    }

    // ----------------------------------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_rejected() {
        let mut problem = Problem::default();
        let x = problem.add_variable(dom(2), None);
        let y = problem.add_variable(dom(2), None);
        Regular::new(vec![x, y, x], no_two_ones(), 0, set(&[0]), &problem);
    }

    #[test]
    #[should_panic(expected = "initial state")]
    fn an_initial_state_outside_the_automaton_is_rejected() {
        build(&[dom(2)], &[0], &no_two_ones(), 2, &[0]);
    }

    #[test]
    #[should_panic(expected = "accepting state")]
    fn an_accepting_state_outside_the_automaton_is_rejected() {
        build(&[dom(2)], &[0], &no_two_ones(), 0, &[0, 2]);
    }

    #[test]
    #[should_panic(expected = "leads outside")]
    fn a_transition_to_a_state_outside_the_automaton_is_rejected() {
        let table = vec![vec![Some(0), Some(5)], vec![None, None]];
        build(&[dom(2)], &[0], &table, 0, &[0]);
    }

    #[test]
    #[should_panic(expected = "transitions for an alphabet")]
    fn a_row_shorter_than_the_alphabet_is_rejected() {
        let table = vec![vec![Some(0)], vec![Some(0), None]];
        build(&[dom(2)], &[0], &table, 0, &[0]);
    }

    #[test]
    #[should_panic(expected = "initial state")]
    fn an_automaton_without_states_is_rejected() {
        build(&[dom(2)], &[0], &vec![], 0, &[]);
    }

    #[test]
    fn extra_columns_are_ignored() {
        // The alphabet has two values, the table three columns.
        let table = vec![vec![Some(0), Some(0), Some(0)]];
        let (_, _, constraint) = build(&[dom(2), dom(2)], &[0, 1], &table, 0, &[0]);
        assert!(constraint.is_satisfied(&[0, 1]));
        assert!(!constraint.is_satisfied(&[2, 1]));
    }

    #[test]
    fn the_columns_are_the_sorted_values_of_the_domains() {
        // Domains {7, -1} and {4}: the alphabet is -1, 4, 7, in columns 0, 1, 2. Only the word
        // "-1 then 4" is accepted.
        let table = vec![
            vec![Some(1), None, None],
            vec![None, Some(2), None],
            vec![None, None, None],
        ];
        let (_, _, constraint) = build(&[vec![7, -1], vec![4]], &[0, 1], &table, 0, &[2]);
        assert!(constraint.is_satisfied(&[-1, 4]));
        assert!(!constraint.is_satisfied(&[7, 4]));
        assert!(!constraint.is_satisfied(&[4, -1]));
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        let (_, _, constraint) = build(&vec![dom(2); 4], &[0, 1, 2, 3], &no_two_ones(), 0, &[0, 1]);
        assert!(constraint.is_satisfied(&[0, 0, 0, 0]));
        assert!(constraint.is_satisfied(&[1, 0, 1, 0]));
        assert!(constraint.is_satisfied(&[0, 1, 0, 1]));
        assert!(!constraint.is_satisfied(&[0, 1, 1, 0]));
        assert!(!constraint.is_satisfied(&[1, 1, 0, 0]));
    }

    #[test]
    fn only_accepting_states_accept() {
        // The same automaton, accepting only after a 1.
        let (_, _, constraint) = build(&vec![dom(2); 3], &[0, 1, 2], &no_two_ones(), 0, &[1]);
        assert!(constraint.is_satisfied(&[0, 0, 1]));
        assert!(!constraint.is_satisfied(&[1, 0, 0]));
        assert!(!constraint.is_satisfied(&[0, 1, 1]));
    }

    #[test]
    fn is_satisfied_agrees_with_the_dfa_on_random_instances() {
        let mut rng = Lcg(5);
        for _ in 0..300 {
            let i = random_instance(&mut rng);
            let (problem, _, constraint) =
                build(&i.domains, &i.scope, &i.table, i.initial, &i.accepting);
            let alphabet = alphabet_of(&i.domains, &i.scope);
            for tuple in product(&i.domains) {
                let word: Vec<isize> = i.scope.iter().map(|&k| tuple[k]).collect();
                assert_eq!(
                    constraint.is_satisfied(&tuple),
                    runs(&i.table, i.initial, &i.accepting, &alphabet, &word),
                    "{tuple:?}"
                );
            }
            drop(problem);
        }
    }

    #[test]
    fn the_word_is_read_in_the_declared_order() {
        // Scope declared as (v2, v0, v1), and an automaton that accepts only the word 1, 0, 0.
        let table = vec![
            vec![None, Some(1)],
            vec![Some(2), None],
            vec![Some(3), None],
            vec![None, None],
        ];
        let (_, _, constraint) = build(&vec![dom(2); 3], &[2, 0, 1], &table, 0, &[3]);
        // v2 = 1, v0 = 0, v1 = 0
        assert!(constraint.is_satisfied(&[0, 0, 1]));
        // The order of the indices would read v0 = 1 first.
        assert!(!constraint.is_satisfied(&[1, 0, 0]));
    }

    #[test]
    fn variables_out_of_scope_are_ignored() {
        let (_, _, constraint) = build(&vec![dom(2); 3], &[0, 2], &no_two_ones(), 0, &[0, 1]);
        assert!(!constraint.is_satisfied(&[1, 0, 1]));
        assert!(constraint.is_satisfied(&[1, 1, 0]));
    }

    #[test]
    fn a_value_outside_the_alphabet_is_rejected() {
        let (_, _, constraint) = build(&vec![dom(2); 2], &[0, 1], &no_two_ones(), 0, &[0, 1]);
        assert!(!constraint.is_satisfied(&[0, 5]));
        assert!(!constraint.is_satisfied(&[-1, 0]));
    }

    #[test]
    fn an_empty_scope_is_the_empty_word() {
        let accepting = build(&[dom(2)], &[], &no_two_ones(), 0, &[0]).2;
        let rejecting = build(&[dom(2)], &[], &no_two_ones(), 0, &[1]).2;
        assert!(accepting.is_satisfied(&[0]));
        assert!(!rejecting.is_satisfied(&[0]));
    }

    #[test]
    fn no_accepting_state_accepts_nothing() {
        let (_, _, constraint) = build(&vec![dom(2); 2], &[0, 1], &no_two_ones(), 0, &[]);
        for tuple in product(&[dom(2), dom(2)]) {
            assert!(!constraint.is_satisfied(&tuple));
        }
    }

    #[test]
    #[should_panic]
    fn is_satisfied_panics_on_an_assignment_that_is_too_short() {
        over(3, &no_one_then_zero(), 0, &[0, 1, 2, 3]).is_satisfied(&[0, 1]);
    }

    // ----------------------------------------------------------------------------------------
    // Scope, ordering and key
    // ----------------------------------------------------------------------------------------

    #[test]
    fn scope_and_name() {
        let (_, vars, constraint) = build(&vec![dom(2); 3], &[2, 0], &no_two_ones(), 0, &[0]);
        assert_eq!(
            constraint.iter_scope().collect::<Vec<_>>(),
            vec![vars[2], vars[0]]
        );
        assert_eq!(constraint.name(), "Regular");
    }

    #[test]
    fn the_structural_key_describes_the_automaton() {
        // Ten states looping on themselves, accepting a few of them, given out of order.
        let table: Table = (0..10).map(|k| vec![Some(k), Some(k)]).collect();
        let (problem, _, constraint) =
            build(&vec![dom(2); 3], &[0, 1, 2], &table, 4, &[9, 3, 7, 1, 5]);
        assert_eq!(
            constraint.structural_key(&problem),
            ConstraintShapeKey::Regular {
                arity: 3,
                transitions: table,
                initial_state: 4,
                accepting_states: vec![1, 3, 5, 7, 9],
            }
        );
    }

    #[test]
    fn the_structural_key_depends_on_every_part_of_the_automaton() {
        let key = |scope: &[usize], table: Table, initial: usize, accepting: &[usize]| {
            let (problem, _, constraint) =
                build(&vec![dom(2); 3], scope, &table, initial, accepting);
            constraint.structural_key(&problem)
        };
        let base = key(&[0, 1, 2], no_two_ones(), 0, &[0, 1]);
        assert_eq!(base, key(&[0, 1, 2], no_two_ones(), 0, &[1, 0]));
        assert_ne!(base, key(&[0, 1], no_two_ones(), 0, &[0, 1]));
        assert_ne!(base, key(&[0, 1, 2], no_two_ones(), 1, &[0, 1]));
        assert_ne!(base, key(&[0, 1, 2], no_two_ones(), 0, &[0]));
        let mut other = no_two_ones();
        other[1][1] = Some(1);
        assert_ne!(base, key(&[0, 1, 2], other, 0, &[0, 1]));
    }

    #[test]
    fn the_scope_must_be_branched_in_its_declared_order() {
        let (_, vars, constraint) = build(&vec![dom(2); 4], &[2, 0, 3], &no_two_ones(), 0, &[0]);
        assert_eq!(
            constraint.precedence_edges(),
            vec![(vars[2], vars[0]), (vars[0], vars[3])]
        );
    }

    #[test]
    fn a_scope_of_zero_or_one_variable_needs_no_precedence() {
        assert!(
            build(&vec![dom(2); 2], &[], &no_two_ones(), 0, &[0])
                .2
                .precedence_edges()
                .is_empty()
        );
        assert!(
            build(&vec![dom(2); 2], &[1], &no_two_ones(), 0, &[0])
                .2
                .precedence_edges()
                .is_empty()
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        // Scope (v3, v0, v2); the layer order puts v3, v1, v0, v4, v2.
        let (_, vars, mut constraint) =
            build(&vec![dom(2); 5], &[3, 0, 2], &no_two_ones(), 0, &[0]);
        constraint.update_variable_ordering(&[vars[3], vars[1], vars[0], vars[4], vars[2]]);
        let in_scope: Vec<bool> = (0..5).map(|l| constraint.is_layer_in_scope(l)).collect();
        assert_eq!(in_scope, vec![true, false, true, false, true]);
    }

    #[test]
    fn layers_beyond_the_first_word_are_tracked() {
        // 130 variables, the scope is at layers 0, 63, 64, 65 and 129 and read in layer order.
        let mut problem = Problem::default();
        let vars = problem.add_variables(130, dom(2), None);
        let scope_layers = [0usize, 63, 64, 65, 129];
        let scope: Vec<VariableIndex> = scope_layers.iter().map(|&l| vars[l]).collect();
        let mut constraint = Regular::new(scope, no_two_ones(), 0, set(&[0]), &problem);
        constraint.update_variable_ordering(&vars);
        for layer in 0..130 {
            assert_eq!(
                constraint.is_layer_in_scope(layer),
                scope_layers.contains(&layer),
                "layer {layer}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "declared relative order")]
    fn an_ordering_that_reverses_the_scope_is_rejected() {
        let (_, vars, mut constraint) = build(&vec![dom(2); 2], &[0, 1], &no_two_ones(), 0, &[0]);
        constraint.update_variable_ordering(&[vars[1], vars[0]]);
    }

    #[test]
    #[should_panic(expected = "declared relative order")]
    fn an_ordering_that_misses_a_scope_variable_is_rejected() {
        let (_, vars, mut constraint) = build(&vec![dom(2); 2], &[0, 1], &no_two_ones(), 0, &[0]);
        constraint.update_variable_ordering(&[vars[0]]);
    }

    #[test]
    fn a_new_ordering_replaces_the_previous_one() {
        let (_, vars, mut constraint) = build(&vec![dom(2); 3], &[0, 1], &no_two_ones(), 0, &[0]);
        constraint.update_variable_ordering(&[vars[0], vars[1], vars[2]]);
        assert!(
            constraint.is_layer_in_scope(0)
                && constraint.is_layer_in_scope(1)
                && !constraint.is_layer_in_scope(2)
        );
        constraint.update_variable_ordering(&[vars[2], vars[0], vars[1]]);
        assert!(
            !constraint.is_layer_in_scope(0)
                && constraint.is_layer_in_scope(1)
                && constraint.is_layer_in_scope(2)
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "has not been called")]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        build(&vec![dom(2); 2], &[0, 1], &no_two_ones(), 0, &[0])
            .2
            .is_layer_in_scope(0);
    }

    // ----------------------------------------------------------------------------------------
    // The property: the state carried by each MDD node, tested without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_three_seeds_of_a_property() {
        let constraint = over(3, &no_one_then_zero(), 2, &[0, 3]);
        assert_eq!(
            states_of(&*constraint.identity_property()),
            Vec::<usize>::new()
        );
        assert_eq!(states_of(&*constraint.empty_property()), vec![2]);
        assert_eq!(
            states_of(&*constraint.empty_property_backward()),
            vec![0, 3]
        );
    }

    #[test]
    fn update_follows_the_transitions_of_every_state_of_the_parent() {
        // From {0, 1}, the value 1 leads to {1} (from 0) and to None (from 1); from {0, 1}
        // the value 0 leads to {0}.
        let constraint = over(3, &no_two_ones(), 0, &[0, 1]);
        let parent = with_states(&constraint, &[0, 1]);
        let mut one = constraint.identity_property();
        one.update(&*parent, 1, true);
        assert_eq!(states_of(&*one), vec![1]);
        let mut zero = constraint.identity_property();
        zero.update(&*parent, 0, true);
        assert_eq!(states_of(&*zero), vec![0]);
        // From {1} alone, the value 1 leads nowhere.
        let mut dead = constraint.identity_property();
        dead.update(&*with_states(&constraint, &[1]), 1, true);
        assert_eq!(states_of(&*dead), Vec::<usize>::new());
    }

    #[test]
    fn update_unites_the_sets_of_the_parents() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0]);
        // The value 2 leads to state 3 from every state.
        let mut node = constraint.identity_property();
        node.update(&*with_states(&constraint, &[0]), 1, true);
        node.update(&*with_states(&constraint, &[2]), 0, true);
        // 0 --1--> 2, and 2 --0--> None.
        assert_eq!(states_of(&*node), vec![2]);
        node.update(&*with_states(&constraint, &[1]), 2, true);
        assert_eq!(states_of(&*node), vec![2, 3]);
    }

    #[test]
    fn update_out_of_scope_copies_the_parent() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0]);
        let parent = with_states(&constraint, &[1, 3]);
        let mut node = constraint.identity_property();
        node.update(&*parent, 1, false);
        assert_eq!(states_of(&*node), vec![1, 3]);
        node.update(&*with_states(&constraint, &[0]), 2, false);
        assert_eq!(states_of(&*node), vec![0, 1, 3]);
    }

    #[test]
    fn update_backward_collects_the_states_that_lead_into_the_child() {
        let constraint = over(3, &no_two_ones(), 0, &[0, 1]);
        // Which states lead into {1} on the value 1? Only state 0. On the value 0: none.
        let child = with_states(&constraint, &[1]);
        let mut one = constraint.identity_property();
        one.update_backward(&*child, 1, true);
        assert_eq!(states_of(&*one), vec![0]);
        let mut zero = constraint.identity_property();
        zero.update_backward(&*child, 0, true);
        assert_eq!(states_of(&*zero), Vec::<usize>::new());
        // Into {0} on the value 0, both states lead.
        let mut zero_to_zero = constraint.identity_property();
        zero_to_zero.update_backward(&*with_states(&constraint, &[0]), 0, true);
        assert_eq!(states_of(&*zero_to_zero), vec![0, 1]);
    }

    #[test]
    fn update_backward_out_of_scope_copies_the_child() {
        let constraint = over(3, &no_two_ones(), 0, &[0, 1]);
        let mut node = constraint.identity_property();
        node.update_backward(&*with_states(&constraint, &[1]), 1, false);
        assert_eq!(states_of(&*node), vec![1]);
    }

    #[test]
    fn a_value_outside_the_alphabet_reaches_nothing_in_either_direction() {
        let constraint = over(3, &no_two_ones(), 0, &[0, 1]);
        let everything = with_states(&constraint, &[0, 1]);
        let mut forward = constraint.identity_property();
        forward.update(&*everything, 7, true);
        assert_eq!(states_of(&*forward), Vec::<usize>::new());
        let mut backward = constraint.identity_property();
        backward.update_backward(&*everything, 7, true);
        assert_eq!(states_of(&*backward), Vec::<usize>::new());
    }

    #[test]
    fn merge_unites_the_sets() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0]);
        let mut a = with_states(&constraint, &[0, 2]);
        a.merge(&*with_states(&constraint, &[2, 3]));
        assert_eq!(states_of(&*a), vec![0, 2, 3]);
        assert_eq!(a.order_key(), vec![3.0]);
    }

    #[test]
    fn merge_is_idempotent_commutative_and_has_the_identity_as_neutral() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0]);
        let a = with_states(&constraint, &[0, 2]);
        let b = with_states(&constraint, &[1, 2]);

        let mut twice = a.clone();
        twice.merge(&*a);
        assert!(same(&*twice, &*a));

        let mut ab = a.clone();
        ab.merge(&*b);
        let mut ba = b.clone();
        ba.merge(&*a);
        assert!(same(&*ab, &*ba));

        let mut with_identity = a.clone();
        with_identity.merge(&*constraint.identity_property());
        assert!(same(&*with_identity, &*a));
    }

    #[test]
    fn eq_and_hash_compare_the_sets() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0]);
        let a = with_states(&constraint, &[0, 2]);
        let b = with_states(&constraint, &[2, 0]);
        assert!(same(&*a, &*b));
        assert_eq!(hash_of(&*a), hash_of(&*b));
        for other in [
            with_states(&constraint, &[0]),
            with_states(&constraint, &[2]),
            with_states(&constraint, &[0, 3]),
            constraint.identity_property(),
        ] {
            assert!(!same(&*a, &*other));
            assert_ne!(hash_of(&*a), hash_of(&*other));
        }
    }

    #[test]
    #[should_panic(expected = "Calling update on property")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        over(2, &no_two_ones(), 0, &[0])
            .identity_property()
            .update(&*foreign(), 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling update_backward on property")]
    fn update_backward_with_a_property_of_another_constraint_is_rejected() {
        over(2, &no_two_ones(), 0, &[0])
            .identity_property()
            .update_backward(&*foreign(), 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        over(2, &no_two_ones(), 0, &[0])
            .identity_property()
            .merge(&*foreign());
    }

    #[test]
    #[should_panic(expected = "Calling eq on property")]
    fn eq_with_a_property_of_another_constraint_is_rejected() {
        same(
            &*over(2, &no_two_ones(), 0, &[0]).identity_property(),
            &*foreign(),
        );
    }

    #[test]
    fn automata_with_more_than_64_states_use_every_state() {
        // A counter: state k is "k ones seen" up to 69, any value keeps the count of ones.
        let q = 70;
        let table: Table = (0..q)
            .map(|k| vec![Some(k), Some((k + 1).min(q - 1))])
            .collect();
        let domains = vec![dom(2); 2];
        let (_, vars, mut constraint) = build(&domains, &[0, 1], &table, 0, &[q - 1]);
        constraint.update_variable_ordering(&vars);
        let mut node = constraint.identity_property();
        node.update(&*with_states(&constraint, &[0, 65, 68]), 1, true);
        assert_eq!(states_of(&*node), vec![1, 66, 69]);
        let mut back = constraint.identity_property();
        back.update_backward(&*with_states(&constraint, &[1, 66, 69]), 1, true);
        // 69 is reached from 68 and from 69 itself (capped).
        assert_eq!(states_of(&*back), vec![0, 65, 68, 69]);
        assert_eq!(states_of(&*constraint.empty_property_backward()), vec![69]);
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    #[test]
    fn an_edge_is_valid_when_some_state_of_the_parent_leads_into_the_child() {
        let constraint = over(3, &no_one_then_zero(), 0, &[0, 1, 2, 3]);
        // (parent states, child states, value, invalid?)
        let cases: Vec<(Vec<usize>, Vec<usize>, isize, bool)> = vec![
            (vec![0], vec![1], 0, false),    // 0 --0--> 1
            (vec![0], vec![2], 0, true),     // 0 --0--> 1, not 2
            (vec![0], vec![1, 2], 1, false), // 0 --1--> 2
            (vec![2], vec![1], 0, true),     // no transition from 2 on 0
            (vec![2, 0], vec![1], 0, false), // one of the two parents' states is enough
            (vec![0, 1], vec![3], 2, false), // both lead to 3
            (vec![0], vec![], 0, true),      // a child without states
            (vec![], vec![1], 0, true),      // a parent without states
            (vec![], vec![], 0, true),
        ];
        for (parent, child, value, invalid) in cases {
            assert_eq!(
                constraint.is_assignment_invalid(
                    &*with_states(&constraint, &parent),
                    &*with_states(&constraint, &child),
                    1,
                    value
                ),
                invalid,
                "{parent:?} --{value}--> {child:?}"
            );
        }
    }

    #[test]
    fn a_value_outside_the_alphabet_is_always_invalid() {
        let constraint = over(3, &no_two_ones(), 0, &[0, 1]);
        let all = with_states(&constraint, &[0, 1]);
        assert!(constraint.is_assignment_invalid(&*all, &*all, 1, 9));
    }

    #[test]
    #[should_panic(expected = "instead of RegularProperty")]
    fn a_property_of_another_constraint_is_rejected_when_filtering() {
        let constraint = over(2, &no_two_ones(), 0, &[0]);
        let own = constraint.empty_property();
        constraint.is_assignment_invalid(&*foreign(), &*own, 0, 0);
    }

    // ----------------------------------------------------------------------------------------
    // Compiled MDDs against a brute-force oracle
    // ----------------------------------------------------------------------------------------

    #[test]
    fn no_one_then_zero_is_exact() {
        for d in 1..=5 {
            let domains = vec![dom(3); d];
            let scope: Vec<usize> = (0..d).collect();
            let mut problem = Problem::default();
            let vars = problem.add_variables(d, dom(3), None);
            regular(&mut problem, vars, no_one_then_zero(), 0, vec![0, 1, 2, 3]);
            let mdd = settled(problem, (0..d).collect());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &scope, &no_one_then_zero(), 0, &[0, 1, 2, 3]),
                "{d} variables"
            );
        }
    }

    #[test]
    fn no_two_ones_counts_the_fibonacci_numbers() {
        // The number of binary words of length n without two consecutive ones is F(n + 2).
        let fib = [2usize, 3, 5, 8, 13, 21, 34, 55];
        for (n, count) in (1..9).zip(fib) {
            let domains = vec![dom(2); n];
            let mut problem = Problem::default();
            let vars: Vec<VariableIndex> = domains
                .iter()
                .map(|d| problem.add_variable(d.clone(), None))
                .collect();
            regular(&mut problem, vars, no_two_ones(), 0, vec![0, 1]);
            let mdd = settled(problem, (0..n).collect());
            assert_eq!(accepted(&mdd).len(), count, "{n} variables");
        }
    }

    #[test]
    fn a_layer_holds_at_most_one_node_per_state() {
        let n = 8;
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(2), None);
        regular(&mut problem, vars, no_two_ones(), 0, vec![0, 1]);
        let mdd = settled(problem, (0..n).collect());
        for layer in 0..=n {
            assert!(
                mdd.number_nodes_in_layer(layer) <= 2,
                "layer {layer} holds {} nodes",
                mdd.number_nodes_in_layer(layer)
            );
        }
    }

    #[test]
    fn random_instances_are_exact_once_settled() {
        let mut rng = Lcg(99);
        for _ in 0..300 {
            let i = random_instance(&mut rng);
            let order = order_for(&i, &mut rng);
            let (problem, _) = problem_of(&i);
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&i.domains, &i.scope, &i.table, i.initial, &i.accepting),
                "domains {:?} scope {:?} table {:?} initial {} accepting {:?} order {:?}",
                i.domains,
                i.scope,
                i.table,
                i.initial,
                i.accepting,
                order
            );
        }
    }

    #[test]
    fn relaxed_mdds_never_lose_a_solution() {
        let mut rng = Lcg(123);
        for _ in 0..300 {
            let i = random_instance(&mut rng);
            let order = order_for(&i, &mut rng);
            let want = expected(&i.domains, &i.scope, &i.table, i.initial, &i.accepting);
            for width in [1usize, 2, 3, usize::MAX] {
                let (problem, _) = problem_of(&i);
                let got = accepted(&compile(problem, order.clone(), width));
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: domains {:?} scope {:?} table {:?} width {width}",
                        i.domains,
                        i.scope,
                        i.table
                    );
                }
            }
        }
    }

    #[test]
    fn a_language_without_a_word_of_the_length_is_unsat() {
        // No accepting state.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        regular(&mut problem, vars, no_two_ones(), 0, vec![]);
        assert!(accepted(&settled(problem, vec![0, 1])).is_empty());

        // Both variables are forced to 1, which the automaton forbids twice in a row.
        let mut problem = Problem::default();
        let a = problem.add_variable(vec![0, 1], None);
        let b = problem.add_variable(vec![0, 1], None);
        regular(&mut problem, vec![a, b], no_two_ones(), 0, vec![0, 1]);
        problem[a].set_domain(vec![1]);
        problem[b].set_domain(vec![1]);
        assert!(accepted(&settled(problem, vec![0, 1])).is_empty());
    }

    #[test]
    fn the_scope_is_read_in_declared_order_whatever_the_indices() {
        // The word 1, 0, 0 read as (v2, v0, v1).
        let table = vec![
            vec![None, Some(1)],
            vec![Some(2), None],
            vec![Some(3), None],
            vec![None, None],
        ];
        let domains = vec![dom(2); 3];
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        regular(
            &mut problem,
            vec![vars[2], vars[0], vars[1]],
            table,
            0,
            vec![3],
        );
        // The compilation order has to respect the scope: v2, v0, v1.
        let mdd = settled(problem, vec![2, 0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![0, 0, 1]]);
    }

    #[test]
    fn a_heuristic_ordering_is_repaired_to_respect_the_scope() {
        let table = vec![
            vec![None, Some(1)],
            vec![Some(2), None],
            vec![Some(3), None],
            vec![None, None],
        ];
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        regular(
            &mut problem,
            vec![vars[2], vars[0], vars[1]],
            table,
            0,
            vec![3],
        );
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
        assert_eq!(accepted(&mdd), vec![vec![0, 0, 1]]);
    }

    #[test]
    fn variables_outside_the_scope_can_sit_between_the_scope_layers() {
        // v1 and v3 are free; the scope is (v0, v2) with "no two ones".
        let domains = vec![dom(2); 4];
        for order in [vec![0, 1, 2, 3], vec![1, 0, 3, 2], vec![1, 3, 0, 2]] {
            let mut problem = Problem::default();
            let vars: Vec<VariableIndex> = domains
                .iter()
                .map(|d| problem.add_variable(d.clone(), None))
                .collect();
            regular(
                &mut problem,
                vec![vars[0], vars[2]],
                no_two_ones(),
                0,
                vec![0, 1],
            );
            // A trivial automaton on the others keeps them in the MDD, read as v1 then v3.
            add_free(&mut problem, vec![vars[1], vars[3]]);
            let mdd = settled(problem, order.clone());
            assert_eq!(
                accepted(&mdd),
                expected(&domains, &[0, 2], &no_two_ones(), 0, &[0, 1]),
                "order {order:?}"
            );
        }
    }

    #[test]
    fn two_automata_over_overlapping_scopes_are_exact() {
        let domains = vec![dom(2); 4];
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        regular(
            &mut problem,
            vec![vars[0], vars[1], vars[2]],
            no_two_ones(),
            0,
            vec![0, 1],
        );
        // The second one: no two consecutive zeros on (v1, v2, v3).
        let no_two_zeros = vec![vec![Some(1), Some(0)], vec![None, Some(0)]];
        regular(
            &mut problem,
            vec![vars[1], vars[2], vars[3]],
            no_two_zeros.clone(),
            0,
            vec![0, 1],
        );
        let mdd = settled(problem, vec![0, 1, 2, 3]);
        let want: Vec<Vec<isize>> = product(&domains)
            .into_iter()
            .filter(|t| {
                let a = runs(&no_two_ones(), 0, &[0, 1], &[0, 1], &[t[0], t[1], t[2]]);
                let b = runs(&no_two_zeros, 0, &[0, 1], &[0, 1], &[t[1], t[2], t[3]]);
                a && b
            })
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn combined_with_not_equals_is_exact() {
        let domains = vec![dom(2); 3];
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        regular(&mut problem, vars.clone(), no_two_ones(), 0, vec![0, 1]);
        not_equals(&mut problem, vars[0], vars[2]);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want: Vec<Vec<isize>> = expected(&domains, &[0, 1, 2], &no_two_ones(), 0, &[0, 1])
            .into_iter()
            .filter(|t| t[0] != t[2])
            .collect();
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
    }

    #[test]
    fn domains_with_negative_and_unsorted_values_are_exact() {
        // The alphabet is -3, 2, 5 whatever the order of the domains. The automaton accepts the
        // words that never go back to a smaller value than the previous one (non decreasing).
        let table = vec![
            vec![Some(0), Some(1), Some(2)],
            vec![None, Some(1), Some(2)],
            vec![None, None, Some(2)],
        ];
        let domains = vec![vec![5, -3], vec![2, 5, -3], vec![-3, 5]];
        let scope = [0usize, 1, 2];
        let mut problem = Problem::default();
        let vars: Vec<VariableIndex> = domains
            .iter()
            .map(|d| problem.add_variable(d.clone(), None))
            .collect();
        regular(&mut problem, vars, table.clone(), 0, vec![0, 1, 2]);
        let mdd = settled(problem, vec![0, 1, 2]);
        let want = expected(&domains, &scope, &table, 0, &[0, 1, 2]);
        assert_eq!(accepted(&mdd), want);
        assert!(!want.is_empty());
        assert!(want.contains(&vec![-3, 2, 5]) && !want.contains(&vec![5, -3, 5]));
    }

    #[test]
    fn a_domain_that_shrinks_after_construction_is_fine() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, dom(2), None);
        regular(&mut problem, vars.clone(), no_two_ones(), 0, vec![0, 1]);
        problem[vars[1]].set_domain(vec![1]);
        let mdd = settled(problem, vec![0, 1, 2]);
        assert_eq!(accepted(&mdd), vec![vec![0, 1, 0]]);
    }

    #[test]
    fn a_value_added_to_a_domain_after_construction_is_rejected() {
        // The value 2 is not in the alphabet seen at construction: no word can use it.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, dom(2), None);
        regular(&mut problem, vars.clone(), no_two_ones(), 0, vec![0, 1]);
        problem[vars[0]].set_domain(vec![0, 1, 2]);
        let mdd = settled(problem, vec![0, 1]);
        assert_eq!(accepted(&mdd), vec![vec![0, 0], vec![0, 1], vec![1, 0]]);
    }

    #[test]
    fn a_long_chain_of_states_compiles_to_the_single_word() {
        // 65 variables; the automaton counts the ones up to 65 and accepts only a count of 65.
        let n = 65;
        let table: Table = (0..=n)
            .map(|k| vec![Some(k), Some((k + 1).min(n))])
            .collect();
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, dom(2), None);
        regular(&mut problem, vars, table, 0, vec![n]);
        let mdd = settled(problem, (0..n).collect());
        assert_eq!(accepted(&mdd), vec![vec![1; n]]);
    }
}
