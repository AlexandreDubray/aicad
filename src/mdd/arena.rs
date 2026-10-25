//! A cache of compiled constraint structures, shared by every constraint of the same shape.
//!
//! # Purpose
//!
//! The learning and search code needs the MDD of every constraint of every instance it sees, and
//! most of those constraints are copies of each other on different variables: the same
//! `NotEquals` on every edge of a graph, the same `Gcc` for every nurse. Their diagrams are the
//! same up to the names of the variables. `MddArena` compiles one diagram per *shape* and hands it
//! out as an `Arc<Mdd>`; [`CompiledConstraint`] pairs it with the constraint's own
//! branching order, which says which real variable each layer stands for.
//!
//! # How it works
//!
//! A shape is a [`ConstraintCompileKey`]: the constraint's `ConstraintShapeKey`, the nominal
//! domain of its variables and the width limit. To compile a shape, the arena builds a small
//! synthetic problem with `arity` fresh variables that all take the key's domain, instantiates the
//! constraint on them, compiles and refines its MDD, and keeps the `Mdd` once the compiler is dropped. The first
//! caller of a shape pays for the compilation (outside the lock); concurrent callers of the
//! same shape end up with the same `Arc`.
//!
//! The arena never decides what the domain is: the caller passes the nominal domain of its
//! network (`nominal_domain(n)` is `0..n`, the one of the benchmarks), and the shape is compiled
//! over that full domain even when a real variable's domain has been narrowed. Keeping the domain
//! inside the key, and building the synthetic variables only from the key, is what lets the arena
//! serve other domains without change.
//!
//! # Design notes
//!
//! - **A structure serves every ordering of the scope.** The template is always compiled with
//!   `MinDomMaxLinked`, `StateSimilarity` and `Greedy` on its synthetic variables, and a real
//!   constraint only computes its own order. This is correct because the constraints are
//!   symmetric in their scope, so any assignment of variables to layers gives the same set of
//!   accepted tuples; the exception, `Regular`, has its order forced by `precedence_edges`, which
//!   both orders respect.
//! - **The kept diagram refers to the synthetic problem.** An `Mdd` holds the problem it was
//!   compiled from, here the template's few variables, and its own order, which names those
//!   variables. Neither is read through the arena: a [`CompiledConstraint`] brings the order of
//!   the real constraint. The cost is one small problem per shape.
//! - **Nothing is ever evicted.** The arena grows with the number of distinct shapes it sees,
//!   which is small for the benchmarks (one `Sum` shape per target, one `Gcc` per set of bounds).
//! - **The key holds the domain, not just its size.** Today every variable is assumed to take the
//!   same domain, but this will be lifted some day. Then a `Sum([x, y], 4)` must not share its
//!   structure with a `Sum([z, t], 4)` if the variables have different domains, so the domain
//!   already belongs to the key.

use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::constraints::{
    AllDifferent, Among, AtLeast, Constraint, ConstraintShapeKey, Gcc, NotEquals, Regular, Sum,
};
use crate::mdd::compiler::enforce_precedence_order;
use crate::mdd::heuristics::{
    CompilationHeuristic, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
};
use crate::mdd::{Compiler, Mdd};
use crate::modelling::{ConstraintIndex, Problem, VariableIndex};

/// Everything the compiled structure of a constraint depends on: its shape, the domain of its
/// variables and the width limit. Two constraints with the same key share one structure,
/// whichever variables and problem they belong to.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ConstraintCompileKey {
    /// The constraint's own shape
    shape: ConstraintShapeKey,
    /// The domain that every variable of the template takes (the network's nominal domain).
    domain: Vec<isize>,
    /// Width limit the template is refined to (`usize::MAX` for an exact structure).
    max_width: usize,
}

/// The nominal domain `0..domain_size` of a network whose variables take the values
/// `0..domain_size`, in the form [`compile_constraint`] takes.
pub(crate) fn nominal_domain(domain_size: usize) -> Vec<isize> {
    (0..domain_size as isize).collect()
}

/// The key of `constraint`, a constraint of the real `problem`, compiled over `domain`. Only the
/// constraint's shape is read from the problem, never the (possibly narrowed) domains of its
/// variables.
fn constraint_shape_key(
    problem: &Problem,
    constraint: ConstraintIndex,
    domain: &[isize],
    max_width: usize,
) -> ConstraintCompileKey {
    ConstraintCompileKey {
        shape: problem[constraint].structural_key(problem),
        domain: domain.to_vec(),
        max_width,
    }
}

/// Rebuilds the constraint that `shape` describes over `synthetic_variables`, fresh variables of
/// `synthetic_problem`, one per `shape.arity()`.
fn instantiate_constraint(
    shape: &ConstraintShapeKey,
    synthetic_variables: Vec<VariableIndex>,
    synthetic_problem: &Problem,
) -> Box<dyn Constraint + Send + Sync> {
    // `arity` is destructured as `_` because it is already included in
    // `synthetic_variables.len()`. It is in the key to differentiate constraints of different
    // arity.
    match shape {
        ConstraintShapeKey::AllDifferent { arity: _ } => {
            Box::new(AllDifferent::new(synthetic_variables, synthetic_problem))
        }
        ConstraintShapeKey::NotEquals => {
            if synthetic_variables.len() != 2 {
                panic!(
                    "NotEquals shape key must instantiate over exactly 2 variables, got {}",
                    synthetic_variables.len()
                );
            }
            Box::new(NotEquals::new(
                synthetic_variables[0],
                synthetic_variables[1],
                synthetic_problem,
            ))
        }
        ConstraintShapeKey::Among {
            arity: _,
            values,
            lb,
            ub,
        } => Box::new(Among::new(
            synthetic_variables,
            FxHashSet::from_iter(values.iter().copied()),
            *lb,
            *ub,
        )),
        ConstraintShapeKey::AtLeast {
            arity: _,
            values,
            lb,
        } => Box::new(AtLeast::new(
            synthetic_variables,
            FxHashSet::from_iter(values.iter().copied()),
            *lb,
        )),
        ConstraintShapeKey::Gcc { arity: _, bounds } => {
            Box::new(Gcc::new(synthetic_variables, bounds.clone()))
        }
        ConstraintShapeKey::Regular {
            arity: _,
            transitions,
            initial_state,
            accepting_states,
        } => Box::new(Regular::new(
            synthetic_variables,
            transitions.clone(),
            *initial_state,
            FxHashSet::from_iter(accepting_states.iter().copied()),
            synthetic_problem,
        )),
        ConstraintShapeKey::Sum { arity: _, target } => {
            Box::new(Sum::new(synthetic_variables, *target, synthetic_problem))
        }
    }
}

/// Compiles the structure of `key` on a synthetic problem with a single constraint (see the module
/// documentation) and refines it to the key's width.
fn compile_template(key: &ConstraintCompileKey) -> Mdd {
    // We create a small synthetic problem with one constraint over its variables because an MDD
    // can only be compiled from a problem.
    let mut synthetic_problem = Problem::default();
    let synthetic_vars: Vec<VariableIndex> = (0..key.shape.arity())
        .map(|_| synthetic_problem.add_variable(key.domain.clone(), None))
        .collect();

    let constraint = instantiate_constraint(&key.shape, synthetic_vars, &synthetic_problem);
    synthetic_problem.add_constraint_boxed(constraint);

    let synthetic_problem = Arc::new(synthetic_problem);
    let constraints: Vec<ConstraintIndex> = synthetic_problem.iter_constraints().collect();
    let mut compiler = Compiler::new(
        Arc::clone(&synthetic_problem),
        CompilationHeuristic {
            ordering: OrderingHeuristic::MinDomMaxLinked,
            merge: MergeHeuristic::StateSimilarity,
            selection: SelectHeuristic::Greedy,
        },
        &constraints,
    );
    compiler.refine(key.max_width);
    compiler.into_mdd()
}

/// Thread-safe cache from a constraint's shape to its shared, compiled structure.
///
/// One arena is shared by a whole dataset build or search run, so that constraints of the same
/// shape, in one instance or across instances, are compiled once. It can be used from several
/// threads.
#[derive(Default)]
pub struct MddArena {
    /// The compiled structure of each shape seen so far.
    cache: RwLock<HashMap<ConstraintCompileKey, Arc<Mdd>>>,
}

impl MddArena {
    /// Creates an empty arena.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct shapes compiled so far.
    pub fn len(&self) -> usize {
        self.cache.read().expect("MddArena lock poisoned").len()
    }

    /// `true` if no shape has been compiled yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shared structure of `key`, compiled on first use.
    fn structure_for(&self, key: &ConstraintCompileKey) -> Arc<Mdd> {
        if let Some(structure) = self.cache.read().expect("MddArena lock poisoned").get(key) {
            return Arc::clone(structure);
        }
        // Compile outside the lock: compilation is the expensive part, and re-checking after
        // acquiring the write lock avoids two racing callers both paying for it -- whichever
        // loses the race just throws its (identical) work away.
        let structure = Arc::new(compile_template(key));
        let mut cache = self.cache.write().expect("MddArena lock poisoned");
        Arc::clone(cache.entry(key.clone()).or_insert(structure))
    }
}

/// A constraint ready for algorithmic processing (e.g., `wmc` or `sampling`): the structure shared
/// through the arena, and the constraint's own branching order.
pub(crate) struct CompiledConstraint {
    /// The structure, shared with every constraint of the same shape.
    pub structure: Arc<Mdd>,
    /// The variable of the constraint's scope that each layer stands for, first layer first.
    pub order: Vec<VariableIndex>,
}

impl CompiledConstraint {
    /// The variable branched on at `layer`.
    pub(crate) fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        self.order[layer]
    }
}

impl CompiledConstraint {
    /// A constraint that branches on the variables of its own diagram, `mdd` having been compiled
    /// from the problem it is read with. For tests, which compile a diagram directly.
    #[cfg(test)]
    pub(crate) fn from_mdd(mdd: Mdd) -> Self {
        let order = (0..mdd.number_decisions())
            .map(|layer| mdd.decision_at_layer(layer))
            .collect();
        Self {
            structure: Arc::new(mdd),
            order,
        }
    }

    /// Number of variables branched on, which is the size of the scope. Unlike the number of
    /// layers of the diagram it is not 0 when the diagram is unsatisfiable.
    pub(crate) fn number_decisions(&self) -> usize {
        self.order.len()
    }
}

impl std::ops::Deref for CompiledConstraint {
    type Target = Mdd;

    fn deref(&self) -> &Mdd {
        &self.structure
    }
}

/// The branching order of a real `constraint`: the two steps `Mdd::new` runs before compiling,
/// the heuristic's order and the repair of the precedences, without compiling anything.
fn compute_constraint_order(
    problem: &Problem,
    constraint: ConstraintIndex,
    ordering: &OrderingHeuristic,
) -> Vec<VariableIndex> {
    let scope: Vec<VariableIndex> = problem[constraint].iter_scope().collect();
    let owned: Vec<Box<dyn Constraint>> = vec![problem[constraint].clone()];
    let order = ordering.get_order(problem, &scope);
    enforce_precedence_order(order, &owned)
}

/// Compiles `constraint`, a constraint of the real `problem`, into a [`CompiledConstraint`]: its
/// structure comes from `arena` (shared with the constraints of the same shape, `domain` and
/// `max_width`) and its branching order is computed with `ordering`.
///
/// `domain` is the nominal domain of the network's variables (`nominal_domain(n)` for the values
/// `0..n`). `max_width` is the width the structure is refined to, `usize::MAX` for an
/// exact one.
pub(crate) fn compile_constraint(
    arena: &MddArena,
    problem: &Problem,
    constraint: ConstraintIndex,
    ordering: &OrderingHeuristic,
    domain: &[isize],
    max_width: usize,
) -> CompiledConstraint {
    let key = constraint_shape_key(problem, constraint, domain, max_width);
    let structure = arena.structure_for(&key);
    let order = compute_constraint_order(problem, constraint, ordering);
    // An unsatisfiable structure has no layer at all, whatever the scope.
    debug_assert!(
        structure.is_unsat() || order.len() + 1 == structure.number_layers(),
        "a real constraint's branching order must have exactly one fewer entry than its \
         arena-shared structure's layer count (the structure has one extra, sink, layer) -- a \
         mismatch here means the constraint's real scope doesn't actually match the shape key it \
         was compiled/looked-up under"
    );
    CompiledConstraint { structure, order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{all_different, among, at_least, gcc, not_equals, regular, sum};
    use crate::mdd::NodeIndex;
    use crate::modelling::ValueIndex;

    /// Whether the compiled structure has a path for `tuple`, a value for every variable of the
    /// problem (the values are the indices, since the domains are `0..d`).
    fn accepts(compiled: &CompiledConstraint, tuple: &[isize]) -> bool {
        if compiled.is_unsat() {
            return false;
        }
        let mut node = compiled.root();
        for layer in 0..compiled.order.len() {
            let value = ValueIndex(tuple[compiled.order[layer].0] as usize);
            let next = compiled[node]
                .iter_children()
                .map(|edge| &compiled[edge])
                .find(|edge| edge.is_active() && edge.assignment() == value)
                .map(|edge| edge.to());
            match next {
                Some(to) => node = to,
                None => return false,
            }
        }
        node == compiled.sink() && !compiled.is_unsat()
    }

    /// Every assignment of `variables` variables over `0..domain`.
    fn all_tuples(variables: usize, domain: isize) -> Vec<Vec<isize>> {
        let mut tuples = vec![vec![]];
        for _ in 0..variables {
            tuples = tuples
                .into_iter()
                .flat_map(|t| {
                    (0..domain).map(move |v| {
                        let mut t = t.clone();
                        t.push(v);
                        t
                    })
                })
                .collect();
        }
        tuples
    }

    /// One instance of each kind of constraint, over variables `0..4` of domain `0..3`, in a
    /// problem where the layers of the real constraints are not in the order of the template's.
    fn one_of_each_constraint() -> (Problem, Vec<ConstraintIndex>) {
        let mut problem = Problem::default();
        let v = problem.add_variables(4, (0..3isize).collect(), None);
        // The last variable is in the most constraints, so the heuristic branches on it first,
        // while the synthetic template branches on its first variable first.
        for other in 0..3 {
            not_equals(&mut problem, v[3], v[other]);
        }
        let constraints = vec![
            all_different(&mut problem, vec![v[0], v[1], v[2]]),
            not_equals(&mut problem, v[1], v[3]),
            among(&mut problem, v.clone(), vec![0, 1], 1, 2),
            at_least(&mut problem, v.clone(), vec![2], 2),
            gcc(&mut problem, v.clone(), vec![(0, 1, 2), (2, 0, 1)]),
            sum(&mut problem, v.clone(), 5),
            // Two consecutive equal values are forbidden.
            regular(
                &mut problem,
                v.clone(),
                vec![
                    vec![Some(1), Some(2), Some(3)],
                    vec![None, Some(2), Some(3)],
                    vec![Some(1), None, Some(3)],
                    vec![Some(1), Some(2), None],
                ],
                0,
                vec![1, 2, 3],
            ),
        ];
        (problem, constraints)
    }

    #[test]
    fn an_exact_structure_accepts_exactly_the_satisfying_tuples() {
        let (problem, constraints) = one_of_each_constraint();
        let arena = MddArena::new();
        for constraint in constraints {
            let compiled = compile_constraint(
                &arena,
                &problem,
                constraint,
                &OrderingHeuristic::MinDomMaxLinked,
                &nominal_domain(3),
                usize::MAX,
            );
            for tuple in all_tuples(4, 3) {
                assert_eq!(
                    accepts(&compiled, &tuple),
                    problem[constraint].is_satisfied(&tuple),
                    "{} on {tuple:?}",
                    problem[constraint].name()
                );
            }
        }
    }

    #[test]
    fn a_width_limited_structure_accepts_every_satisfying_tuple() {
        let (problem, constraints) = one_of_each_constraint();
        let arena = MddArena::new();
        for constraint in constraints {
            for width in [1, 2, 3] {
                let compiled = compile_constraint(
                    &arena,
                    &problem,
                    constraint,
                    &OrderingHeuristic::MinDomMaxLinked,
                    &nominal_domain(3),
                    width,
                );
                for tuple in all_tuples(4, 3) {
                    if problem[constraint].is_satisfied(&tuple) {
                        assert!(
                            accepts(&compiled, &tuple),
                            "{} at width {width} lost {tuple:?}",
                            problem[constraint].name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_order_lists_the_scope_of_the_constraint_once() {
        let (problem, constraints) = one_of_each_constraint();
        let arena = MddArena::new();
        let mut reordered = 0;
        for constraint in constraints {
            let compiled = compile_constraint(
                &arena,
                &problem,
                constraint,
                &OrderingHeuristic::MinDomMaxLinked,
                &nominal_domain(3),
                usize::MAX,
            );
            let declared: Vec<_> = problem[constraint].iter_scope().collect();
            reordered += usize::from(compiled.order != declared);
            let mut order = compiled.order.clone();
            order.sort();
            let mut scope: Vec<_> = problem[constraint].iter_scope().collect();
            scope.sort();
            assert_eq!(order, scope, "{}", problem[constraint].name());
            assert_eq!(compiled.number_layers(), scope.len() + 1);
            assert_eq!(compiled.root(), NodeIndex(0, 0));
        }
        // The other tests are only meaningful if some layers are not in the declared order.
        assert!(reordered > 0);
    }

    #[test]
    fn the_width_and_the_domain_are_part_of_the_key() {
        let mut problem = Problem::default();
        let v = problem.add_variables(3, (0..3isize).collect(), None);
        let c = all_different(&mut problem, v);
        let arena = MddArena::new();
        let compile = |domain_size: usize, max_width| {
            compile_constraint(
                &arena,
                &problem,
                c,
                &OrderingHeuristic::MinDomMaxLinked,
                &nominal_domain(domain_size),
                max_width,
            )
        };
        let exact = compile(3, usize::MAX);
        let again = compile(3, usize::MAX);
        let narrow = compile(3, 2);
        let wider_domain = compile(4, usize::MAX);
        assert!(Arc::ptr_eq(&exact.structure, &again.structure));
        assert!(!Arc::ptr_eq(&exact.structure, &narrow.structure));
        assert!(!Arc::ptr_eq(&exact.structure, &wider_domain.structure));
        assert_eq!(arena.len(), 3);
        assert!(!arena.is_empty());
        assert!(MddArena::new().is_empty());
    }

    #[test]
    fn threads_asking_for_the_same_shape_share_one_structure() {
        let mut problem = Problem::default();
        let v = problem.add_variables(5, (0..5isize).collect(), None);
        let c = all_different(&mut problem, v);
        let arena = MddArena::new();
        let structures: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        compile_constraint(
                            &arena,
                            &problem,
                            c,
                            &OrderingHeuristic::MinDomMaxLinked,
                            &nominal_domain(5),
                            usize::MAX,
                        )
                        .structure
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(arena.len(), 1);
        assert!(
            structures
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0], &pair[1]))
        );
    }

    #[test]
    fn structurally_identical_constraints_from_different_scopes_share_one_arena_entry() {
        // Two "nurse"-style constraints: same bounds, but disjoint VariableIndex ranges (one
        // problem, 4 variables total -- first 2 form one constraint, last 2 form the other).
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, (0..3isize).collect(), None);
        let g1 = gcc(
            &mut problem,
            vec![vars[0], vars[1]],
            vec![(0, 0, 1), (1, 0, 1)],
        );
        let g2 = gcc(
            &mut problem,
            vec![vars[2], vars[3]],
            vec![(0, 0, 1), (1, 0, 1)],
        );
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let a = compile_constraint(
            &arena,
            &problem,
            g1,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g2,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        assert!(Arc::ptr_eq(&a.structure, &b.structure));
        assert_eq!(arena.len(), 1);
        assert_ne!(
            a.order, b.order,
            "each constraint must keep its own real scope"
        );
    }

    #[test]
    fn differently_shaped_constraints_do_not_share() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, (0..3isize).collect(), None);
        let g1 = gcc(&mut problem, vec![vars[0], vars[1]], vec![(0, 0, 1)]);
        let g2 = gcc(&mut problem, vec![vars[2], vars[3]], vec![(0, 1, 2)]);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let a = compile_constraint(
            &arena,
            &problem,
            g1,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g2,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        assert!(!Arc::ptr_eq(&a.structure, &b.structure));
        assert_eq!(arena.len(), 2);
    }

    /// The key of `AllDifferent` carries its arity, so `AllDifferent(x, y, z)` and
    /// `AllDifferent(x, y, z, t)` do not collide.
    #[test]
    fn all_different_constraints_of_different_arity_do_not_share() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, (0..4isize).collect(), None);
        let g3 = all_different(&mut problem, vec![vars[0], vars[1], vars[2]]);
        let g4 = all_different(&mut problem, vec![vars[0], vars[1], vars[2], vars[3]]);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let a = compile_constraint(
            &arena,
            &problem,
            g3,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(4),
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g4,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(4),
            usize::MAX,
        );
        assert!(!Arc::ptr_eq(&a.structure, &b.structure));
        assert_eq!(arena.len(), 2);
        assert_eq!(a.order.len(), 3);
        assert_eq!(b.order.len(), 4);
    }

    /// The graph-colouring symmetric case: two `NotEquals` edges in the *same* problem whose
    /// endpoints have different degree (so `MinDomMaxLinked`'s whole-problem-aware scoring would
    /// give a naively-compiled `Mdd::new` a different literal branching order for each) must
    /// still dedup to one arena entry once compiled via the synthetic-problem mechanism.
    #[test]
    fn symmetric_graph_coloring_edges_with_different_degree_endpoints_still_dedup() {
        let mut problem = Problem::default();
        let colours: Vec<isize> = (0..3).collect();
        let vars = problem.add_variables(6, colours, None);
        // vars[0] has high degree (linked to vars[2..6]); vars[1] has degree 1 (only the edge
        // below). Real Mdd::new over just {vars[0], vars[1]} would see this asymmetry via
        // `problem[candidate].number_constraints()`/scores, unlike a synthetic 2-variable
        // problem containing only this one constraint.
        let e1 = not_equals(&mut problem, vars[0], vars[1]);
        for i in 2..6 {
            not_equals(&mut problem, vars[0], vars[i]);
        }
        // A second, low-degree-on-both-ends edge with the same shape.
        let e2 = not_equals(&mut problem, vars[4], vars[5]);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let a = compile_constraint(
            &arena,
            &problem,
            e1,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            e2,
            &OrderingHeuristic::MinDomMaxLinked,
            &nominal_domain(3),
            usize::MAX,
        );
        assert!(
            Arc::ptr_eq(&a.structure, &b.structure),
            "symmetric NotEquals edges with different-degree endpoints must still dedup"
        );
        assert_eq!(arena.len(), 1);
    }
}
