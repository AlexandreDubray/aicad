//! `MddArena`: a cache from a single constraint's *shape* to one shared, compiled `MddStructure`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::constraints::{Constraint, ConstraintShapeKey};
use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
use crate::mdd::mdd::enforce_precedence_order;
use crate::mdd::{Mdd, MddStructure};
use crate::modelling::{ConstraintIndex, Problem, VariableIndex};

/// One constraint's compiled shape, together with the two extra parameters that also affect its
/// compiled structure. Two real constraints with the same `ConstraintCompileKey` compile to
/// byte-identical `MddStructure`s, regardless of which real `VariableIndex` or which real
/// `Arc<Problem>` they come from.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConstraintCompileKey {
    /// The constraint's own shape
    shape: ConstraintShapeKey,
    /// Domain size of all variable of the problem being solved (assumed to be identical)
    domain_size: usize,
    /// Maximum width for constraint compilation
    max_width: usize,
}

/// Computes `constraint`'s `ConstraintCompileKey` against `problem` (the *real* problem the
/// constraint actually belongs to -- only used to read its own shape, never its variables' real,
/// possibly-narrowed domains).
fn constraint_shape_key(
    problem: &Problem,
    constraint: ConstraintIndex,
    domain_size: usize,
    max_width: usize,
) -> ConstraintCompileKey {
    ConstraintCompileKey {
        shape: problem[constraint].structural_key(problem),
        domain_size,
        max_width,
    }
}

/// Rebuilds a concrete `Box<dyn Constraint>` for one shape key, over `synthetic_variables` (fresh
/// variables of `synthetic_problem`, one per `shape.arity()`).
fn instantiate_constraint(
    shape: &ConstraintShapeKey,
    synthetic_variables: Vec<VariableIndex>,
    synthetic_problem: &Problem,
) -> Box<dyn Constraint + Send + Sync> {
    use crate::constraints::{AllDifferent, Among, AtLeast, Gcc, NotEquals, Regular, Sum};
    use rustc_hash::FxHashSet;

    // `arity` is destructured (as `_`) but not read here -- it's already fully accounted for by
    // `synthetic_variables`' own length (`compile_template` builds exactly `shape.arity()` of
    // them); it exists on `ConstraintShapeKey` so the key itself is self-describing (see that
    // type's doc), not because this constructor needs it as a separate input.
    match shape {
        ConstraintShapeKey::AllDifferent { arity: _ } => {
            Box::new(AllDifferent::new(synthetic_variables, synthetic_problem))
        }
        ConstraintShapeKey::NotEquals => {
            assert_eq!(
                synthetic_variables.len(),
                2,
                "NotEquals shape key must instantiate over exactly 2 variables"
            );
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

/// Builds the small synthetic `Problem` `key` describes (see the module doc), compiles+refines an
/// exact `Mdd` over its own (only) constraint, and extracts the resulting `MddStructure`.
fn compile_template(key: &ConstraintCompileKey) -> MddStructure {
    let mut synthetic_problem = Problem::default();
    let domain: Vec<isize> = (0..key.domain_size as isize).collect();
    let synthetic_vars: Vec<VariableIndex> = (0..key.shape.arity())
        .map(|_| synthetic_problem.add_variable(domain.clone(), None))
        .collect();

    let constraint = instantiate_constraint(&key.shape, synthetic_vars, &synthetic_problem);
    synthetic_problem.add_constraint_boxed(constraint);

    let synthetic_problem = Arc::new(synthetic_problem);
    let constraints: Vec<ConstraintIndex> = synthetic_problem.iter_constraints().collect();
    let mut mdd = Mdd::new(
        Arc::clone(&synthetic_problem),
        OrderingHeuristic::MinDomMaxLinked,
        MergeHeuristic::LessRelaxed,
        SelectHeuristic::Greedy,
        &constraints,
    );
    mdd.refine(key.max_width);
    mdd.into_structure()
}

/// Thread-safe cache from a constraint's shape to its shared, compiled `MddStructure`. Shared
/// across an entire dataset build (`compile_constraint_mdds`) or an entire
/// `MddSamplingDecode`/search run, so structurally-identical constraints -- whether from the same
/// instance (e.g. two nurses under the same case file) or from different instances entirely --
/// compile once.
#[derive(Default)]
pub struct MddArena {
    cache: RwLock<HashMap<ConstraintCompileKey, Arc<MddStructure>>>,
}

impl MddArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct shapes compiled so far -- exposed for tests/diagnostics.
    pub fn len(&self) -> usize {
        self.cache.read().expect("MddArena lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn structure_for(&self, key: &ConstraintCompileKey) -> Arc<MddStructure> {
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

/// One real constraint's compiled structure (shared, via `MddArena`) plus its own real branching
/// order (never shared -- cheap to recompute, and specific to this constraint's actual
/// `VariableIndex`s).
pub struct CompiledConstraint {
    pub structure: Arc<MddStructure>,
    pub order: Vec<VariableIndex>,
}

impl CompiledConstraint {
    pub fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        self.order[layer]
    }
}

impl std::ops::Deref for CompiledConstraint {
    type Target = MddStructure;

    fn deref(&self) -> &MddStructure {
        &self.structure
    }
}

impl std::ops::Index<crate::mdd::NodeIndex> for CompiledConstraint {
    type Output = crate::mdd::Node;

    fn index(&self, index: crate::mdd::NodeIndex) -> &Self::Output {
        &self.structure[index]
    }
}

impl std::ops::Index<crate::mdd::EdgeIndex> for CompiledConstraint {
    type Output = crate::mdd::Edge;

    fn index(&self, index: crate::mdd::EdgeIndex) -> &Self::Output {
        &self.structure[index]
    }
}

impl crate::mdd::MddView for CompiledConstraint {
    fn root(&self) -> crate::mdd::NodeIndex {
        self.structure.root()
    }

    fn sink(&self) -> crate::mdd::NodeIndex {
        self.structure.sink()
    }

    fn number_layers(&self) -> usize {
        self.structure.number_layers()
    }

    fn number_nodes_in_layer(&self, layer: usize) -> usize {
        self.structure.number_nodes_in_layer(layer)
    }

    fn nodes_in_layer(&self, layer: usize) -> impl Iterator<Item = crate::mdd::NodeIndex> + '_ {
        self.structure.nodes_in_layer(layer)
    }

    fn is_unsat(&self) -> bool {
        self.structure.is_unsat()
    }
}

impl crate::mdd::MddViewWithOrder for CompiledConstraint {
    fn decision_at_layer(&self, layer: usize) -> VariableIndex {
        self.order[layer]
    }
}

/// Computes `constraint`'s real branching order the cheap way -- `OrderingHeuristic::get_order`
/// plus `enforce_precedence_order`, exactly the two steps `Mdd::new` itself runs before compiling
/// -- without compiling a full `Mdd` for it. `problem`/`constraint` here are the *real* one.
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

/// Compiles `constraint` (a real constraint of `problem`) into a `CompiledConstraint`: its
/// structure via `arena` (shared with every other constraint of the same shape and `max_width`),
/// its own real branching order computed fresh. `domain_size` is the nominal domain size the
/// synthetic template variable is compiled with -- see the module doc; it must match the
/// network's configured domain size the same way `mdd_dataset::compile_constraint_mdds`'s
/// `domain_size` parameter already did. `max_width` is the `Mdd::refine` width the shared
/// template is compiled to -- `usize::MAX` for an always-exact caller like
/// `mdd_dataset::compile_constraint_mdds`, or a caller's own configured search-time budget like
/// `nls::decode::MddSamplingDecode`'s `MddCompilationConfig::max_width`.
pub fn compile_constraint(
    arena: &MddArena,
    problem: &Problem,
    constraint: ConstraintIndex,
    ordering: &OrderingHeuristic,
    domain_size: usize,
    max_width: usize,
) -> CompiledConstraint {
    let key = constraint_shape_key(problem, constraint, domain_size, max_width);
    let structure = arena.structure_for(&key);
    let order = compute_constraint_order(problem, constraint, ordering);
    debug_assert_eq!(
        order.len() + 1,
        structure.number_layers(),
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
    use crate::modelling::{all_different, gcc, not_equals};

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
            3,
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g2,
            &OrderingHeuristic::MinDomMaxLinked,
            3,
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
            3,
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g2,
            &OrderingHeuristic::MinDomMaxLinked,
            3,
            usize::MAX,
        );
        assert!(!Arc::ptr_eq(&a.structure, &b.structure));
        assert_eq!(arena.len(), 2);
    }

    /// `AllDifferent`'s shape-key variant carries its own `arity` (see `ConstraintShapeKey`'s
    /// doc) -- confirms e.g. `AllDifferent(x, y, z)` and `AllDifferent(x, y, z, t)` still don't
    /// collide now that grouping is gone and the key is just the constraint's own shape.
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
            4,
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            g4,
            &OrderingHeuristic::MinDomMaxLinked,
            4,
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
            3,
            usize::MAX,
        );
        let b = compile_constraint(
            &arena,
            &problem,
            e2,
            &OrderingHeuristic::MinDomMaxLinked,
            3,
            usize::MAX,
        );
        assert!(
            Arc::ptr_eq(&a.structure, &b.structure),
            "symmetric NotEquals edges with different-degree endpoints must still dedup"
        );
        assert_eq!(arena.len(), 1);
    }
}
