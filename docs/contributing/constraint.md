# Adding a constraint

A constraint is a Rust type that tells the MDD compiler what it forbids. The contract, the
protocol the compiler follows and the laws an implementation must obey are written once, in the
module documentation of `aicad::constraints` (the Rust documentation). This page lists the places
a constraint touches, in order, including the Python side. Start from `src/constraints/among.rs`,
the shortest constraint.

The set of constraints is **closed**: the compiler shares compiled structures between constraints of
the same shape through an enum (`ConstraintShapeKey`) that rebuilds each constraint. A constraint
therefore has to be added to this crate; it cannot be plugged in from another one.

## Steps

1. **Write the module** `src/constraints/<name>.rs`, with the `//!` header sections of the
   existing ones: meaning, how it compiles (property update, node merging, edge filtering), design
   notes, example. Say what an empty scope, a repeated variable and a changed domain do.
2. **Implement `Constraint`** and a private property type implementing `ConstraintProperty`. The
   constructor should panic, with a clear message, on input that cannot mean anything.
   `is_satisfied` is the specification: everything else is an implementation of it that can be wrong.
3. **Declare it** in `src/constraints/mod.rs` and add a builder in `builders.rs` that creates the
   constraint and adds it to a `Problem`.
4. **Add a variant to `ConstraintShapeKey`** (in `src/modelling/constraint.rs`) and to `arity`, and
   return it from `structural_key`. The key must hold everything the compiled structure depends on
   and nothing else, in a canonical form: sort sets, so that two declarations in a different order
   share a structure.
5. **Rebuild it in the arena**: add the case to `instantiate_constraint` in `src/mdd/arena.rs`. The
   match is exhaustive, so the compiler points at it.
6. **Expose it to Python**:
   - a method on `PyProblem` in `src/pyaicad/problem.rs` that checks its arguments and calls the
     builder (errors are `PyValueError`s with a message that names the argument);
   - a constraint class and a function in `python/pyaicad/model.py`, validating the arguments
     with the helpers there and calling the native method in `_apply`;
   - an export in `python/pyaicad/__init__.py`, an entry in `python/pyaicad/_core.pyi`, and a line
     in `docs/api.md`.
7. **Document it** where users look: the docstring of the Python function (with a small doctest)
   and, if it opens a new kind of model, a sentence in the modelling chapter of the Python and Rust guides.

## Worked example: `Sum`

`Sum` ($\sum_i x_i = t$) is the smallest constraint that needs a real summary, so every step above can be
read on it. All the Rust below is pulled from the source, not retyped.

**The idea.** A node of the diagram has to remember, for the paths that reach it, what the rest of
the diagram needs to know about the sum so far. Remembering every possible partial sum would give as
many nodes as sums. `Sum` keeps only the smallest and the largest, an interval, which is cheap and
lets the compiler discard an edge whose paths cannot reach the target.

**The property** (step 2) holds that interval. `isize::MAX` and `isize::MIN` are the sentinels of
"no path yet", which is the identity of the fold:

```{literalinclude} ../../src/constraints/sum.rs
:language: rust
:start-at: struct SumProperty {
:end-at: }
```

**Folding and merging** are the two operations the compiler uses to build and shrink properties. An
edge of value `v` shifts the interval of its parent by `v` (only in the scope; other layers copy it),
and several parents are combined by widening; merging two nodes does the same.

```{literalinclude} ../../src/constraints/sum.rs
:language: rust
:start-at: fn update(&mut self, other: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
:end-before: fn order_key
:dedent: 4
```

(The `downcast_ref` and its `panic!` only check that the compiler did not hand a property of another
kind of constraint to this one; every property type repeats it.)

**Filtering** is what makes the diagram smaller. An edge between a parent whose sums lie in
$[m_p, M_p]$ and a child in $[m_c, M_c]$ lies on paths whose total is somewhere in
$[m_p + v + m_c,\ M_p + v + M_c]$; if the target is outside, the edge goes:

```{literalinclude} ../../src/constraints/sum.rs
:language: rust
:start-at: // A node without any path (the sentinels)
:end-at: local_min > self.target || local_max < self.target
:dedent: 8
```

This is bounds reasoning: a total inside the interval that no path reaches is not detected, so a
compiled diagram may keep paths that do not sum to the target until nodes are split enough for the
intervals to become points. That is why the tests compare with `is_satisfied` and a brute-force
count, and why the module header says it.

**The key** (step 4) holds the arity and the target, and nothing else:

```{literalinclude} ../../src/constraints/sum.rs
:language: rust
:start-at: fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
:end-before: fn update_variable_ordering
:dedent: 4
```

and the arena rebuilds a `Sum` from it on a synthetic problem (step 5):

```{literalinclude} ../../src/mdd/arena.rs
:language: rust
:start-at: ConstraintShapeKey::Sum { arity: _, target } => {
:end-at: synthetic_problem))
:dedent: 8
:append: "}"
```

The constructor refuses a repeated variable, because the property cannot tell how many times a layer
counts:

```{literalinclude} ../../src/constraints/sum.rs
:language: rust
:start-at: let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
:end-before: Self {
:dedent: 8
```

**The builder and the Python side** (steps 3 and 6) are a few lines each:

```rust
// src/constraints/builders.rs
pub fn sum(problem: &mut Problem, variables: Vec<VariableIndex>, target: isize) -> ConstraintIndex {
    problem.add_constraint(Sum::new(variables, target, problem))
}

// src/pyaicad/problem.rs, a method of PyProblem
fn add_sum(&mut self, scope: Vec<usize>, target: isize) -> PyResult<()> {
    let vars = scope.into_iter().map(VariableIndex).collect();
    sum(self.mutate()?, vars, target);
    Ok(())
}
```

```python
# python/pyaicad/model.py
@dataclass(frozen=True)
class _SumEquals(Constraint):
    scope: Tuple[Var, ...]
    target: int

    def _apply(self, native: Any) -> None:
        native.add_sum(_idx(self.scope), self.target)


def sum_equals(variables: Iterable[Var], target: int) -> Constraint:
    """The variables add up to ``target``. Values may be negative."""
    scope = _variables(variables, "sum_equals", distinct=True)
    return _SumEquals(scope, operator.index(target))
```

Python validates the shape of the input (`distinct=True` rejects a repeated variable with a
`ModelError` that names the function); Rust owns what the constraint means.

## Tests

- The compiled MDD must contain every solution at every width and be exact once fully refined,
  compared with `is_satisfied` and a brute-force enumeration on small instances.
- Add an instance of the constraint to `test_contract` at the end of `src/constraints/mod.rs`,
  which checks the laws of the contract (identity, merge, update direction).
- Add it to the oracle in `tests/oracle.py`, so that the Python tests compare compiled solution
  counts with a brute-force count, and add a Python test for the argument validation.
