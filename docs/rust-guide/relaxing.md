# Relaxed diagrams and refinement

When an exact diagram is too large, compile a **relaxed** one: it has at most `width` nodes per
layer. It keeps every solution as a path and adds paths that are not solutions, so its path count
is an upper bound on the number of solutions. `count_solutions` returns an arbitrary-precision
integer, since a relaxed diagram can hold far more paths than any fixed-width integer.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start relaxed]
:end-before: // [docs:end relaxed]
```

A path of a relaxed diagram is not necessarily a solution: check it against the problem.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start relaxed_path]
:end-before: // [docs:end relaxed_path]
```

Compilers are independent: make as many as you need for the same problem.

## Refinement

A diagram can be tightened in place instead of compiled again: `refine(width)` raises the budget
and splits nodes where the relaxation lost information. The path count only goes down, and never
below the number of solutions. `refine(usize::MAX)` goes on until the diagram is as exact as the
propagation allows.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start refine]
:end-before: // [docs:end refine]
```

## Freezing

A compiler keeps what refining needs: the constraints, their properties and the heuristics, which
is often most of the memory. Once the diagram is tight enough, `into_mdd` consumes the compiler and
keeps the diagram. In Python `freeze` does the same on the diagram object, and refining it
afterwards is a runtime error; in Rust it is a type error, since only a `Compiler` has `refine`.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start freeze]
:end-before: // [docs:end freeze]
```

## Choosing what to compile

The constraints given to `Compiler::new` are the ones that are compiled. A variable that is in none
of them has no layer; `number_decisions` and `decision_at_layer` list the variables that do.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start subset]
:end-before: // [docs:end subset]
```

The merge heuristic says which nodes are joined when a layer is full; at the same width it can give
tighter or looser diagrams. `MostLikely` carries the probability of every value of every variable.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start merge]
:end-before: // [docs:end merge]
```

## Solutions and proofs from relaxed diagrams

An exact diagram decides the problem: it has a solution or it is unsatisfiable. A relaxed one
decides less. `get_solution` returns a solution only if the diagram holds a path of exact nodes, so
`None` proves nothing, except when the diagram is unsatisfiable, which is a proof even for a
relaxed diagram.

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start pigeons]
:end-before: // [docs:end pigeons]
```

```{literalinclude} ../../tests/rust_guide/relaxing.rs
:language: rust
:start-after: // [docs:start decide]
:end-before: // [docs:end decide]
```
