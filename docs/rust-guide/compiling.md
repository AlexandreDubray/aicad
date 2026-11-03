# Compiling a problem

A `Compiler` turns a problem into a decision diagram, an `Mdd`. In an **exact** diagram every path
from the root to the sink is one solution of the problem, so questions about the solutions are
answered by reading the diagram instead of searching: how many there are, a random one, whether
there is any.

The examples of this chapter and the next use a latin square, which this function builds, with its
grid of variables:

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start latin]
:end-before: // [docs:end latin]
```

The diagram is read through methods of `Mdd`, and two helpers keep the examples short: the width of
every layer, and a sampled path put back in the order of the variables (a path lists its values in
the order of the layers):

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start helpers]
:end-before: // [docs:end helpers]
```

## Compiling

`Compiler::new` takes the problem (as an `Arc`, since compilation shares it), the heuristics and the
constraints to compile. It starts from the loosest diagram; `refine(width)` tightens it up to
`width` nodes per layer, and `usize::MAX` lifts the limit, so the diagram is exact.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start exact]
:end-before: // [docs:end exact]
```

A compiler works on the problem it was given: the `Arc` makes the problem final once compiled. To
add a clue, build the problem again, or before wrapping it.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start clue]
:end-before: // [docs:end clue]
```

## Heuristics

Two heuristics drive the compilation. They decide how large the diagram gets, never what it
contains.

`ordering`
: the variable of each layer, from the root to the sink. `OrderingHeuristic::MinDomMaxLinked`, the
  default, starts with the variable of smallest domain, then takes the one most linked to those
  already placed. `OrderingHeuristic::Custom` gives your own order, as variable indices.

`merge`
: which nodes are joined when a layer is full, so it only matters for relaxed diagrams (next
  chapter). The default is `MergeHeuristic::StateSimilarity`.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start heuristics]
:end-before: // [docs:end heuristics]
```

## Reading the diagram

A sample draws a path with every value of a layer equally likely, and in an exact diagram every
solution is then equally likely. `set_seed` makes the draws reproducible.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start sample]
:end-before: // [docs:end sample]
```

`get_solution` returns the values with the variables they belong to, in the order of the layers.

## Unsatisfiable problems

Contradictory constraints are detected during compilation, and an exact diagram is then a proof.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start unsat]
:end-before: // [docs:end unsat]
```

## Drawing

`as_graphviz` returns a [Graphviz](https://graphviz.org) description (`dot -Tpdf`) of a diagram,
practical for small ones.

```{literalinclude} ../../tests/rust_guide/compiling.rs
:language: rust
:start-after: // [docs:start dot]
:end-before: // [docs:end dot]
```

## Memory

An exact diagram can be too large to build, and nothing bounds the memory of an unbounded
compilation. The next chapter shows how to bound the size of the diagram.
