# Local search

A compiled diagram answers questions about a problem; a **local search** finds solutions to
problems that are too large to compile. Every search runs the same loop. It starts from a random
assignment, and at each iteration *destroys* some variables (frees them) and *repairs* them (picks
new values that satisfy the constraints as well as it can), until it finds a solution or reaches a
limit. The searches differ only in how they repair:

- the **belief propagation** search repairs with belief propagation on the diagrams of the
  constraints, from uniform probabilities. It needs nothing but the problems;
- the **neural prior** searches (`NeuralPrior`, `NeuralPriorBp`) start from the predictions of a
  trained network (see [](learning.md); needs the `learning` feature).

A local search finds solutions but cannot prove that there are none: to prove it, compile the
problem. Many problems can be searched together: they are stepped as one batch. A search that is not
one of these, or a different destroy operator, can be plugged in from your own crate: see
*Contributing*.

## A problem family

The examples use random graph colourings: a graph with 30 nodes and 3 colours, drawn so that a
colouring is known to exist. Generating problems is up to you; the library only sees the problems.

```{literalinclude} ../../tests/rust_guide/local_search.rs
:language: rust
:start-after: // [docs:start coloring]
:end-before: // [docs:end coloring]
```

## Running a search

`solve` takes the problems (all of them are stepped together as one batch unless `batch_size` says
otherwise), a `SolveConfig` and, for the neural searches, a checkpoint. The configuration holds the
limits, which apply to each batch (whichever is reached first stops the search), the seed that makes
the run reproducible, and the architecture of the search with its settings.

The default search destroys every variable at each iteration, which is a restart from scratch.
Destroying only a fraction makes it a real local search; `destroy_kind` chooses which ones: at
random, where the constraints are violated (`Worst`), or around a variable (`Related`).

```{literalinclude} ../../tests/rust_guide/local_search.rs
:language: rust
:start-after: // [docs:start search]
:end-before: // [docs:end search]
```

## Results

A `Solution` carries the assignment when one was found, its status, and how the search went. The
same seed gives the same run: the seed decides where the search starts and every choice it makes.

```{literalinclude} ../../tests/rust_guide/local_search.rs
:language: rust
:start-after: // [docs:start result]
:end-before: // [docs:end result]
```

`aicad::set_num_threads` fixes how many threads the library uses for its own work.
