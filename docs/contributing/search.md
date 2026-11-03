# Adding a local-search architecture

A local search repeatedly destroys part of each assignment and repairs it, until every problem is
solved or the budget is spent. The engine (`LocalSearch`) runs that loop and keeps the budget and
the list of solved problems. What a step does belongs to the **architecture**, which implements
two small traits in `src/local_search/mod.rs`:

- `SearchArchitecture::setup(problems)` builds the state of a search over a batch of problems, or
  returns an error naming the restriction when the architecture cannot search them together.
- `SearchState::step(problems, assignments, pending, rng)` performs one iteration: it destroys and
  repairs the assignments of the rows flagged in `pending` and leaves the others as they are.
  `solved` and `retain` let it free per-row data; both do nothing by default.

The engine owns the assignments; the state holds only what the architecture needs besides them.

## From another crate

Implement the two traits, wrap the architecture in `LocalSearch::new(Box::new(...))` and run it with
`run_batched`, which batches the problems, applies the budget and seeds each batch. The helpers
`destroy_rows` and `row_seeds` are public, so that a step can reuse the destroy operators and keep
its randomness reproducible. `tests/custom_search.rs` is a complete, compiled example: an
architecture that redraws the destroyed variables at random.

## Into this crate, and into Python

1. Write the architecture in `src/local_search/<name>.rs`, with a settings struct next to it
   (`Default` included). The existing `bp_search.rs` is the model for a search that needs no network;
   `neural_prior.rs` for one that does.
2. Add a variant holding the settings to `SearchConfig` in `src/local_search/config.rs` (gated on
   `learning` if it needs burn), and handle it in `solve` in `src/local_search/mod.rs`.
3. Add a variant to `PySearch` in `src/pyaicad/local_search.rs` with its conversion. A pyo3 enum
   cannot gate variants on a feature, so a variant that needs `learning` stays and the conversion
   returns an error without it.
4. Add the search class to `python/pyaicad/search.py` (frozen dataclass, defaults, `with_`,
   validation in `__post_init__`, `require_learning` if it needs the full build), export it, and add it
   to `_core.pyi` and `docs/api.md`.

## Worked example: a random repair

`tests/custom_search.rs` is compiled as a separate crate, so everything in it is public API. The
architecture destroys some variables with any `DestroyOperator` and repairs them by redrawing their
values at random: the simplest repair there is.

```{literalinclude} ../../tests/custom_search.rs
:language: rust
:start-after: // [docs:start architecture]
:end-before: // [docs:end architecture]
```

Reading it from the top:

- `Redraw` is the architecture. It holds what is common to all problems (here the destroy operator)
  and is shared between runs.
- `setup` is called once per batch, and builds the state. It is the place to refuse what the
  architecture cannot search, with an error that says why; the engine passes that error to the caller
  of `run_batched`.
- `step` does one iteration for all the rows at once. `destroy_rows` asks the operator for the
  variables of each pending row, in parallel and reproducibly (it draws one seed per row from `rng`),
  then the repair writes the new values straight into `assignments`.
- Rows not in `pending` are already solved: leave them as they are.

Running it takes the engine, the problems, a batch size, a budget and a seed:

```{literalinclude} ../../tests/custom_search.rs
:language: rust
:start-after: // [docs:start run]
:end-before: // [docs:end run]
```

A real repair replaces the loop in `step`. `bp_search.rs` repairs with belief propagation on the
diagrams of the constraints, and `neural_prior.rs` with the logits of a network; both keep per-row
data in their state, which is what `solved` and `retain` are for.

## Tests

- It solves small satisfiable problems, and leaves already-satisfying assignments alone.
- Two runs with the same seed give the same iterations.
- `setup` refuses what the architecture cannot search, with a message that names the restriction.
- A batch with several problems gives each the same result as running it alone.
