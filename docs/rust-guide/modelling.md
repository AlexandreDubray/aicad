# Modelling

The Rust guide follows the [Python guide](../python-guide/modelling.md) chapter by chapter, with the
same problems and the same examples. Every piece of code in it is taken from `tests/rust_guide`,
which `cargo test` compiles and runs, so it matches the library.

A problem is a `Problem`: variables with finite integer domains, and constraints over them.
Everything else in the library starts from a problem. The constraints are built by functions of
`aicad::constraints` that add them to a problem: `all_different`, `not_equals`, `equal` (in
`aicad::modelling`), `among`, `at_least`, `sum`, `gcc` and `regular`.

## The model

`add_variables` creates variables and returns their indices, so a roster of 3 nurses over 5 days is
three lists of five. The constraints take the problem, the indices of their scope and their
parameters. In the Python guide the same model is built with `int_vars` and `Model.add`.

```{literalinclude} ../../tests/rust_guide/modelling.rs
:language: rust
:start-after: // [docs:start model]
:end-before: // [docs:end model]
```

`regular` states a pattern on a sequence of variables with an automaton: here a night shift is never
directly followed by a day shift. State 0 is "not after a night", state 1 is "after a night"; the
columns of the table are the sorted values of the domain (off, day, night), and `None` forbids a
transition. Constructors panic, with a message that names the problem, on input that cannot mean
anything, such as a repeated variable in the scope of `all_different`.

## Checking an assignment

An assignment lists one value per variable, in the order of their indices. `Problem::is_solution`
tells whether it satisfies every constraint, and `is_satisfied` of a constraint whether it satisfies
that one.

```{literalinclude} ../../tests/rust_guide/modelling.rs
:language: rust
:start-after: // [docs:start check]
:end-before: // [docs:end check]
```

Finding a solution is the job of the next chapters.
