# Orientation

## A Rust crate and a Python package

`aicad` is a Rust crate, and `pyaicad` is the Python package built from it. The Python bindings are
thin wrappers over public Rust items: they translate arguments and errors, and hold no algorithm.
The rule behind the design is that **everything that can be done from Python can be done from
Rust**, so that applications that care about runtime can use the Rust API directly. A feature is
therefore implemented in Rust first and exposed to Python second.

Python code is not pluggable. Crossing between Python and Rust in an inner loop is slow, so a new
constraint, destroy operator, search architecture or network is written in Rust, in this crate or
in your own.

## Layers

The modules are layered, and a module only depends on the ones before it:

```
utils < modelling < constraints < mdd < sampling < bp < destroy < neural < learning < local_search < pyaicad
```

`scripts/check_layering.py` enforces the order in CI. When a change needs something from a later
layer, the code usually belongs in a later layer.

## Features

| Feature    | What it adds                                                          |
|------------|-----------------------------------------------------------------------|
| (default)  | Modelling, constraints, MDD compiler, sampling, belief propagation, destroy operators, the local-search engine and the belief-propagation search. No heavy dependency. |
| `learning` | The [burn](https://burn.dev) dependency: `neural`, `learning`, and the neural searches. |
| `python`   | The PyO3 bindings (`pyaicad`). Combine with `learning` for the full wheel. |

Code that needs burn is gated with `#[cfg(feature = "learning")]`. A change must build and pass its
tests with and without the feature (see [](checklist.md)). The Python package is built in two
flavours, *slim* (`--features python`) and *full* (`--features python,learning`); the Python side
reads `pyaicad.HAS_LEARNING` and raises `FeatureUnavailableError` when the slim build is asked for
something that needs learning.

## Where the documentation is

- the **Python guide**, the **Rust guide** and the Python API reference: this site;
- the **public Rust API**: docs.rs, built with the `learning` feature;
- the **contributor Rust documentation**, private items included: built in CI with
  `cargo doc --document-private-items`, next to this site.

Every module starts with a `//!` header in the same shape: a one-line summary, what it means, how it
works, design notes, and a runnable example. Follow it for new modules; `cargo doc` runs the
examples as tests and fails on a broken link.

## Plugging code in from another crate

Several extension points are open on purpose, so that work can live in your own crate without being
merged here. Each is demonstrated by a test that is compiled as a separate crate, and that therefore
fails to build if the public API stops being enough:

- a **local-search architecture**: `tests/custom_search.rs`;
- a **neural network**, trained and used for search: `tests/custom_network.rs`;
- a **destroy operator** is a trait object (`DestroyOperator`) and goes into any search that takes one.

Constraints are the exception: the set is closed, because the compiler shares compiled structures
through an enum of shapes (see [](constraint.md)).
