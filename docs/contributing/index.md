# Contributing

This part of the documentation is for people who change `aicad` itself or build on its Rust API:
adding a constraint, a destroy operator, a local-search architecture or a neural network. Using the
library from Python is covered by the [Python guide](../python-guide/modelling.md) and the [Rust guide](../rust-guide/modelling.md).

The [all-items Rust API](https://alexandredubray.github.io/aicad/rust-api/aicad/index.html) lists every item of the crate, private ones included.

Read [](orientation.md) first, then the chapter for what you want to add. Every chapter ends with the
tests that a change must come with, and [](checklist.md) lists what to run before you open a pull
request.

```{toctree}
:maxdepth: 1

orientation
constraint
destroy
search
network
checklist
```
