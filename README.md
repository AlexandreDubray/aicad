# AICAD

![](assets/logo.png#gh-light-mode-only)
![](assets/logo_rev.png#gh-dark-mode-only)

MDD compilation for constraint satisfaction with search strategies built on top of it.
Compile a set of constraints into a decision diagram, query it, or solve it, with or without a
trained neural network.

## Python

The main entry point for using the library is the `pyaicad` Python package.

```python
import pyaicad as ac

model = ac.Model()
x = model.int_vars((4, 4), domain=range(4))              # a 4 x 4 grid of variables
model.add(ac.all_different(row) for row in x)             # latin square: rows ...
model.add(ac.all_different(col) for col in zip(*x))       # ... and columns

mdd = ac.compile(model)                                   # exact diagram; compile(model, width=8) relaxes
mdd.count()                                               # 576
ac.solve(model).assignment                                # solve by compilation (the default)
```

- **Model**: `Model`, `all_different`, `not_equal`, `equal`, `among`, `at_least`, `sum_equals`,
  `gcc`, `regular`.
- **Compile**: `compile(model, width=None)` returns an `MDD` that can be counted, sampled and
  inspected. This function is useful to integrate MDD-based reasoning inside other project.
- **Solve**: `solve(model_or_models, method, limits)` with `Compilation(...)` or `NLS(...)`, giving
  a `Result` (status, assignment, runtimes). Status is `FEASIBLE` or `INFEASIBLE` if the search find
  a solution or proves the problem has no solution. In case of timeout, status is `UNKNOWN`.
- **Learn**: `pyaicad.learning` trains the networks used by `NLS`: `Dataset`, `fit` with a named
  recipe (`"consformer"`, `"consformer-mdd"`), `load`.

`examples/` has a quick start and a complete train-then-solve script. The models and experiments of
the papers are scripts in the same style.

Build the extension with `maturin develop` and run the tests with `pytest`.

## Rust

The Rust crate is the core the Python package is built on: `modelling`, `constraints`, `mdd` and
`sampling`. It is public so that other solvers can embed the MDD compiler, but it is
**experimental** and may change between minor versions. The modules above it (`learning`, `local_search`,
the bindings) are private.
