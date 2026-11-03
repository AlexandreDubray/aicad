# AICAD

![](docs/_static/logo.png#gh-light-mode-only)
![](docs/_static/logo_rev.png#gh-dark-mode-only)

_Blazingly fast_ MDD compilation for constraint satisfaction problems with search strategies built on top of it.
Compile a set of constraints into a decision diagram, query it, or solve it, with or without a
trained neural network.

`aicad` is a Rust crate, and `pyaicad` is the Python package built on it.

```python
import pyaicad as ac

model = ac.Model()
x = model.int_vars((4, 4), domain=range(4))              # a 4 x 4 grid of variables
model.add(ac.all_different(row) for row in x)             # latin square: rows ...
model.add(ac.all_different(col) for col in zip(*x))       # ... and columns

mdd = ac.Compiler(model).compile()                        # exact diagram
mdd.count()                                               # 576
```

## Documentation

Everything about using and extending the library is in the
**[aicad documentation](https://alexandredubray.github.io/aicad/)**:

- [Installation](https://alexandredubray.github.io/aicad/install.html): the Python package (slim and
  full builds), the Rust crate, and building from source.
- The Python guide and the Rust guide: modelling, compiling, relaxing, local search and learning,
  with the same examples in both languages.
- The Python API reference, and the [Rust API](https://docs.rs/aicad).
- [Contributing](https://alexandredubray.github.io/aicad/contributing/index.html): adding a
  constraint, a destroy operator, a search architecture or a neural network.

## License

MIT, see [LICENSE](LICENSE).
