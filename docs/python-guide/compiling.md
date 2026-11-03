---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Compiling a model

A {class}`~pyaicad.Compiler` turns a model into a decision diagram, an {class}`~pyaicad.MDD`. In
an **exact** diagram every path from the root to the sink is one solution of the model, so
questions about the solutions are answered by reading the diagram instead of searching: how many
there are, a random one, whether there is any.

```{code-cell} python
import pyaicad as ac

# A 4 x 4 latin square: every row and every column holds each of 0..3 once.
n = 4
model = ac.Model()
grid = model.int_vars((n, n), domain=range(n))
model.add(ac.all_different(row) for row in grid)
model.add(ac.all_different([grid[i][j] for i in range(n)]) for j in range(n))

compiler = ac.Compiler(model)
mdd = compiler.compile()
print(mdd)
print("solutions:", mdd.count())
```

A compiler is made for a model as it is at that moment. If you add constraints afterwards, make a
new compiler.

## Heuristics

Two heuristics drive the compilation. They decide how large the diagram gets, never what it
contains.

`order`
: the variable of each layer, from the root to the sink. `"auto"`, the default, is the
  min-dom-max-linked heuristic: it starts with the variable of smallest domain, then takes the one
  most linked to those already placed. A list of variables gives your own order.

`merge`
: which nodes are joined when a layer is full, so it only matters for relaxed diagrams (next
  chapter). The default is `"state_similarity"`.

```{code-cell} python
auto = ac.Compiler(model, order="auto").compile()
print("nodes per layer, auto order:", auto.widths)

row_by_row = [grid[i][j] for i in range(n) for j in range(n)]
by_rows = ac.Compiler(model, order=row_by_row).compile()
print("nodes per layer, row by row:", by_rows.widths)

assert by_rows.count() == auto.count()   # same solutions, other shape
```

## Reading the diagram

```{code-cell} python
ac.set_seed(0)
for values in mdd.sample(3):             # in an exact diagram, every solution is equally likely
    assert model.check(values)
    print(values)

print("one solution:", mdd.solution())
```

A solution is a list of values in the order of `model.variables`, like any assignment.

## Unsatisfiable models

Contradictory constraints are detected during compilation, and an exact diagram is then a proof.

```{code-cell} python
model.add(ac.equal(grid[0][0], 2))
model.add(ac.equal(grid[0][1], 2))       # the same value twice in the first row
impossible = ac.Compiler(model).compile()
print("unsatisfiable:", impossible.is_unsat, "- solutions:", impossible.count(), "-", impossible.solution())
```

## Drawing

`to_dot` returns a [Graphviz](https://graphviz.org) description (`dot -Tpdf`) of a diagram,
practical for small ones.

```{code-cell} python
small = ac.Model()
x = small.int_vars(3, domain=range(3))
small.add(ac.all_different(x))
print(ac.Compiler(small).compile().to_dot()[:200], "...")
```

## Memory

An exact diagram can be too large to build, and nothing bounds the memory of an unbounded
compilation. The next chapter shows how to bound the size of the diagram.
