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

# Relaxed diagrams and refinement

When an exact diagram is too large, compile a **relaxed** one: it has at most `width` nodes per
layer. It keeps every solution as a path and adds paths that are not solutions, so its path count
is an upper bound on the number of solutions.

```{code-cell} python
import pyaicad as ac

n = 4
model = ac.Model()
grid = model.int_vars((n, n), domain=range(n))
model.add(ac.all_different(row) for row in grid)
model.add(ac.all_different([grid[i][j] for i in range(n)]) for j in range(n))

compiler = ac.Compiler(model)
exact = compiler.compile().count()

mdd = compiler.compile(width=2)
print(f"width {mdd.width}: {mdd.count()} paths for {exact} solutions")
```

A path of a relaxed diagram is not necessarily a solution: check it against the model.

```{code-cell} python
relaxed = compiler.compile(width=256)
print(f"width 256: {relaxed.count()} paths")
print("a random path is a solution:", model.check(relaxed.sample(1)[0]))
```

One compiler hands out as many diagrams as you need, each independent of the others.

## Refinement

A diagram can be tightened in place instead of compiled again: `refine(width)` raises the
budget and splits nodes where the relaxation lost information. The path count only goes down, and
never below the number of solutions. `refine(None)` goes on until the diagram is as exact as the
propagation allows.

`compile` and `refine` also take `max_iterations`, the most passes over the layers that the
refinement makes. The later passes tighten the diagram little for what they cost, so a limit trades
a looser diagram, still a relaxation, for time; it is chosen at each call.

```{code-cell} python
mdd = compiler.compile(width=2)
for width in (4, 8, 32, None):
    mdd.refine(width)
    print(f"width {width}: {mdd.count()} paths, widest layer {max(mdd.widths)}")
assert mdd.count() == exact
```

## Freezing

A diagram that can be refined keeps its compilation alive: the constraints, their properties and
the heuristics, which is often most of its memory. Once it is tight enough, `freeze` drops all
that and keeps the diagram. Counting, sampling and drawing still work, refining does not.

```{code-cell} python
mdd.freeze()
print("frozen:", mdd.is_frozen, "- still", mdd.count(), "paths")
try:
    mdd.refine(8)
except RuntimeError as error:
    print("refining a frozen diagram fails:", error)
```

## Choosing what to compile

The `constraints` argument compiles a subset of the constraints. A variable that is in none of
them has no layer; `mdd.scope` lists the variables that do.

```{code-cell} python
rows_only = ac.Compiler(model, constraints=range(n)).compile()
print("rows alone:", rows_only.count(), "paths over", len(rows_only.scope), "variables")
```

The merge heuristic says which nodes are joined when a layer is full; at the same width it can
give tighter or looser diagrams.

```{code-cell} python
column_major = [grid[i][j] for j in range(n) for i in range(n)]
for merge in ("less_relaxed", "most_likely", "state_similarity"):
    mdd = ac.Compiler(model, order=column_major, merge=merge).compile(width=4)
    print(f"merge={merge!r}: {mdd.count()} paths")
```

## Solutions and proofs from relaxed diagrams

An exact diagram decides the model: it has a solution or it is unsatisfiable. A relaxed one
decides less. `solution()` returns a solution only if the diagram holds a path of exact nodes, so
`None` proves nothing, except when the diagram is unsatisfiable, which is a proof even for a
relaxed diagram.

```{code-cell} python
def pigeons(pigeons, holes):
    """Every pigeon in a hole, no two in the same hole: feasible iff pigeons <= holes."""
    model = ac.Model()
    hole = model.int_vars(pigeons, domain=range(holes))
    model.add(ac.all_different(hole))
    return model

compiler = ac.Compiler(pigeons(5, 5))
for width in (1, 4, None):
    mdd = compiler.compile(width=width)
    print(f"5 pigeons, 5 holes, width {width}: unsatisfiable={mdd.is_unsat}, solution={mdd.solution()}")

mdd = ac.Compiler(pigeons(5, 4)).compile(width=1)
print("5 pigeons, 4 holes, width 1: unsatisfiable =", mdd.is_unsat)
```
