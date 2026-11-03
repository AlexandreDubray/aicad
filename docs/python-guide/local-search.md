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

# Local search

A compiled diagram answers questions about a model; a **local search** finds solutions to problems
that are too large to compile. Every search of `pyaicad` runs the same loop. It starts from a random
assignment, and at each iteration *destroys* some variables (frees them) and *repairs* them (picks
new values that satisfy the constraints as well as it can), until it finds a solution or reaches
a limit. The searches differ only in how they repair:

- {class}`~pyaicad.BeliefPropagationSearch` repairs with belief propagation on the diagrams of the
  constraints, from uniform probabilities. It needs nothing but the problem.
- {class}`~pyaicad.NeuralPriorSearch` and {class}`~pyaicad.NeuralPriorBPSearch` start from the
  predictions of a trained network (see [](learning.md); full build).

A local search finds solutions but cannot prove that there are none: to prove it, compile the model.

## A problem family

The examples use random graph colourings: a graph with 30 nodes and 3 colours, drawn so that a
colouring is known to exist. Generating problems is up to you; the library only sees the models.

```{code-cell} python
import random
import pyaicad as ac

def coloring(rng, nodes, colours, degree):
    """A random feasible graph colouring: hide a colouring, draw edges between nodes of other colours."""
    hidden = [rng.randrange(colours) for _ in range(nodes)]
    model = ac.Model()
    colour = model.int_vars(nodes, domain=range(colours))
    pairs = [(i, j) for i in range(nodes) for j in range(i + 1, nodes) if hidden[i] != hidden[j]]
    probability = min(1.0, degree * nodes / 2 / len(pairs))
    for i, j in pairs:
        if rng.random() < probability:
            model.add(ac.not_equal(colour[i], colour[j]))
    return model

rng = random.Random(0)
problems = [coloring(rng, nodes=30, colours=3, degree=4.0) for _ in range(20)]
print(problems[0])
```

## Running a search

`local_search` takes a problem (or a list of problems, stepped together as one batch), a search and
its limits. A {class}`~pyaicad.Limits` applies to each problem separately; whichever limit is
reached first stops the search.

```{code-cell} python
limits = ac.Limits(time=5, iterations=300)

search = ac.BeliefPropagationSearch()
results = ac.local_search(problems, search, limits, seed=0)
print(f"destroy everything: solved {sum(r.found for r in results)} of {len(problems)}")
```

The default search destroys every variable at each iteration, which is a restart from scratch.
`with_` makes a copy of a search with some options changed. Destroying only a fraction of the
variables makes it a real local search; `destroy` chooses which ones: at `"random"`, where the
constraints are violated (`"worst"`), or around a variable (`"related"`).

```{code-cell} python
for destroy in ("random", "worst", "related"):
    variant = search.with_(destroy=destroy, destroy_fraction=0.2)
    results = ac.local_search(problems, variant, limits, seed=0)
    solved = [r for r in results if r.found]
    print(f"destroy {destroy:>8} 20%: solved {len(solved):>2} of {len(problems)}")
```

## Results

A {class}`~pyaicad.Result` carries the assignment when one was found, and how the search went.

```{code-cell} python
best = search.with_(destroy="worst", destroy_fraction=0.2)
result = ac.local_search(problems[0], best, limits, seed=0)
print(result.status.value, "after", result.iterations, "iterations")
if result.found:
    assert problems[0].check(result.assignment)
```
