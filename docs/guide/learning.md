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

# Training a network

```{note}
This chapter needs the full build of pyaicad (see [](../install.md)).
```

The neural searches repair with the predictions of a network. A network is trained on a set of
problems, one problem per example. The problems have their own constraints but the same variables,
which is what a ConsFormer network needs. The trained network lives in a *checkpoint* directory,
so a network trained once can be used by any later program.

```{code-cell} python
:tags: [hide-input]

import random
import tempfile
import pyaicad as ac
from pyaicad import learning

def coloring(rng, nodes, colours, degree):
    hidden = [rng.randrange(colours) for _ in range(nodes)]
    model = ac.Model()
    colour = model.int_vars(nodes, domain=range(colours))
    pairs = [(i, j) for i in range(nodes) for j in range(i + 1, nodes) if hidden[i] != hidden[j]]
    probability = min(1.0, degree * nodes / 2 / len(pairs))
    for i, j in pairs:
        if rng.random() < probability:
            model.add(ac.not_equal(colour[i], colour[j]))
    return model
```

## Data

`Dataset.split` divides the problems you train on into a training and a validation set. The
validation problems pick the best network during training. (`fit` also takes a plain list and splits
it 80/20 itself.)

There is no test set in a `Dataset` on purpose. The problems you evaluate on are yours to keep
apart: generate them separately and never pass them to `fit`, so that the network has certainly not
seen them.

```{code-cell} python
rng = random.Random(0)
training_problems = [coloring(rng, nodes=20, colours=3, degree=3.5) for _ in range(108)]
test_problems = [coloring(rng, nodes=20, colours=3, degree=3.5) for _ in range(12)]

data = learning.Dataset.split(training_problems, 0.1, seed=0)
print(len(data.train), "training,", len(data.validation), "validation,", len(test_problems), "test")
```

## Recipes

A *recipe* is a named way of training: the loss, the architecture and the optimiser. Settings of a
recipe are changed by name.

```{code-cell} python
print(sorted(learning.RECIPES))
```

`consformer-mdd` scores the network with the exact probability that its predictions satisfy each
constraint, computed on the diagram of the constraint.

```{code-cell} python
checkpoint_dir = tempfile.mkdtemp()
checkpoint = learning.fit(
    data,
    "consformer-mdd",
    checkpoint_dir=checkpoint_dir,
    seed=0,
    epochs=4,
    validation_interval=2,
    embedding_size=16,
    hidden_size=16,
    expand_size=32,
    num_heads=2,
)
print("network saved in", checkpoint.path)
```

`fit` keeps the best network in the checkpoint directory. The network here is tiny and trained for
a few epochs, so it will not beat belief propagation alone on problems this easy; the example shows
the workflow.

## Searching with the network

The checkpoint is a parameter of the neural searches, which otherwise take the same options as
the belief-propagation one. `NeuralPriorSearch` repairs with the network directly;
`NeuralPriorBPSearch` refines its predictions with belief propagation on the constraint diagrams
before choosing.

```{code-cell} python
limits = ac.Limits(time=10, iterations=100)
searches = {
    "belief propagation": ac.BeliefPropagationSearch(destroy="worst", destroy_fraction=0.2),
    "neural prior": ac.NeuralPriorSearch(checkpoint, destroy="worst", destroy_fraction=0.2),
    "neural prior + BP": ac.NeuralPriorBPSearch(checkpoint, destroy="worst", destroy_fraction=0.2),
}
for name, search in searches.items():
    results = ac.local_search(test_problems, search, limits, seed=0)
    print(f"{name:>20}: solved {sum(r.found for r in results)} of {len(results)} test problems")
```

## Using a checkpoint later

`learning.load` reads a checkpoint back. A network is tied to the problems it was trained on:
same number of variables, same values. New problems of that family are solved with it directly.
`batch_size=1` solves them one after the other, each with its own limits, as a deployment would;
by default all the problems are stepped together on the device.

```{code-cell} python
checkpoint = learning.load(checkpoint_dir)
print("a network over", checkpoint.domain_size, "values")

new_problems = [coloring(random.Random(123 + k), nodes=20, colours=3, degree=3.5) for k in range(3)]
search = ac.NeuralPriorBPSearch(checkpoint, destroy="worst", destroy_fraction=0.2, batch_size=1)
for model, result in zip(new_problems, ac.local_search(new_problems, search, limits, seed=0)):
    print(result.status.value, "after", result.iterations, "iterations")
    if result.found:
        assert model.check(result.assignment)
```

```{note}
Compilation and local search are reproducible given their seed. Training is seeded too, but
bit-for-bit reproducibility of a training run is not guaranteed.
```
