"""Train a network on latin square problems, then solve unseen ones with it.

Run with ``python examples/train_latin_squares.py``. It trains a tiny network for a few seconds on
CPU; the models and recipes of the experiments are used the same way, with other problems.
"""

import random
import tempfile

import pyaicad as ac
from pyaicad import learning

n = 4


def latin_square(clues):
    """A latin square problem; ``clues`` maps (row, column) to a given value."""
    model = ac.Model()
    grid = model.int_vars((n, n), domain=range(n))
    model.add(ac.all_different(row) for row in grid)
    model.add(ac.all_different([grid[i][j] for i in range(n)]) for j in range(n))
    for (i, j), value in clues.items():
        model.add(ac.equal(grid[i][j], value))
    return model


def random_problem(rng):
    cells = rng.sample([(i, j) for i in range(n) for j in range(n)], 2)
    return latin_square({cell: rng.randrange(n) for cell in cells})


rng = random.Random(0)
problems = [random_problem(rng) for _ in range(200)]
# One problem is one training example; the test problems are never seen during training.
data = learning.Dataset.split(problems, (0.8, 0.1, 0.1), seed=0)

with tempfile.TemporaryDirectory() as directory:
    # A recipe is a name; settings are overridden by keyword.
    checkpoint = learning.fit(
        data,
        "consformer-mdd",
        checkpoint_dir=directory,
        seed=0,
        epochs=20,
        validation_interval=5,
        embedding_size=32,
        hidden_size=32,
        expand_size=32,
    )

    # Neural local search repairs an assignment with the network; many problems run as one batch.
    method = ac.NLS.preset("consformer-mdd", checkpoint)
    results = ac.solve(data.test, method, ac.Limits(time=10, iterations=500), seed=0)

solved = [r for r in results if r.found]
print(f"solved {len(solved)} of {len(results)} unseen problems")
for model, result in zip(data.test, results):
    if result.found:
        assert model.check(result.assignment)
