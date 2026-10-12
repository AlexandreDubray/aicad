"""Model a problem, compile it into an MDD, and solve it.

Run with ``python examples/quickstart.py``.
"""

import pyaicad as ac

# A 4 x 4 latin square: every row and every column holds each of 0..3 once.
n = 4
model = ac.Model()
grid = model.int_vars((n, n), domain=range(n))
model.add(ac.all_different(row) for row in grid)
model.add(ac.all_different([grid[i][j] for i in range(n)]) for j in range(n))
model.add(ac.equal(grid[0][0], 2))  # a clue

# Compile: with no width the diagram is exact, so its paths are the solutions.
mdd = ac.compile(model)
print(mdd, "-", mdd.count(), "solutions")
print("a random one:", mdd.sample(1)[0])

# A relaxed diagram has at most `width` nodes per layer; it is smaller, and counts a superset.
relaxed = ac.compile(model, width=4)
print(relaxed, "-", relaxed.count(), "paths, of which only some are solutions")

# Solve: the default method compiles an exact diagram and reads a solution off it.
result = ac.solve(model)
print(result.status.value, result.assignment, f"in {result.runtime:.3f}s")
assert model.check(result.assignment)
