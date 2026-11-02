"""Compiling a model into an exact decision diagram: the compiler, its heuristics, the queries.

An exact diagram has one path per solution of the model. Counting, sampling and detecting
unsatisfiability are then read off the diagram rather than searched for. The heuristics given to
the compiler decide how large the diagram gets, never what it contains.

Run with ``python examples/02_compile_exact.py``.
"""

import pyaicad as ac

# A 4 x 4 latin square: every row and every column holds each of 0..3 once.
n = 4
model = ac.Model()
grid = model.int_vars((n, n), domain=range(n))
model.add(ac.all_different(row) for row in grid)
model.add(ac.all_different([grid[i][j] for i in range(n)]) for j in range(n))

# The compiler is made for a model and the heuristics that drive the compilation. Without a
# width the diagram it compiles is exact.
#   order: the variable of each layer, from the root to the sink.
#   merge: which nodes are joined when a layer is full. Only relaxed diagrams (example 03) merge.
# These are the defaults, written out. "auto" is the min-dom-max-linked heuristic: it starts with
# the variable of smallest domain and then takes the one most linked to those already placed.
compiler = ac.Compiler(model, order="auto", merge="state_similarity")
mdd = compiler.compile()
print(mdd)
print("solutions:", mdd.count())  # 576 latin squares of order 4
print("nodes per layer, auto order:", mdd.widths)

# Any other order is a list of variables. It changes the size of the diagram, not the solutions
# it holds: here the grid is filled row by row.
row_by_row = [grid[i][j] for i in range(n) for j in range(n)]
by_rows = ac.Compiler(model, order=row_by_row).compile()
print("nodes per layer, row by row:", by_rows.widths)
assert by_rows.count() == mdd.count()

# A compiler works on the model as it was when it was made: after adding a clue, make a new one.
model.add(ac.equal(grid[0][0], 2))
mdd = ac.Compiler(model).compile()
print("with grid[0][0] = 2:", mdd.count())  # 576 / 4

# A path of the diagram is an assignment of the model, one value per variable. In an exact
# diagram every solution is equally likely to be drawn.
ac.set_seed(0)
for values in mdd.sample(3):
    assert model.check(values)
    print(values)
print("a solution:", mdd.solution())

# Contradictory constraints are detected during compilation.
model.add(ac.equal(grid[0][1], 2))  # the same value twice in the first row
impossible = ac.Compiler(model).compile()
print("unsatisfiable:", impossible.is_unsat, "- solutions:", impossible.count(), "-", impossible.solution())

# The diagram can be drawn with Graphviz (``dot -Tpdf``).
small = ac.Model()
x = small.int_vars(3, domain=range(3))
small.add(ac.all_different(x))
dot = ac.Compiler(small).compile().to_dot()
print(f"the DOT description of a 3-variable diagram has {len(dot.splitlines())} lines")
