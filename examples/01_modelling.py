"""Modelling: variables, constraints, and checking an assignment.

A weekly shift roster for a small team, and a solution read off its diagram. Run with
``python examples/01_modelling.py``.
"""

import pyaicad as ac

DAYS, NURSES = 5, 3
OFF, DAY, NIGHT = 0, 1, 2

model = ac.Model()
# roster[n][d] is the shift of nurse n on day d.
roster = model.int_vars((NURSES, DAYS), domain=[OFF, DAY, NIGHT])

# Every day has exactly one nurse on the night shift and one or two on the day shift.
for d in range(DAYS):
    column = [roster[n][d] for n in range(NURSES)]
    model.add(ac.gcc(column, {NIGHT: (1, 1), DAY: (1, 2)}))

# A nurse works at most five days a week.
for row in roster:
    model.add(ac.among(row, [DAY, NIGHT], 0, 5))

# A night shift is never directly followed by a day shift. This is a pattern on consecutive
# days, which an automaton states directly: state 0 is "not after a night", state 1 "after a
# night". The columns of the table are the sorted values of the domains: OFF, DAY, NIGHT.
after_night = [
    [0, 0, 1],  # state 0: OFF -> 0, DAY -> 0, NIGHT -> 1
    [0, None, 1],  # state 1: OFF -> 0, DAY is forbidden, NIGHT -> 1
]
for row in roster:
    model.add(ac.regular(row, after_night, initial_state=0, accepting_states=[0, 1]))

# The first nurse is on leave on the last day.
model.add(ac.equal(roster[0][DAYS - 1], OFF))

print(model)

# A candidate assignment lists one value per variable, in the order of ``model.variables``.
everyone_off = [OFF] * model.num_vars
# We can check if it is a solution by using model.check(assignment)
print("everyone off is a solution:", model.check(everyone_off))
# We can also check each constraint independently
print("Number of satisfied constraints: {}".format(len([c for c in model.constraints if model.is_constraint_satisfied(c, everyone_off)])))

# A solution comes from compiling the model. The compiler is made for the model, with the
# heuristics that drive the compilation; without a width the diagram it hands out is exact, and
# its paths are the solutions. A solution is read off it, unless there is none.
compiler = ac.Compiler(model)
mdd = compiler.compile()
print(mdd, "-", mdd.count(), "rosters")
if mdd.is_unsat:
    raise SystemExit("no roster satisfies the constraints")
solution = mdd.solution()
assert model.check(solution)
for n in range(NURSES):
    print(f"nurse {n}:", " ".join("-DN"[v] for v in solution[n * DAYS : (n + 1) * DAYS]))
