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

# Modelling

A problem is a {class}`~pyaicad.Model`: variables with finite integer domains, and constraints
over them. Everything else in the library starts from a model.

## Variables

`int_vars` creates variables. Given a number it returns a list, given a shape it returns nested
lists, so a roster of 3 nurses over 5 days is a 3 by 5 grid.

```{code-cell} python
import pyaicad as ac

DAYS, NURSES = 5, 3
OFF, DAY, NIGHT = 0, 1, 2

model = ac.Model()
roster = model.int_vars((NURSES, DAYS), domain=[OFF, DAY, NIGHT])
model.num_vars
```

`roster[n][d]` is the shift of nurse `n` on day `d`. Variables have no names: you keep the lists
that `int_vars` returned.

## Constraints

Constraints are built by functions and added to the model with `add`, one at a time or as any
iterable. The available ones are {func}`~pyaicad.all_different`, {func}`~pyaicad.not_equal`,
{func}`~pyaicad.equal`, {func}`~pyaicad.among`, {func}`~pyaicad.at_least`,
{func}`~pyaicad.sum_equals`, {func}`~pyaicad.gcc` and {func}`~pyaicad.regular`.

```{code-cell} python
# Every day has exactly one nurse on the night shift and one or two on the day shift.
for d in range(DAYS):
    column = [roster[n][d] for n in range(NURSES)]
    model.add(ac.gcc(column, {NIGHT: (1, 1), DAY: (1, 2)}))

# A nurse works at most five days a week.
model.add(ac.among(row, [DAY, NIGHT], 0, 5) for row in roster)

# The first nurse is on leave on the last day.
model.add(ac.equal(roster[0][DAYS - 1], OFF))
```

`regular` states a pattern on a sequence of variables with an automaton. Here, a night shift is
never directly followed by a day shift. State 0 is "not after a night", state 1 is "after a night";
the columns of the table are the sorted values of the domain (off, day, night), and `None` forbids
a transition.

```{code-cell} python
after_night = [
    [0, 0, 1],     # state 0: off -> 0, day -> 0, night -> 1
    [0, None, 1],  # state 1: off -> 0, day is forbidden, night -> 1
]
model.add(ac.regular(row, after_night, initial_state=0, accepting_states=[0, 1]) for row in roster)

len(model.constraints)
```

## Checking an assignment

An assignment lists one value per variable, in the order of `model.variables`. `model.check` tells
whether it satisfies every constraint, and `is_constraint_satisfied` whether it satisfies one.

```{code-cell} python
everyone_off = [OFF] * model.num_vars
print("everyone off is a solution:", model.check(everyone_off))

violated = [c for c in model.constraints if not model.is_constraint_satisfied(c, everyone_off)]
print(len(violated), "of", len(model.constraints), "constraints are violated")
```

Finding a solution is the job of the next chapters.
