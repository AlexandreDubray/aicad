"""An independent, brute-force reference for the semantics of the constraints."""

import itertools

import pyaicad as ac
from pyaicad import model as _m


def holds(c, a):
    """Whether constraint ``c`` holds for the assignment ``a`` (a list of values)."""
    vals = [a[v.index] for v in c.scope]
    if isinstance(c, _m._AllDifferent):
        return len(set(vals)) == len(vals)
    if isinstance(c, _m._NotEqual):
        return vals[0] != vals[1]
    if isinstance(c, _m._Equal):
        return vals[0] == c.value
    if isinstance(c, _m._Among):
        return c.lb <= sum(v in c.values for v in vals) <= c.ub
    if isinstance(c, _m._AtLeast):
        return sum(v in c.values for v in vals) >= c.lb
    if isinstance(c, _m._SumEquals):
        return sum(vals) == c.target
    if isinstance(c, _m._Gcc):
        return all(lo <= vals.count(value) <= hi for value, lo, hi in c.bounds)
    if isinstance(c, _m._Regular):
        # columns are the sorted values found in the domains of the scope
        raise NotImplementedError("use holds_regular")
    raise AssertionError(c)


def holds_regular(c, a, symbols):
    state = c.initial_state
    for v in c.scope:
        nxt = c.transitions[state][symbols.index(a[v.index])]
        if nxt is None:
            return False
        state = nxt
    return state in c.accepting_states


def solutions(model, regular_symbols=None):
    """All the solutions of ``model``, by enumeration."""
    domains = [model.domain(v) for v in model.variables]
    out = []
    for a in itertools.product(*domains):
        a = list(a)
        ok = True
        for c in model.constraints:
            if isinstance(c, _m._Regular):
                ok = holds_regular(c, a, regular_symbols)
            else:
                ok = holds(c, a)
            if not ok:
                break
        if ok:
            out.append(a)
    return out
