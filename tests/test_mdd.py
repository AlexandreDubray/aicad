import pytest

import pyaicad as ac
from oracle import solutions


def all_different_4():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.all_different(x))
    return m


def chain_not_equal():
    m = ac.Model()
    x = m.int_vars(5, domain=range(3))
    m.add(ac.not_equal(a, b) for a, b in zip(x, x[1:]))
    return m


def among_():
    m = ac.Model()
    x = m.int_vars(5, domain=range(3))
    m.add(ac.among(x, [0, 1], 1, 3))
    return m


def at_least_():
    m = ac.Model()
    x = m.int_vars(5, domain=range(3))
    m.add(ac.at_least(x, [2], 2))
    return m


def sum_():
    m = ac.Model()
    x = m.int_vars(4, domain=[-1, 0, 1, 2])
    m.add(ac.sum_equals(x, 2))
    return m


def gcc_():
    m = ac.Model()
    x = m.int_vars(5, domain=range(3))
    m.add(ac.gcc(x, {0: (1, 2), 1: (0, 3)}))
    return m


def equal_and_all_different():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.all_different(x), ac.equal(x[2], 3))
    return m


def overlapping():
    m = ac.Model()
    x = m.int_vars(5, domain=range(3))
    m.add(
        ac.all_different(x[:3]),
        ac.all_different(x[2:]),
        ac.among(x, [0], 1, 2),
    )
    return m


def latin_3x3():
    m = ac.Model()
    g = m.int_vars((3, 3), domain=range(3))
    m.add(ac.all_different(r) for r in g)
    m.add(ac.all_different([g[i][j] for i in range(3)]) for j in range(3))
    return m


MODELS = [
    all_different_4,
    chain_not_equal,
    among_,
    at_least_,
    sum_,
    gcc_,
    equal_and_all_different,
    overlapping,
    latin_3x3,
]


@pytest.mark.parametrize("build", MODELS)
def test_exact_count_matches_enumeration(build):
    m = build()
    mdd = ac.compile(m)
    expected = solutions(m)
    assert mdd.count() == len(expected)
    sol = mdd.solution()
    assert sol is not None and sol in expected
    ac.set_seed(1)
    for s in mdd.sample(20):
        assert s in expected


def test_regular_count_matches_enumeration():
    # no two consecutive ones
    m = ac.Model()
    x = m.int_vars(6, domain=[0, 1])
    m.add(ac.regular(x, [[0, 1], [0, None]], 0, [0, 1]))
    expected = solutions(m, regular_symbols=[0, 1])
    assert len(expected) == 21
    assert ac.compile(m).count() == 21


@pytest.mark.parametrize("build", MODELS)
def test_relaxed_diagram_is_a_superset(build):
    m = build()
    exact = ac.compile(m).count()
    relaxed = ac.compile(m, width=2)
    assert relaxed.count() >= exact
    assert max(relaxed.widths) <= 2
    assert relaxed.width == 2


def test_unsat():
    m = ac.Model()
    x = m.int_vars(3, domain=range(2))
    m.add(ac.all_different(x))
    mdd = ac.compile(m)
    assert mdd.is_unsat
    assert mdd.count() == 0
    assert mdd.solution() is None
    assert mdd.sample(3) == []


def test_structure_is_consistent_with_count():
    m = latin_3x3()
    mdd = ac.compile(m)
    widths = mdd.widths
    assert len(widths) == mdd.num_layers and widths[0] == 1 and widths[-1] == 1
    assert mdd.count() == 12
    assert mdd.to_dot().lstrip().startswith(("digraph", "strict digraph"))


def test_subset_of_constraints_and_custom_order():
    m = ac.Model()
    x = m.int_vars(4, domain=range(2))
    m.add(ac.not_equal(x[0], x[1]), ac.not_equal(x[2], x[3]))
    first_only = ac.compile(m, constraints=[0])
    assert first_only.num_layers == 3  # two variables and the sink
    assert first_only.count() == 2
    ordered = ac.compile(m, order=[x[3], x[2], x[1], x[0]])
    assert ordered.count() == 4


def test_compile_argument_errors():
    m = all_different_4()
    with pytest.raises(ValueError):
        ac.compile(m, width=0)
    with pytest.raises(ValueError):
        ac.compile(m, merge="nope")
    with pytest.raises(ValueError):
        ac.compile(m, order="nope")
    with pytest.raises(ac.ModelError):
        ac.compile(m, constraints=[3])
    with pytest.raises(ac.ModelError):
        ac.compile(m, order=[0, 0, 1, 2])


@pytest.mark.parametrize("merge", ["less_relaxed", "most_likely", "state_similarity"])
def test_merge_heuristics_run(merge):
    m = latin_3x3()
    assert ac.compile(m, width=2, merge=merge).count() >= 12


def two_of_four_constrained():
    m = ac.Model()
    x = m.int_vars(4, domain=range(3))
    m.add(ac.not_equal(x[3], x[1]))
    return m, x


def test_a_diagram_of_some_variables_gives_values_for_its_scope():
    m, x = two_of_four_constrained()
    mdd = ac.compile(m)
    assert mdd.scope == [x[1], x[3]]
    assert mdd.count() == 6
    for values in [mdd.solution()] + mdd.sample(10):
        assert len(values) == 2 and values[0] != values[1]


def test_a_diagram_of_the_whole_model_has_every_variable_in_its_scope():
    m = all_different_4()
    assert ac.compile(m).scope == m.variables


def test_solving_gives_a_value_to_variables_outside_every_constraint():
    m, _ = two_of_four_constrained()
    r = ac.solve(m, ac.Compilation())
    assert r.found and len(r.assignment) == 4
    assert m.check(r.assignment)
