import pickle

import pytest

import pyaicad as ac
from pyaicad import ModelError


def test_variables_and_shapes():
    m = ac.Model()
    a = m.int_var([3, 1, 2])
    flat = m.int_vars(4, domain=range(2))
    grid = m.int_vars((2, 3), domain=[5])
    bools = m.bool_vars((2,))
    assert (a.index, [v.index for v in flat]) == (0, [1, 2, 3, 4])
    assert len(grid) == 2 and len(grid[0]) == 3 and grid[1][2].index == 10
    assert [m.domain(v) for v in bools] == [(0, 1), (0, 1)]
    assert m.domain(a) == (3, 1, 2)  # order is kept
    assert m.num_vars == 13 and m.variables[0] == a


@pytest.mark.parametrize("bad", [[], [1, 1], [0.5], ["a"]])
def test_bad_domains(bad):
    with pytest.raises(ModelError):
        ac.Model().int_var(bad)


def test_constraint_validation():
    m = ac.Model()
    x = m.int_vars(3, domain=range(3))
    other = ac.Model().int_var(range(3))
    with pytest.raises(ModelError):
        ac.all_different([x[0], x[0]])
    with pytest.raises(ModelError):
        ac.all_different([x[0], 7])
    with pytest.raises(ModelError):
        ac.among(x, [1], 2, 1)
    with pytest.raises(ModelError):
        ac.gcc(x, {1: (3, 2)})
    with pytest.raises(ModelError):
        ac.gcc(x, [(1, 0, 1), (1, 0, 2)])
    with pytest.raises(ModelError):
        ac.regular(x, [[1], [None]], 0, [5])
    with pytest.raises(ModelError):
        ac.table(x, [(0, 1, 2), (0, 1)])  # a tuple of the wrong length
    with pytest.raises(ModelError):
        ac.table([x[0], x[0]], [(0, 0)])  # a repeated variable
    with pytest.raises(ModelError):
        ac.table(x, [(0, 1, "a")])  # not an integer
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0, 0], [5, 5, 5], [1, 1, 1])  # a list of the wrong length
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0, 0, 0], [5, 5, 5], [1, -1, 1])  # a negative duration
    with pytest.raises(ModelError):
        ac.disjunctive([x[0], x[0]], [0, 0], [5, 5], [1, 1])  # a repeated variable
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, precedences=[(0, 3)])  # not a job
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, precedences=[(1, 1)])  # itself
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, precedences=[(0, 1), (1, 2), (2, 0)])  # cycle
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, setup_times=[[0] * 3] * 2)  # too few rows
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, setup_times=[[0] * 3, [0] * 3, [0] * 2])  # ragged
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, setup_times=[[0, 1, 0], [0, 0, -1], [0] * 3])  # negative
    with pytest.raises(ModelError):
        ac.disjunctive(x, [0] * 3, [5] * 3, [1] * 3, setup_relaxation="everything")  # unknown
    with pytest.raises(ModelError):
        m.add(ac.not_equal(x[0], other))  # variable from another model
    with pytest.raises(ModelError):
        m.add(ac.equal(x[0], 9))  # not in the domain
    with pytest.raises(ModelError):
        m.add("nope")
    assert m.num_constraints == 0  # nothing was added by the failed calls


def test_add_accepts_constraints_and_iterables():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.all_different(x), [ac.not_equal(x[0], x[1]), ac.equal(x[3], 2)])
    assert m.num_constraints == 3
    assert m.domain(x[3]) == (2,)


def test_check_and_constraint_satisfaction():
    m = ac.Model()
    x = m.int_vars(3, domain=range(3))
    m.add(ac.all_different(x[:2]), ac.not_equal(x[1], x[2]))
    assert m.check([0, 1, 2])
    assert not m.check([0, 1, 1])
    assert not m.check([0, 1, 7])  # outside the domain
    first, second = m.constraints
    assert [m.is_constraint_satisfied(c, [0, 1, 1]) for c in m.constraints] == [True, False]
    assert m.is_constraint_satisfied(first, [0, 0, 2]) is False
    assert m.is_constraint_satisfied(second, [0, 0, 2]) is True
    with pytest.raises(ModelError):
        m.is_constraint_satisfied(ac.not_equal(x[0], x[1]), [0, 1, 2])
    with pytest.raises(ModelError):
        m.check([0, 1])


def test_model_can_grow_after_use():
    m = ac.Model()
    x = m.int_vars(2, domain=range(2))
    assert m.check([0, 0])
    m.add(ac.not_equal(*x))  # the native problem is rebuilt
    assert not m.check([0, 0])
    y = m.int_var(range(2))
    assert m.check([0, 1, 1]) and y.index == 2


def test_copy_is_independent():
    m = ac.Model()
    x = m.int_vars(2, domain=range(2))
    c = m.copy()
    c.add(ac.not_equal(*x))  # the original's variables work in the copy
    assert m.num_constraints == 0 and c.num_constraints == 1
    assert m.check([0, 0]) and not c.check([0, 0])


def test_pickle_roundtrip():
    m = ac.Model()
    x = m.int_vars(3, domain=range(3))
    m.add(ac.all_different(x), ac.equal(x[0], 1))
    m.check([1, 0, 2])  # builds the native problem, which must not be pickled
    n = pickle.loads(pickle.dumps(m))
    assert n.num_vars == 3 and n.num_constraints == 2
    assert n.check([1, 0, 2]) and not n.check([0, 1, 2])
    assert n.domain(x[0]) == (1,)


def test_equal_does_not_change_other_models_domains():
    m = ac.Model()
    x = m.int_var(range(3))
    m.add(ac.equal(x, 2))
    assert m.check([2]) and not m.check([1])


def test_constraint_satisfaction_with_equal_constraints():
    m = ac.Model()
    x = m.int_vars(3, domain=range(3))
    m.add(ac.equal(x[0], 1), ac.all_different(x), ac.not_equal(x[1], x[2]))
    assert [m.is_constraint_satisfied(c, [1, 0, 0]) for c in m.constraints] == [True, False, False]
    assert [m.is_constraint_satisfied(c, [0, 1, 2]) for c in m.constraints] == [False, True, True]
