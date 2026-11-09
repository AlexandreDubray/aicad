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


SETUP_RELAXATIONS = ("last_job", "last_set", "arc_times")


@pytest.mark.parametrize("relaxation", SETUP_RELAXATIONS)
def test_disjunctive_with_setup_times_count_matches_enumeration(relaxation):
    setup = [[0, 2, 1, 0], [1, 0, 3, 2], [0, 1, 0, 2], [2, 0, 1, 0]]
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(
        ac.disjunctive(
            x,
            [0, 1, 0, 3],
            [14, 14, 14, 16],
            [3, 2, 4, 2],
            precedences=[(2, 0)],
            setup_times=setup,
            setup_relaxation=relaxation,
        )
    )
    with_setup = solutions(m)
    plain = ac.Model()
    y = plain.int_vars(4, domain=range(4))
    plain.add(ac.disjunctive(y, [0, 1, 0, 3], [14, 14, 14, 16], [3, 2, 4, 2], precedences=[(2, 0)]))
    assert 0 < len(with_setup) < len(solutions(plain))
    assert ac.Compiler(m).compile().count() == len(with_setup)
    for width in (1, 2):
        assert ac.Compiler(m).compile(width=width).count() >= len(with_setup)


def test_the_relaxations_of_the_setup_times_agree_on_the_solutions():
    setup = [[0, 2, 1, 0], [1, 0, 3, 2], [0, 1, 0, 2], [2, 0, 1, 0]]
    counts = set()
    for relaxation in SETUP_RELAXATIONS:
        m = ac.Model()
        x = m.int_vars(4, domain=range(4))
        m.add(
            ac.disjunctive(
                x,
                [0, 1, 0, 3],
                [14, 14, 14, 16],
                [3, 2, 4, 2],
                setup_times=setup,
                setup_relaxation=relaxation,
            )
        )
        counts.add(ac.Compiler(m).compile().count())
    assert len(counts) == 1


@pytest.mark.parametrize("relaxation", SETUP_RELAXATIONS)
def test_padded_disjunctive_count_matches_enumeration(relaxation):
    # Three jobs on two slots and on a fourth one: the dummy is 3, the jobs do not all appear.
    setup = [[0, 2, 1], [1, 0, 3], [0, 1, 0]]
    for slots in (2, 4):
        m = ac.Model()
        x = m.int_vars(slots, domain=range(4))
        m.add(
            ac.disjunctive(
                x,
                [0, 1, 0],
                [8, 7, 9],
                [3, 2, 4],
                precedences=[(2, 1)],
                setup_times=setup,
                setup_relaxation=relaxation,
                padding=True,
            )
        )
        expected = solutions(m)
        assert 0 < len(expected) < 4**slots
        assert ac.Compiler(m).compile().count() == len(expected)
        for width in (1, 2):
            assert ac.Compiler(m).compile(width=width).count() >= len(expected)


def test_jobs_split_between_two_padded_resources():
    # Every job runs on exactly one of the two machines, which are two padded sequences.
    m = ac.Model()
    machines = [m.int_vars(3, domain=range(4)) for _ in range(2)]
    release, deadline, duration = [0, 0, 1], [4, 5, 7], [2, 2, 3]
    for machine in machines:
        m.add(ac.disjunctive(machine, release, deadline, duration, padding=True))
    m.add(ac.gcc(machines[0] + machines[1], {0: (1, 1), 1: (1, 1), 2: (1, 1)}))
    expected = solutions(m)
    assert expected
    assert ac.Compiler(m).compile().count() == len(expected)
    # No machine has a job twice, and a job is on one machine only.
    for a in expected:
        assert sorted(v for v in a if v != 3) == [0, 1, 2]


def test_a_matrix_of_zeros_is_no_setup():
    m = ac.Model()
    x = m.int_vars(3, domain=range(3))
    m.add(ac.disjunctive(x, [0] * 3, [6] * 3, [2] * 3, setup_times=[[0] * 3] * 3))
    assert ac.Compiler(m).compile().count() == 6


@pytest.mark.parametrize("build", MODELS)
def test_exact_count_matches_enumeration(build):
    m = build()
    mdd = ac.Compiler(m).compile()
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
    assert ac.Compiler(m).compile().count() == 21


def test_table_count_matches_enumeration():
    m = ac.Model()
    x = m.int_vars(4, domain=range(3))
    m.add(ac.table([x[2], x[0], x[3]], [(0, 1, 2), (1, 1, 0), (2, 0, 2), (2, 0, 1), (0, 1, 2)]))
    m.add(ac.not_equal(x[1], x[0]))
    expected = solutions(m)
    assert len(expected) > 0
    assert ac.Compiler(m).compile().count() == len(expected)
    for width in (1, 2):
        assert ac.Compiler(m).compile(width=width).count() >= len(expected)


def test_disjunctive_count_matches_enumeration():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.disjunctive(x, [0, 1, 0, 3], [9, 6, 12, 14], [3, 2, 4, 2]))
    expected = solutions(m)
    assert 0 < len(expected) < 24
    assert ac.Compiler(m).compile().count() == len(expected)
    for width in (1, 2):
        assert ac.Compiler(m).compile(width=width).count() >= len(expected)


def test_disjunctive_with_precedences_count_matches_enumeration():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(
        ac.disjunctive(
            x, [0, 1, 0, 3], [12, 12, 12, 14], [3, 2, 4, 2], precedences=[(2, 0), (1, 3), (1, 3)]
        )
    )
    with_precedences = solutions(m)
    plain = ac.Model()
    y = plain.int_vars(4, domain=range(4))
    plain.add(ac.disjunctive(y, [0, 1, 0, 3], [12, 12, 12, 14], [3, 2, 4, 2]))
    assert 0 < len(with_precedences) < len(solutions(plain))
    assert ac.Compiler(m).compile().count() == len(with_precedences)


@pytest.mark.parametrize("build", MODELS)
def test_relaxed_diagram_is_a_superset(build):
    m = build()
    exact = ac.Compiler(m).compile().count()
    relaxed = ac.Compiler(m).compile(width=2)
    assert relaxed.count() >= exact
    assert max(relaxed.widths) <= 2
    assert relaxed.width == 2


def test_unsat():
    m = ac.Model()
    x = m.int_vars(3, domain=range(2))
    m.add(ac.all_different(x))
    mdd = ac.Compiler(m).compile()
    assert mdd.is_unsat
    assert mdd.count() == 0
    assert mdd.solution() is None
    assert mdd.sample(3) == []


def test_structure_is_consistent_with_count():
    m = latin_3x3()
    mdd = ac.Compiler(m).compile()
    widths = mdd.widths
    assert len(widths) == mdd.num_layers and widths[0] == 1 and widths[-1] == 1
    assert mdd.count() == 12
    assert mdd.to_dot().lstrip().startswith(("digraph", "strict digraph"))


def test_subset_of_constraints_and_custom_order():
    m = ac.Model()
    x = m.int_vars(4, domain=range(2))
    m.add(ac.not_equal(x[0], x[1]), ac.not_equal(x[2], x[3]))
    first_only = ac.Compiler(m, constraints=[0]).compile()
    assert first_only.num_layers == 3  # two variables and the sink
    assert first_only.count() == 2
    ordered = ac.Compiler(m, order=[x[3], x[2], x[1], x[0]]).compile()
    assert ordered.count() == 4


def test_compile_argument_errors():
    m = all_different_4()
    with pytest.raises(ValueError):
        ac.Compiler(m).compile(width=0)
    with pytest.raises(ValueError):
        ac.Compiler(m, merge="nope")
    with pytest.raises(ValueError):
        ac.Compiler(m, order="nope")
    with pytest.raises(ac.ModelError):
        ac.Compiler(m, constraints=[3])
    with pytest.raises(ac.ModelError):
        ac.Compiler(m, order=[0, 0, 1, 2])


@pytest.mark.parametrize("merge", ["less_relaxed", "most_likely", "state_similarity"])
def test_merge_heuristics_run(merge):
    m = latin_3x3()
    assert ac.Compiler(m, merge=merge).compile(width=2).count() >= 12


def two_of_four_constrained():
    m = ac.Model()
    x = m.int_vars(4, domain=range(3))
    m.add(ac.not_equal(x[3], x[1]))
    return m, x


def test_a_diagram_of_some_variables_gives_values_for_its_scope():
    m, x = two_of_four_constrained()
    mdd = ac.Compiler(m).compile()
    assert mdd.scope == [x[1], x[3]]
    assert mdd.count() == 6
    for values in [mdd.solution()] + mdd.sample(10):
        # One value per variable of the model; the two outside every constraint take the first
        # value of their domain.
        assert len(values) == 4 and values[1] != values[3]
        assert m.check(values)


def test_a_diagram_of_the_whole_model_has_every_variable_in_its_scope():
    m = all_different_4()
    assert ac.Compiler(m).compile().scope == m.variables


def test_a_relaxed_diagram_may_find_no_solution_without_being_unsatisfiable():
    m = ac.Model()
    grid = m.int_vars((4, 4), domain=range(4))
    m.add(ac.all_different(row) for row in grid)
    m.add(ac.all_different([grid[i][j] for i in range(4)]) for j in range(4))
    mdd = ac.Compiler(m).compile(width=2)
    assert not mdd.is_unsat
    found = mdd.solution()
    assert found is None or m.check(found)
    assert m.check(ac.Compiler(m).compile().solution())


def test_refine_tightens_a_relaxed_diagram_step_by_step_to_the_exact_one():
    model = ac.Model()
    grid = model.int_vars((4, 4), domain=range(4))
    model.add(ac.all_different(row) for row in grid)
    model.add(ac.all_different([grid[i][j] for i in range(4)]) for j in range(4))
    exact = ac.Compiler(model).compile().count()
    mdd = ac.Compiler(model).compile(width=2)
    counts = [mdd.count()]
    for width in (4, 8, 32, None):
        mdd.refine(width)
        assert mdd.width == width
        counts.append(mdd.count())
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == exact == 576


def test_refine_refuses_a_width_below_one():
    model = ac.Model()
    model.add(ac.all_different(model.int_vars(2, domain=range(2))))
    with pytest.raises(ValueError, match="at least 1"):
        ac.Compiler(model).compile().refine(0)


def test_a_compiler_hands_out_independent_diagrams_of_the_model_it_was_made_for():
    model = ac.Model()
    x = model.int_vars(3, domain=range(3))
    model.add(ac.all_different(x))
    compiler = ac.Compiler(model)
    exact, relaxed = compiler.compile(), compiler.compile(width=1)
    relaxed.refine()
    assert exact.count() == relaxed.count() == 6
    # Later changes to the model do not reach the compiler.
    model.add(ac.equal(x[0], 0))
    assert compiler.compile().count() == 6 and compiler.model.num_constraints == 1
    assert ac.Compiler(model).compile().count() == 2


def test_a_frozen_diagram_answers_but_cannot_be_refined():
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.all_different(x))
    mdd = ac.Compiler(m).compile(width=2)
    mdd.refine()
    assert not mdd.is_frozen
    before = (mdd.count(), mdd.widths, mdd.scope)
    assert mdd.freeze() is mdd
    assert mdd.is_frozen
    assert (mdd.count(), mdd.widths, mdd.scope) == before
    assert mdd.freeze().is_frozen  # idempotent
    assert m.check(mdd.sample(1)[0])
    with pytest.raises(RuntimeError, match="frozen"):
        mdd.refine(8)


def test_an_iteration_limit_gives_a_looser_but_sound_diagram():
    model = ac.Model()
    model.add(ac.all_different(model.int_vars(7, domain=range(7))))
    compiler = ac.Compiler(model)
    full = compiler.compile(width=6).count()
    limited = compiler.compile(width=6, max_iterations=1).count()
    # Every one of the 7! solutions stays a path; the later passes remove the other paths.
    assert limited > full >= 5040
    # A limit that is never reached changes nothing.
    assert compiler.compile(width=6, max_iterations=1000).count() == full


def test_the_iteration_limit_is_chosen_at_each_call():
    model = ac.Model()
    model.add(ac.all_different(model.int_vars(7, domain=range(7))))
    mdd = ac.Compiler(model).compile(width=6, max_iterations=1)
    once = mdd.count()
    mdd.refine(6)  # no limit this time
    assert mdd.count() < once


@pytest.mark.parametrize("bad", [0, -1])
def test_an_iteration_limit_below_one_is_refused(bad):
    model = ac.Model()
    model.add(ac.all_different(model.int_vars(2, domain=range(2))))
    with pytest.raises(ValueError, match="at least 1"):
        ac.Compiler(model).compile(max_iterations=bad)
    with pytest.raises(ValueError, match="at least 1"):
        ac.Compiler(model).compile().refine(max_iterations=bad)
