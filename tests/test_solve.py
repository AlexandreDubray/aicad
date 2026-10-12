import pytest

import pyaicad as ac
from pyaicad import CheckpointError, Limits, Status


def latin(n=3):
    m = ac.Model()
    g = m.int_vars((n, n), domain=range(n))
    m.add(ac.all_different(r) for r in g)
    m.add(ac.all_different([g[i][j] for i in range(n)]) for j in range(n))
    return m


def impossible():
    m = ac.Model()
    x = m.int_vars(3, domain=range(2))
    m.add(ac.all_different(x))
    return m


def test_default_is_exact_compilation():
    m = latin()
    r = ac.solve(m)
    assert r.status is Status.FEASIBLE and r.found
    assert m.check(r.assignment)
    assert r.method == "compilation" and r.iterations is None
    assert 0 <= r.compile_time <= r.runtime


def test_infeasible_is_proved():
    r = ac.solve(impossible())
    assert r.status is Status.INFEASIBLE and r.assignment is None and not r.found


def test_relaxed_never_returns_a_wrong_solution():
    m = latin(4)
    r = ac.solve(m, ac.Compilation(width=1))
    assert r.status in (Status.FEASIBLE, Status.UNKNOWN)
    if r.found:
        assert m.check(r.assignment)
    else:
        assert r.assignment is None


def test_relaxed_diagram_can_prove_infeasibility():
    assert ac.solve(impossible(), ac.Compilation(width=1)).status is Status.INFEASIBLE


def test_many_problems_keep_their_order():
    models = [latin(), impossible(), latin(4)]
    results = ac.solve(models)
    assert [r.status for r in results] == [Status.FEASIBLE, Status.INFEASIBLE, Status.FEASIBLE]
    assert len(results[2].assignment) == 16
    assert ac.solve([]) == []


def test_compilation_refuses_limits():
    with pytest.raises(ValueError):
        ac.solve(latin(), ac.Compilation(), Limits(time=1))


def test_limits_validation():
    for bad in ({"time": 0}, {"time": -1.0}, {"iterations": 0}):
        with pytest.raises(ValueError):
            Limits(**bad)


def test_argument_types():
    with pytest.raises(TypeError):
        ac.solve(latin(), method="nls")
    with pytest.raises(TypeError):
        ac.solve([latin(), 3])


def test_nls_option_validation():
    with pytest.raises(ValueError):
        ac.NLS(destroy="nope")
    with pytest.raises(ValueError):
        ac.NLS(prior="uniform")  # needs decode="mdd"
    with pytest.raises(ValueError):
        ac.NLS(destroy_fraction=0)
    with pytest.raises(ValueError):
        ac.NLS.preset("nope")
    assert ac.NLS.preset("consformer-mdd", decode="logits").decode == "logits"


def test_nls_needs_a_checkpoint_with_a_network_prior(tmp_path):
    with pytest.raises(CheckpointError):
        ac.solve(latin(), ac.NLS())
    with pytest.raises(CheckpointError):
        ac.solve(latin(), ac.NLS(tmp_path))  # not a checkpoint


def test_nls_with_uniform_prior_and_mdd_decoding_solves():
    models = [latin(4) for _ in range(3)]
    method = ac.NLS(
        prior="uniform",
        decode="mdd",
        destroy="related",
        destroy_fraction=0.5,
        stochastic=True,
        bp_iterations=3,
    )
    results = ac.solve(models, method, Limits(iterations=500), seed=0)
    again = ac.solve(models, method, Limits(iterations=500), seed=0)
    assert [r.iterations for r in results] == [r.iterations for r in again]
    assert len(results) == 3
    for m, r in zip(models, results):
        assert r.method == "nls" and r.iterations is not None and r.iterations >= 0
        if r.found:
            assert m.check(r.assignment)
    assert any(r.found for r in results)
    assert all(r.compile_time <= r.runtime for r in results)


def test_nls_single_model_gives_a_single_result():
    r = ac.solve(latin(4), ac.NLS(prior="uniform", decode="mdd"), Limits(iterations=20), seed=1)
    assert isinstance(r, ac.Result)
