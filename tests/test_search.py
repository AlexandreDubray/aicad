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


def test_limits_validation():
    for bad in ({"time": 0}, {"time": -1.0}, {"iterations": 0}):
        with pytest.raises(ValueError):
            Limits(**bad)


def test_argument_types():
    with pytest.raises(TypeError):
        ac.local_search(latin(), "bp")
    with pytest.raises(TypeError):
        ac.local_search([latin(), 3], ac.BeliefPropagationSearch())


def test_search_option_validation(tmp_path):
    with pytest.raises(ValueError):
        ac.BeliefPropagationSearch(destroy="nope")
    with pytest.raises(ValueError):
        ac.BeliefPropagationSearch(destroy_fraction=0)
    with pytest.raises(ValueError):
        ac.BeliefPropagationSearch(bp_iterations=0)
    with pytest.raises(ValueError):
        ac.BeliefPropagationSearch(batch_size=0)
    with pytest.raises(ValueError):
        ac.NeuralPriorSearch(tmp_path, destroy="nope")
    with pytest.raises(ValueError):
        ac.NeuralPriorBPSearch(tmp_path, bp_iterations=0)
    # A search has the settings of its architecture and no other.
    with pytest.raises(TypeError):
        ac.BeliefPropagationSearch(checkpoint=tmp_path)
    with pytest.raises(TypeError):
        ac.NeuralPriorSearch(tmp_path, bp_iterations=2)
    with pytest.raises(TypeError):
        ac.NeuralPriorBPSearch(tmp_path, temperature=2.0)
    assert ac.NeuralPriorBPSearch(tmp_path).destroy == "related"
    assert ac.BeliefPropagationSearch().with_(bp_iterations=3).bp_iterations == 3


def test_a_consformer_search_needs_a_checkpoint(tmp_path):
    with pytest.raises(TypeError):
        ac.NeuralPriorSearch()
    with pytest.raises(CheckpointError):
        ac.local_search(latin(), ac.NeuralPriorSearch(tmp_path))  # not a checkpoint
    with pytest.raises(CheckpointError):
        ac.local_search(latin(), ac.NeuralPriorBPSearch(tmp_path))


def test_belief_propagation_search_solves():
    models = [latin(4) for _ in range(3)]
    search = ac.BeliefPropagationSearch(
        destroy="related",
        destroy_fraction=0.5,
        stochastic=True,
        bp_iterations=3,
    )
    results = ac.local_search(models, search, Limits(iterations=500), seed=0)
    assert len(results) == 3
    for m, r in zip(models, results):
        assert r.search == "bp" and r.iterations is not None and r.iterations >= 0
        if r.found:
            assert m.check(r.assignment)
    assert any(r.found for r in results)
    assert all(r.compile_time <= r.runtime for r in results)


def test_belief_propagation_search_batches_problems_of_different_sizes():
    models = [latin(3), latin(4), impossible()]
    results = ac.local_search(models, ac.BeliefPropagationSearch(), Limits(iterations=500), seed=0)
    assert len(results) == 3
    assert ac.local_search([], ac.BeliefPropagationSearch()) == []
    assert results[2].status is Status.UNKNOWN  # a search never proves infeasibility
    for m, r in zip(models, results):
        if r.found:
            assert m.check(r.assignment)


def test_belief_propagation_search_single_model_gives_a_single_result():
    r = ac.local_search(latin(4), ac.BeliefPropagationSearch(), Limits(iterations=20), seed=1)
    assert isinstance(r, ac.Result)
