import pytest

import pyaicad as ac
from pyaicad import CheckpointError, learning
from pyaicad.learning import Dataset, Recipe


def latin_with_hint(n, hint):
    m = ac.Model()
    g = m.int_vars((n, n), domain=range(n))
    m.add(ac.all_different(r) for r in g)
    m.add(ac.all_different([g[i][j] for i in range(n)]) for j in range(n))
    m.add(ac.equal(g[0][0], hint))
    return m


def problems(count=8, n=3):
    return [latin_with_hint(n, i % n) for i in range(count)]


def test_split_is_reproducible_and_complete():
    ps = problems(10)
    a, b = Dataset.split(ps, seed=3), Dataset.split(ps, seed=3)
    assert [id(m) for m in a.train] == [id(m) for m in b.train]
    assert (len(a.train), len(a.validation), len(a.test)) == (8, 1, 1)
    assert {id(m) for m in a} == {id(m) for m in ps}
    two = Dataset.split(ps, (0.7, 0.3), seed=0)
    assert (len(two.train), len(two.validation), len(two.test)) == (7, 3, 0)


def test_dataset_needs_training_and_validation_problems():
    with pytest.raises(ValueError):
        Dataset(train=[], validation=problems(1))
    with pytest.raises(ValueError):
        Dataset(train=problems(1), validation=[])


def test_recipe_replace_routes_names():
    r = Recipe.named("consformer-mdd")
    s = r.replace(epochs=3, embedding_size=16, merge="most_likely")
    assert (s.training.epochs, s.network.embedding_size, s.merge) == (3, 16, "most_likely")
    assert r.training.epochs == 10  # the original is untouched
    with pytest.raises(TypeError):
        r.replace(epoch=3)
    with pytest.raises(ValueError):
        Recipe.named("nope")
    with pytest.raises(ValueError):
        r.replace(selection="nope")


def test_load_checks_the_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        learning.load(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.mpk").write_bytes(b"")
    with pytest.raises(CheckpointError):
        learning.load(tmp_path)  # no domain_size


@pytest.mark.parametrize("recipe", ["consformer", "consformer-mdd"])
def test_fit_then_solve_with_the_checkpoint(tmp_path, recipe):
    data = Dataset.split(problems(8), (0.75, 0.25), seed=0)
    ckpt = learning.fit(
        data,
        recipe,
        checkpoint_dir=tmp_path / recipe,
        seed=0,
        epochs=2,
        batch_size=2,
        validation_interval=1,
        embedding_size=16,
        hidden_size=16,
        expand_size=16,
    )
    assert ckpt.domain_size == 3  # 'auto': covers the values 0..2
    assert learning.load(ckpt.path).config == ckpt.config
    test = problems(3)
    for method in (ac.NLS(ckpt), ac.NLS.preset(recipe, ckpt.path)):
        results = ac.solve(test, method, ac.Limits(iterations=30), seed=0)
        assert len(results) == 3
        for m, r in zip(test, results):
            if r.found:
                assert m.check(r.assignment)


def test_fit_takes_a_plain_list_and_a_domain_size(tmp_path):
    ckpt = learning.fit(
        problems(6),
        "consformer",
        checkpoint_dir=tmp_path / "plain",
        seed=1,
        epochs=1,
        validation_interval=1,
        batch_size=2,
        domain_size=5,
        embedding_size=8,
        hidden_size=8,
        expand_size=8,
    )
    assert ckpt.domain_size == 5


def test_fit_rejects_unknown_settings(tmp_path):
    with pytest.raises(TypeError):
        learning.fit(problems(4), checkpoint_dir=tmp_path, epoch=1)


def test_fit_refuses_a_run_that_would_never_validate(tmp_path):
    with pytest.raises(ValueError, match="never be validated"):
        learning.fit(problems(4), checkpoint_dir=tmp_path, epochs=2, validation_interval=10)
