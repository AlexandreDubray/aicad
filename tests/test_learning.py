import pytest
from conftest import needs_learning

import pyaicad as ac
from pyaicad import CheckpointError, learning
from pyaicad.learning import SGD, Adam, AdamW, Dataset, Recipe


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
    assert (len(a.train), len(a.validation)) == (8, 2)
    assert {id(m) for m in a} == {id(m) for m in ps}
    assert not {id(m) for m in a.train} & {id(m) for m in a.validation}
    more = Dataset.split(ps, 0.3, seed=0)
    assert (len(more.train), len(more.validation)) == (7, 3)


def test_split_keeps_both_sets_non_empty_and_refuses_nonsense():
    two = Dataset.split(problems(2), 0.01)
    assert (len(two.train), len(two.validation)) == (1, 1)
    nearly_all = Dataset.split(problems(5), 0.99)
    assert (len(nearly_all.train), len(nearly_all.validation)) == (1, 4)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between"):
            Dataset.split(problems(4), bad)
    with pytest.raises(ValueError, match="at least two"):
        Dataset.split(problems(1))


def test_a_dataset_has_no_test_set():
    assert not hasattr(Dataset.split(problems(4)), "test")


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


def test_the_default_optimiser_is_adam_and_its_settings_can_be_replaced():
    r = Recipe.named("consformer")
    assert r.training.optimiser == Adam()
    s = r.replace(lr=1e-3, beta_1=0.8)
    assert s.training.optimiser == Adam(lr=1e-3, beta_1=0.8)
    assert r.training.optimiser.lr == 3e-4  # the original is untouched


def test_replacing_the_optimiser_changes_which_settings_exist():
    r = Recipe.named("consformer")
    with pytest.raises(TypeError):
        r.replace(momentum=0.9)  # Adam has no momentum
    s = r.replace(optimiser=SGD(), momentum=0.9, lr=0.1)
    assert s.training.optimiser == SGD(lr=0.1, momentum=0.9)
    with pytest.raises(TypeError):
        s.replace(beta_1=0.8)
    with pytest.raises(TypeError):
        r.replace(optimiser="adam")


def test_optimiser_settings_are_checked():
    with pytest.raises(ValueError):
        Adam(lr=0)
    with pytest.raises(ValueError, match="not both"):
        Adam(grad_clip_norm=1.0, grad_clip_value=1.0)
    with pytest.raises(ValueError):
        Adam(beta_1=1.0)
    with pytest.raises(ValueError, match="need a momentum"):
        SGD(nesterov=True)
    with pytest.raises(ValueError):
        AdamW(weight_decay=-1.0)


@pytest.mark.parametrize(
    "optimiser",
    [
        Adam(lr=1e-2, weight_decay=1e-3, grad_clip_norm=1.0),
        AdamW(lr=1e-2, grad_clip_value=1.0),
        SGD(lr=1e-2, momentum=0.9, nesterov=True, grad_clip_norm=5.0),
    ],
)
@needs_learning
def test_every_optimiser_trains(tmp_path, optimiser):
    ckpt = learning.fit(
        problems(6),
        "consformer",
        checkpoint_dir=tmp_path / type(optimiser).__name__,
        seed=0,
        optimiser=optimiser,
        epochs=2,
        validation_interval=1,
        batch_size=2,
        embedding_size=8,
        hidden_size=8,
        expand_size=8,
    )
    assert ckpt.domain_size == 3


def test_destroy_is_a_training_setting():
    r = Recipe.named("consformer")
    assert r.training.destroy == "random"
    assert r.replace(destroy="worst").training.destroy == "worst"
    with pytest.raises(ValueError):
        r.replace(destroy="nope")


def test_load_checks_the_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        learning.load(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.mpk").write_bytes(b"")
    with pytest.raises(CheckpointError):
        learning.load(tmp_path)  # no domain_size


@needs_learning
@pytest.mark.parametrize("recipe", ["consformer", "consformer-mdd"])
def test_fit_then_solve_with_the_checkpoint(tmp_path, recipe):
    data = Dataset.split(problems(8), 0.25, seed=0)
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
    for method in (ac.NeuralPriorSearch(ckpt), ac.NeuralPriorBPSearch(ckpt.path)):
        results = ac.local_search(test, method, ac.Limits(iterations=30), seed=0)
        assert len(results) == 3
        for m, r in zip(test, results):
            if r.found:
                assert m.check(r.assignment)


@needs_learning
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


@needs_learning
def test_fit_rejects_unknown_settings(tmp_path):
    with pytest.raises(TypeError):
        learning.fit(problems(4), checkpoint_dir=tmp_path, epoch=1)


@needs_learning
def test_fit_refuses_a_run_that_would_never_validate(tmp_path):
    with pytest.raises(ValueError, match="never be validated"):
        learning.fit(problems(4), checkpoint_dir=tmp_path, epochs=2, validation_interval=10)


@needs_learning
def test_training_can_start_from_the_weights_of_an_earlier_checkpoint(tmp_path):
    settings = dict(
        epochs=1,
        validation_interval=1,
        batch_size=2,
        embedding_size=8,
        hidden_size=8,
        expand_size=8,
    )
    first = tmp_path / "first"
    learning.fit(problems(6), "consformer", checkpoint_dir=first, seed=0, **settings)
    resumed = learning.fit(
        problems(6),
        "consformer",
        checkpoint_dir=tmp_path / "resumed",
        seed=0,
        initial_weights=first,
        **settings,
    )
    assert (resumed.path / "weights.mpk").is_file()


@needs_learning
def test_missing_initial_weights_are_an_error(tmp_path):
    with pytest.raises(ValueError, match="nowhere"):
        learning.fit(
            problems(6),
            "consformer",
            checkpoint_dir=tmp_path / "out",
            initial_weights=tmp_path / "nowhere",
            epochs=1,
            validation_interval=1,
            embedding_size=8,
            hidden_size=8,
            expand_size=8,
        )


@pytest.mark.skipif(ac.HAS_LEARNING, reason="checks the build without the learning feature")
def test_a_build_without_learning_refuses_training_and_neural_searches_clearly(tmp_path):
    with pytest.raises(ac.FeatureUnavailableError, match="learning"):
        learning.fit(problems(4), checkpoint_dir=tmp_path, epochs=1, validation_interval=1)
    with pytest.raises(ac.FeatureUnavailableError, match="NeuralPriorSearch"):
        ac.NeuralPriorSearch(tmp_path)
    with pytest.raises(ac.FeatureUnavailableError, match="NeuralPriorBPSearch"):
        ac.NeuralPriorBPSearch(tmp_path)
