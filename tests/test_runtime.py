import os
import subprocess
import sys

import numpy as np
import pytest

import pyaicad as ac


@pytest.fixture
def restore_threads():
    before = ac.get_num_threads()
    yield
    ac.set_num_threads(before)


def test_the_default_is_at_least_one_and_at_most_the_cores():
    out = subprocess.run(
        [sys.executable, "-c", "import pyaicad; print(pyaicad.get_num_threads())"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert 1 <= int(out.stdout) <= (os.cpu_count() or 1)


def test_set_and_get(restore_threads):
    ac.set_num_threads(3)
    assert ac.get_num_threads() == 3
    ac.set_num_threads(np.int64(2))
    assert ac.get_num_threads() == 2


@pytest.mark.parametrize("n", [0, -1])
def test_fewer_than_one_thread_is_refused(restore_threads, n):
    before = ac.get_num_threads()
    with pytest.raises(ValueError, match="at least 1"):
        ac.set_num_threads(n)
    assert ac.get_num_threads() == before


@pytest.mark.parametrize("n", [2.5, "2", None, True])
def test_a_non_integer_is_refused(restore_threads, n):
    with pytest.raises(TypeError, match="integer"):
        ac.set_num_threads(n)


def test_it_can_be_changed_between_uses(restore_threads):
    m = ac.Model()
    x = m.int_vars(4, domain=range(4))
    m.add(ac.all_different(x))
    ac.set_num_threads(1)
    assert ac.Compiler(m).compile().count() == 24
    ac.set_num_threads(2)
    assert ac.Compiler(m).compile().count() == 24


def test_version_reports_the_running_build():
    info = ac.version()
    assert info.version == ac.__version__
    assert info.commit is None or len(info.commit) == 40
    assert info.rustc.startswith("rustc ")
    assert info.profile in ("debug", "release")
    assert ("learning" in info.features) == ac.HAS_LEARNING
    assert "python" in info.features
    assert info.version in str(info)


def test_a_data_log_starts_with_the_build_info(tmp_path):
    import json

    import pyaicad.diagnostics as diag

    path = tmp_path / "log.jsonl"
    diag.enable_data_log(path)
    diag.disable_data_log()
    first = json.loads(path.read_text().splitlines()[0])
    assert first["event"] == "build_info"
    assert first["version"] == ac.version().version
    assert first["commit"] == ac.version().commit
    assert tuple(first["features"]) == ac.version().features
