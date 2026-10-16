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
    assert ac.compile(m).count() == 24
    ac.set_num_threads(2)
    assert ac.compile(m).count() == 24
