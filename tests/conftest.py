"""Shared test setup.

A build of pyaicad without the `learning` feature has no training and no neural searches. The
tests that need them are marked `needs_learning` and skipped there, with the reason shown by
`pytest -rs`.
"""

import pytest

import pyaicad as ac

needs_learning = pytest.mark.skipif(
    not ac.HAS_LEARNING,
    reason="this build of pyaicad has no `learning` feature (build with --features learning)",
)


def pytest_report_header(config):
    return f"pyaicad learning feature: {'yes' if ac.HAS_LEARNING else 'NO (training tests skipped)'}"
