# Contributing to aicad

Thank you for helping. The full guide is in the documentation under *Contributing* (`docs/contributing/`):
it explains the layering of the crate, the features, and how to add a constraint, a destroy operator,
a local-search architecture or a neural network, including from your own crate.

Before opening a pull request, run the checks listed in `docs/contributing/checklist.md`: clippy, the Rust
tests and docs with and without the `learning` feature, `scripts/check_layering.py`, and the Python tests
in the slim and full builds.
