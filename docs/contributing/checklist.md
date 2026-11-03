# Before you open a pull request

Run everything that CI runs. The Rust commands are run from the repository root.

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features learning -- -D warnings
cargo clippy --all-targets --features python,learning -- -D warnings
cargo test
cargo test --features learning
python scripts/check_layering.py
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features learning --document-private-items
```

Then the Python package, in both flavours, in a virtual environment:

```
maturin develop --features python               # slim
pytest -rs                                      # the learning tests are skipped
maturin develop --features python,learning      # full
pytest -rs
sphinx-build -W docs docs/_build                # runs every example of the Python guide
```

## What a good change has

- **Tests that fail without it**, in the style of the neighbouring tests. A new extension point has
  a test that uses it the way a downstream user would (see `tests/custom_search.rs` and
  `tests/custom_network.rs`, which are compiled as separate crates).
- **Guards with clear messages** on input that cannot mean anything: a panic (Rust) or an error
  (Python) that names the argument and what was expected, rather than undefined behaviour.
- **Documentation in the same change**: a `//!` header on every new module, doc comments on public
  items (the crate denies missing docs on the layers that have them), and an entry in the Python
  guide or API reference when something is user-visible.
- **No dependency from a layer on a later one**, and nothing that needs burn outside the `learning`
  feature.
- **Both builds working**: the slim wheel must still work, and refuse what needs learning with
  `FeatureUnavailableError`.
