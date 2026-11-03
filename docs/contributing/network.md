# Adding a neural network

A network is plugged in through four traits in `aicad::neural` and `aicad::learning`. Training
(`train_model`) and search (`NeuralPriorSearch`, `NeuralPriorBpSearch`, `solve_with_network`) are
generic over them, and neither depends on the other. Enable the `learning` feature to use them.

| Trait           | What it is                                                                 |
|-----------------|----------------------------------------------------------------------------|
| `Batch<B>`      | The problems prepared for one forward pass: their current assignments, the variables to re-predict, how logits become assignments, and which problems it can hold together. |
| `Network<B, Ba>` | The module: `forward(&batch)` returns logits of shape `(batch, variables, domain)`. |
| `NetworkConfig<B>` | The hyperparameters, and `init(problems, device)` to build a randomly initialised network from them. |
| `Loss<B, Ba>`   | The scalar to minimise, from the logits and the batch.                      |

**What a network accepts is the network's business.** `Batch::check_problems` states the requirement
(for ConsFormer: the same number of variables in every problem) next to the code that relies on it,
and every caller asks the batch type instead of assuming. Document the restrictions of a network in
its own module header: that is where the Python guide sends readers.

## From another crate

`tests/custom_network.rs` is a complete example that is compiled as a separate crate: a toy network
with its own batch, configuration and loss, trained, loaded from a checkpoint and used in a search.
The steps are:

1. **Define the batch** and implement `Batch`. For training, also write a `burn` `Batcher` that
   builds it from samples (a sample can be a plain `Arc<Problem>`), and give `train_model` a dataset
   of samples. At inference time the search builds the batch itself with `Batch::for_assignments`.
2. **Define the module and its configuration** with burn's `Module` and `Config` derives, and
   implement `Network` for the module and `NetworkConfig` for the configuration. Anything that
   depends on the problems (positions of variables, say) is read in `init`.
3. **Define the loss** and implement `Loss`. `Loss::prepare` runs once before training, with every
   problem the loss will score, for whatever is expensive to precompute.
4. **Train** with `train_model`. It saves the best weights in the checkpoint directory; saving the
   configuration (`config.save(dir.join("config.json"))`) is up to the caller.
5. **Load** with `load_network::<B, YourConfig>(dir, problems, device)`.
6. **Search** with `solve_with_network::<B, YourBatch<B>, YourNetwork<B>>`, with a `SolveConfig` whose
   search is `SearchConfig::NeuralPrior` or `NeuralPriorBp`.

Choosing the backend is the library's job: implement `TrainingJob` or `InferenceJob` on a struct
holding what the work needs, and pass it to `run_training` or `run_inference`. They run the job on
Cuda when there is a GPU and on the CPU otherwise, wrapped in `Autodiff` for training.

## Into this crate, and into Python

A network that ships with `aicad` additionally needs:

1. Its module under `src/neural/<name>/` (module header with its restrictions) and its losses under
   `src/learning/<name>/`, with a `recipes.rs` that fixes a training run from settings, as
   `learning/consformer/recipes.rs` does.
2. A way for a checkpoint to say which network it holds. Today the Python and `solve` entry points
   load a checkpoint as ConsFormer; the first network after it has to record the architecture in the
   checkpoint's `config.json` and dispatch on it.
3. The Python side: training entry points and a config class in `src/pyaicad/learn.rs`, a recipe in
   `python/pyaicad/learning.py`, and `python/pyaicad/_core.pyi`. These stay thin: the settings are
   converted and the Rust function does the work.

Extending the neural search with a new architecture is meant to happen in your own crate or fork
with the Rust API, as above, not through a Python hook.

## Worked example: a toy network

`tests/custom_network.rs` defines a deliberately silly network (it reads each variable alone and
learns to predict the value 0), because the point is the plumbing, which is the same for any network.
The file is compiled as a separate crate.

**The batch** holds what the network reads, and says how its logits become assignments and which
problems it accepts together:

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start batch]
:end-before: // [docs:end batch]
```

**The batcher** is burn's `Batcher`: how training turns a handful of samples (here plain
`Arc<Problem>`) into a batch. Inference never goes through it; the search builds batches with
`Batch::for_assignments`.

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start batcher]
:end-before: // [docs:end batcher]
```

**The module and its configuration** are burn's `Module` and `Config`, plus the two traits of this
crate. `NetworkConfig::init` receives the problems, for whatever the network reads from them:

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start network]
:end-before: // [docs:end network]
```

**The loss** returns a scalar for the logits of a batch:

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start loss]
:end-before: // [docs:end loss]
```

**Training.** The job runs on whichever backend the library picks (`Autodiff<Cuda>` or
`Autodiff<NdArray>`), calls `train_model`, and saves the configuration next to the weights it wrote:

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start train]
:end-before: // [docs:end train]
```

**Searching.** Loading reads `config.json` and the weights, and `solve_with_network` runs the
neural search with the loaded network:

```{literalinclude} ../../tests/custom_network.rs
:language: rust
:start-after: // [docs:start solve]
:end-before: // [docs:end solve]
```

Finally, `run_training(Train { .. })` and `run_inference(Solve { .. })` execute the jobs, as the test
at the end of the file does.

## Tests

- `check_problems` accepts what the network handles and refuses the rest with a clear message.
- Logits have the shape `(batch, variables, domain)`; `decode` only changes the variables of
  `var_masks` and stays inside the domains.
- A short training run lowers the loss, writes `weights`, and the checkpoint loads back to the same
  outputs.
- A search with the network returns one solution per problem, over a batch with more than one problem.
