# Training a network

```{note}
This chapter needs the `learning` feature (see [](../install.md)).
```

The neural searches repair with the predictions of a network. A network is trained on a set of
problems, one problem per example, and the trained network lives in a *checkpoint* directory, so a
network trained once can be used by any later program.

```{admonition} Each network has its own restrictions
:class: important
This chapter shows the workflow. What a given network accepts (how its problems may differ, how
many variables or values it handles, what it needs from a problem) depends on the network, and is
documented with it in the Rust documentation of the `aicad` crate, for the ConsFormer network in
the `neural::consformer` module. The examples below use graph colourings of the same size because
that is what the ConsFormer network accepts. The functions check what they can and return an error
that names the problem.
```

## Data

The training problems are a list of `Arc<Problem>`. Training needs a validation set to pick the best
network; give one, or training holds out 20% of the problems itself. (The Python guide splits with
`Dataset.split`; in Rust you split your own `Vec`.)

There is no test set in training on purpose. The problems you evaluate on are yours to keep apart:
generate them separately and never pass them to training, so that the network has certainly not seen
them. The `coloring` function is the one of the previous chapter.

```{literalinclude} ../../tests/rust_guide/learning.rs
:language: rust
:start-after: // [docs:start data]
:end-before: // [docs:end data]
```

## Training

A network is described by its configuration (`ConsFormerConfig`), and the optimisation by a
`TrainingConfig`. `train_consformer_mdd` scores the network with the exact probability that its
predictions satisfy each constraint, computed on the constraint's diagram, labelled beforehand by
`bp_rounds` rounds of belief propagation (0 scores the predictions as they are, the messages being
constants of the gradient); `train_consformer` uses a
hand-written penalty per constraint type. Both run on the best device found (CUDA if there is a GPU,
the CPU otherwise), and have an `_on` variant that takes the backend. They write the best network and
its configuration to the checkpoint directory.

```{literalinclude} ../../tests/rust_guide/learning.rs
:language: rust
:start-after: // [docs:start train]
:end-before: // [docs:end train]
```

A network from another crate is trained with `train_model`: see *Contributing*.

## Searching with the network

The checkpoint is given to `solve`, with a neural architecture in the configuration. The settings
are the ones of the belief propagation search, plus what the network adds.

```{literalinclude} ../../tests/rust_guide/learning.rs
:language: rust
:start-after: // [docs:start search]
:end-before: // [docs:end search]
```

## Using a checkpoint later

A checkpoint is a directory: the configuration is `config.json` and the weights are next to it. A
network is tied to the problems it was trained on, in a way that depends on the network (see the
note at the top of the chapter): problems that fit it are solved with it directly. `with_batch_size(1)`
solves them one after the other, each with its own limits, as a deployment would; by default all the
problems are stepped together on the device.

```{literalinclude} ../../tests/rust_guide/learning.rs
:language: rust
:start-after: // [docs:start reuse]
:end-before: // [docs:end reuse]
```

Compilation and local search are reproducible given their seed. Training is seeded too, but
bit-for-bit reproducibility of a training run is not guaranteed.
