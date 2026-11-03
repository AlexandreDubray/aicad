# Adding a destroy operator

A destroy operator chooses which variables of an assignment a local search frees, or a training step
masks. It is a small trait in `src/destroy.rs`:

```rust
pub trait DestroyOperator: Send + Sync {
    /// The indices of the variables of `problem` to destroy, given the current assignment.
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;
}
```

The `rng` is the only source of randomness an operator may use: the engine seeds one per row, in
order, so a seeded run stays reproducible even though rows are processed in parallel.

## From another crate

Implement the trait on your own type and put it in a `Box<dyn DestroyOperator>`. Local searches that
take an operator (`destroy_rows`, `ConsFormerBatcher::destroy`, the neural searches built with
`new`) use it directly. Nothing needs to be merged here.

## Into this crate, and into Python

1. Write the struct and its `impl DestroyOperator` in `src/destroy.rs`, with a `fraction`-style field
   like the existing ones so that it can be built from `DestroyKind::build(fraction)`.
2. Add a variant to `DestroyKind` and to its `build`.
3. Add the variant to `PyDestroyKind` and its conversion in `src/pyaicad/local_search.rs`, and the
   string name to the two tables in `python/pyaicad/search.py` and `python/pyaicad/learning.py`
   (`"random"`, `"worst"`, `"related"` today). Update the docstrings that list the names.

## Worked example: `RandomDestroy`

The random operator is the shortest one. It frees a `fraction` of the *free* variables (those with more than one
value), always at least one, and uses only the `rng` it is given.

```{literalinclude} ../../src/destroy.rs
:language: rust
:start-at: /// Uniformly destroys a `fraction` of the free (domain size > 1) variables.
:end-before: /// Stochastic worst removal
```

Three details are worth copying into a new operator:

- A variable with a single value is never destroyed: there is nothing to repair, and a training mask
  would otherwise put the given values of a problem (the hints of a Sudoku) into the loss.
- Nothing is destroyed when there is no free variable, instead of panicking.
- The result is clamped to at least one variable, so that a small `fraction` still moves the search.

It is made available by name through `DestroyKind`, which is a plain value that builds the operator:

```{literalinclude} ../../src/destroy.rs
:language: rust
:start-at: pub enum DestroyKind {
:end-at: Related,
:append: "}"
```

```{literalinclude} ../../src/destroy.rs
:language: rust
:start-at: pub fn build(self, fraction: f64) -> Box<dyn DestroyOperator> {
:end-at: DestroyKind::Related =>
:dedent: 4
:append: "    }\n}"
```

and to Python as the string `"random"` in the tables of `search.py` and `learning.py`.

## Tests

- It returns only valid indices, never repeats one, and handles a problem with no free variable
  (all domains of size one).
- With a fixed seed it returns the same variables twice.
- Its intent holds on a small case (for a "worst" operator, it picks the variables of violated
  constraints more often than the others).
