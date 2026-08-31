//! Small tensor <-> `Vec<Vec<isize>>` conversion helpers shared by every batched, tensor-driven
//! solver (`nls::NeuralLocalSearch`, `sampling::solve::SequentialImputationSolver`). Both represent
//! a batch of rows -- one per (problem, population member) pair -- as a `[rows, n]` integer tensor
//! holding each variable's raw problem value directly (not a domain index), and need to move back
//! and forth between that and plain Rust data to do CPU-side bookkeeping (constraint checking,
//! destroy-operator heuristics, MDD-based conditionals, ...) that has no tensor equivalent.

use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Packs `rows` (each of length `n`, raw problem values) into a `[rows.len(), n]` tensor.
pub fn rows_to_tensor<B: Backend>(
    rows: &[Vec<isize>],
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let data: Vec<i64> = rows
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as i64))
        .collect();
    Tensor::<B, 1, Int>::from_data(data.as_slice(), device).reshape([rows.len(), n])
}

/// The inverse of `rows_to_tensor`: unpacks a `[p, n]` tensor back into `p` rows of `n` raw
/// problem values each.
pub fn to_rows<B: Backend>(assignments: &Tensor<B, 2, Int>, p: usize, n: usize) -> Vec<Vec<isize>> {
    let flat: Vec<i64> = assignments
        .clone()
        .into_data()
        .to_vec::<B::IntElem>()
        .expect("assignment tensor should be integer")
        .into_iter()
        .map(|v| v.elem::<i64>())
        .collect();
    (0..p)
        .map(|i| {
            flat[i * n..(i + 1) * n]
                .iter()
                .map(|&v| v as isize)
                .collect()
        })
        .collect()
}
