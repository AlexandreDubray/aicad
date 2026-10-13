//! Conversions between rows of assignments and an integer tensor.
//!
//! # Meaning
//!
//! The neural local search keeps a batch as a `[rows, n]` integer tensor with one row per
//! (problem, population member) pair, and does its bookkeeping (constraint checks, destroy
//! operators, MDD conditioning) on plain `Vec<Vec<isize>>` rows on the CPU. [`rows_to_tensor`]
//! and [`to_rows`] move between the two.
//!
//! # Design notes
//!
//! - **The tensor holds raw problem values, not domain indices.** A row can then be checked
//!   against the constraints as it is, without translating through each variable's domain.

use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Packs `rows` (each of length `n`, raw problem values) into a `[rows.len(), n]` tensor.
pub(crate) fn rows_to_tensor<B: Backend>(
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
pub(crate) fn to_rows<B: Backend>(
    assignments: &Tensor<B, 2, Int>,
    p: usize,
    n: usize,
) -> Vec<Vec<isize>> {
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
