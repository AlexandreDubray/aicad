//! The examples of the Rust guide of the documentation, compiled and run as tests.
//!
//! The guide pulls its code from these files, so the examples cannot drift from the library: a
//! change that breaks one fails `cargo test`. Each chapter mirrors the chapter of the Python guide
//! of the same name. This file is compiled as a separate crate, so it uses only the public API.

mod compiling;
#[cfg(feature = "learning")]
mod learning;
mod local_search;
mod modelling;
mod relaxing;
