//! The losses ConsFormer is trained with.

pub(crate) mod decomposed_mdd_loss;
pub(crate) mod loss;
pub(crate) mod utils;

pub(crate) use decomposed_mdd_loss::DecomposedMddLoss;
pub(crate) use loss::ConsFormerLoss;
