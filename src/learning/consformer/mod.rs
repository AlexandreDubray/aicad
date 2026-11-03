//! The losses ConsFormer is trained with.

mod decomposed_mdd_loss;
mod loss;
mod recipes;
mod utils;

pub use decomposed_mdd_loss::DecomposedMddLoss;
pub use loss::ConsFormerLoss;
pub use recipes::{
    train_consformer, train_consformer_mdd, train_consformer_mdd_on, train_consformer_on,
};
