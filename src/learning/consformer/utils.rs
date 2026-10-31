//! Helpers shared by the losses: turning a network output into probabilities that a loss can score.
//!
//! Both losses read the logits as the probabilities of the variables the network re-predicts,
//! and as one-hot values for the others: [`gumbel_softmax`] turns the logits into probabilities
//! and [`blend_with_current`] pins the variables that are not re-predicted to their current value.

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// Soft (differentiable) Gumbel-softmax relaxation: adds Gumbel(0,1) noise
/// to `logits` before the softmax, instead of taking a plain softmax.
pub(super) fn gumbel_softmax<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = logits.device();
    let u = Tensor::<B, 3>::random(
        logits.dims(),
        burn::tensor::Distribution::Uniform(1e-20, 1.0),
        &device,
    );
    let neg_log_u = -u.log();
    let gumbel = -neg_log_u.log();
    softmax(logits + gumbel, 2)
}

pub(super) fn blend_with_current<B: Backend>(
    probs: Tensor<B, 3>,
    assignments: Tensor<B, 2, Int>,
    var_masks: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let [batch_size, number_vars, domain_size] = probs.dims();
    let device = probs.device();

    let arange: Tensor<B, 3, Int> = Tensor::<B, 1, Int>::arange(0..domain_size as i64, &device)
        .reshape([1, 1, domain_size])
        .repeat_dim(0, batch_size)
        .repeat_dim(1, number_vars);
    let assign_3d: Tensor<B, 3, Int> = assignments
        .reshape([batch_size, number_vars, 1])
        .repeat_dim(2, domain_size);
    let one_hot: Tensor<B, 3> = assign_3d.equal(arange).float();

    let mask_3d: Tensor<B, 3, Bool> = var_masks
        .reshape([batch_size, number_vars, 1])
        .repeat_dim(2, domain_size);

    one_hot.mask_where(mask_3d, probs)
}
