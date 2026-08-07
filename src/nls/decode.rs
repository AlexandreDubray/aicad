//! Decoding operators for neural local search.
//! The following decoding strategies are implemented:
//!     - Use an argmax: Always select the value associated with the highest logit
//!     - Use a softmax: sample proportionnaly to the logits

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Distribution, Int, Tensor};

/// Turns this iteration's logits into the next assignment. Only positions
/// flagged in `destroy_mask` may change; everywhere else the current value
/// is kept, regardless of what the network predicted there.
pub trait DecodingOperator<B: Backend>: Send + Sync {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Bool>,
        current: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int>;
}

/// Greedy / MAP decoding: takes the most likely value per variable.
pub struct Argmax;

impl<B: Backend> DecodingOperator<B> for Argmax {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Bool>,
        current: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        let proposed: Tensor<B, 2, Int> = logits.argmax(2).squeeze();
        blend(proposed, current, destroy_mask)
    }
}

/// Stochastic decoding: samples a value per variable from
/// `softmax(logits / temperature)`.
pub struct Sampling {
    pub temperature: f64,
}

impl<B: Backend> DecodingOperator<B> for Sampling {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Bool>,
        current: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let u = Tensor::<B, 3>::random(logits.dims(), Distribution::Uniform(1e-20, 1.0), &device);
        let neg_log_u = -u.log(); // -ln(u), > 0 since u in (0, 1)
        let gumbel = -neg_log_u.log(); // Gumbel(0, 1) noise: -ln(-ln(u))

        let scaled = logits.div_scalar(self.temperature) + gumbel;
        let proposed: Tensor<B, 2, Int> = scaled.argmax(2).squeeze();
        blend(proposed, current, destroy_mask)
    }
}

/// `result[i] = destroy_mask[i] ? proposed[i] : current[i]`
fn blend<B: Backend>(
    proposed: Tensor<B, 2, Int>,
    current: Tensor<B, 2, Int>,
    destroy_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 2, Int> {
    let device = destroy_mask.device();
    let destroy_i = destroy_mask.int();
    let keep_i = Tensor::<B, 2, Int>::ones(destroy_i.dims(), &device) - destroy_i.clone();
    proposed * destroy_i + current * keep_i
}
