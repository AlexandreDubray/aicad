//! The optimisers that train a network, chosen and configured at run time.
//!
//! Burn gives each optimiser its own type, so the training loop could only be written for one of
//! them. [`OptimiserConfig`] names the optimiser to use with its hyper-parameters and learning
//! rate, and [`Optimiser`] is what it builds: a single type that takes a gradient step whichever
//! optimiser is behind it.
//!
//! The hyper-parameters are those of Burn's own configurations ([`AdamConfig`], [`AdamWConfig`]
//! and [`SgdConfig`]), which already hold the weight decay, the momentum and the gradient
//! clipping of their optimiser. Adding an optimiser means one variant in each of the two enums.

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{
    Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, Optimizer, Sgd, SgdConfig,
};
use burn::tensor::backend::AutodiffBackend;

/// Which optimiser to train with, and how it is configured.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum OptimiserConfig {
    /// Adam (Kingma and Ba, 2015), with the weight decay of $L_2$ regularisation if configured.
    Adam {
        /// Learning rate.
        lr: f64,
        /// Moment decay rates, epsilon, AMSGrad, weight decay and gradient clipping.
        config: AdamConfig,
    },
    /// Adam with a decoupled weight decay (Loshchilov and Hutter, 2019).
    AdamW {
        /// Learning rate.
        lr: f64,
        /// Moment decay rates, epsilon, AMSGrad, weight decay and gradient clipping.
        config: AdamWConfig,
    },
    /// Stochastic gradient descent, with momentum and weight decay if configured.
    Sgd {
        /// Learning rate.
        lr: f64,
        /// Momentum, weight decay and gradient clipping.
        config: SgdConfig,
    },
}

impl Default for OptimiserConfig {
    fn default() -> Self {
        Self::AdamW {
            lr: 3e-4,
            config: AdamWConfig::new(),
        }
    }
}

impl OptimiserConfig {
    /// The learning rate.
    pub(crate) fn lr(&self) -> f64 {
        match self {
            Self::Adam { lr, .. } | Self::AdamW { lr, .. } | Self::Sgd { lr, .. } => *lr,
        }
    }

    /// Builds the optimiser, for a network of type `M`.
    pub(crate) fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> Optimiser<B, M> {
        let algorithm = match self {
            Self::Adam { config, .. } => Algorithm::Adam(config.init()),
            Self::AdamW { config, .. } => Algorithm::AdamW(config.init()),
            Self::Sgd { config, .. } => Algorithm::Sgd(config.init()),
        };
        Optimiser {
            lr: self.lr(),
            algorithm,
        }
    }
}

/// The optimiser of [`OptimiserConfig`] that was chosen, with its state.
pub(crate) struct Optimiser<B: AutodiffBackend, M: AutodiffModule<B>> {
    lr: f64,
    algorithm: Algorithm<B, M>,
}

enum Algorithm<B: AutodiffBackend, M: AutodiffModule<B>> {
    Adam(OptimizerAdaptor<Adam, M, B>),
    AdamW(OptimizerAdaptor<AdamW, M, B>),
    Sgd(OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>),
}

impl<B: AutodiffBackend, M: AutodiffModule<B>> Optimiser<B, M> {
    /// Moves the parameters of `module` one step along `grads`, and returns the updated module.
    pub(crate) fn step(&mut self, module: M, grads: GradientsParams) -> M {
        match &mut self.algorithm {
            Algorithm::Adam(optim) => optim.step(self.lr, module, grads),
            Algorithm::AdamW(optim) => optim.step(self.lr, module, grads),
            Algorithm::Sgd(optim) => optim.step(self.lr, module, grads),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::grad_clipping::GradientClippingConfig;
    use burn::nn::{Linear, LinearConfig};
    use burn::optim::momentum::MomentumConfig;
    use burn::tensor::Tensor;

    use super::*;

    type Ad = Autodiff<NdArray>;

    /// The squared norm of the output of `model` on a fixed input: it is zero for the zero model,
    /// and any optimiser must bring it down.
    fn loss(model: &Linear<Ad>, device: &NdArrayDevice) -> Tensor<Ad, 1> {
        let input = Tensor::<Ad, 2>::from_floats([[1.0, 2.0], [-1.0, 0.5]], device);
        let out = model.forward(input);
        (out.clone() * out).sum().reshape([1])
    }

    fn value(loss: Tensor<Ad, 1>) -> f32 {
        loss.into_data().to_vec::<f32>().unwrap()[0]
    }

    #[test]
    fn every_optimiser_lowers_the_loss() {
        let device = NdArrayDevice::default();
        let configs = [
            OptimiserConfig::default(),
            OptimiserConfig::AdamW {
                lr: 1e-2,
                config: AdamWConfig::new(),
            },
            OptimiserConfig::Sgd {
                lr: 1e-2,
                config: SgdConfig::new()
                    .with_momentum(Some(MomentumConfig::new()))
                    .with_gradient_clipping(Some(GradientClippingConfig::Norm(10.0))),
            },
        ];
        for config in configs {
            let config = match config {
                OptimiserConfig::Adam { config, .. } => OptimiserConfig::Adam { lr: 1e-2, config },
                other => other,
            };
            let mut model = LinearConfig::new(2, 2).init::<Ad>(&device);
            let mut optimiser = config.init::<Ad, Linear<Ad>>();
            let before = value(loss(&model, &device));
            for _ in 0..20 {
                let grads = GradientsParams::from_grads(loss(&model, &device).backward(), &model);
                model = optimiser.step(model, grads);
            }
            let after = value(loss(&model, &device));
            assert!(
                after < before,
                "{config:?}: loss went from {before} to {after}"
            );
        }
    }

    #[test]
    fn the_configuration_survives_a_round_trip_through_json() {
        let config = OptimiserConfig::Sgd {
            lr: 0.5,
            config: SgdConfig::new().with_momentum(Some(MomentumConfig::new())),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: OptimiserConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
        assert_eq!(back.lr(), 0.5);
    }
}
