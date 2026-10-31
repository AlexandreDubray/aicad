//! Python view of the optimisers of `crate::learning::optimiser`.
//!
//! One class per optimiser, each holding every hyper-parameter, so that the Python side can pick
//! an optimiser by building the matching class. The defaults live in Python (the dataclasses
//! of `pyaicad.learning`); nothing here chooses a value.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamConfig, AdamWConfig, SgdConfig};
use pyo3::prelude::*;

use crate::learning::optimiser::OptimiserConfig;

/// An optimiser and its hyper-parameters.
///
/// For every optimiser, `grad_clip_norm` clips the global $L_2$ norm of each gradient tensor and
/// `grad_clip_value` clips each element of the gradient. When both are set the norm takes
/// precedence.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyOptimiser {
    /// Adam, whose `weight_decay` is an $L_2$ penalty (`None` for none).
    Adam {
        lr: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        amsgrad: bool,
        weight_decay: Option<f64>,
        grad_clip_norm: Option<f64>,
        grad_clip_value: Option<f64>,
    },
    /// Adam with a decoupled weight decay.
    AdamW {
        lr: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        amsgrad: bool,
        weight_decay: f64,
        grad_clip_norm: Option<f64>,
        grad_clip_value: Option<f64>,
    },
    /// Stochastic gradient descent. Without `momentum` the `dampening` and `nesterov` settings
    /// have no effect.
    Sgd {
        lr: f64,
        momentum: Option<f64>,
        dampening: f64,
        nesterov: bool,
        weight_decay: Option<f64>,
        grad_clip_norm: Option<f64>,
        grad_clip_value: Option<f64>,
    },
}

fn clipping(norm: Option<f64>, value: Option<f64>) -> Option<GradientClippingConfig> {
    match (norm, value) {
        (Some(norm), _) => Some(GradientClippingConfig::Norm(norm as f32)),
        (None, Some(value)) => Some(GradientClippingConfig::Value(value as f32)),
        (None, None) => None,
    }
}

fn l2(penalty: Option<f64>) -> Option<WeightDecayConfig> {
    penalty.map(|p| WeightDecayConfig::new(p as f32))
}

impl From<&PyOptimiser> for OptimiserConfig {
    fn from(optimiser: &PyOptimiser) -> Self {
        match *optimiser {
            PyOptimiser::Adam {
                lr,
                beta_1,
                beta_2,
                epsilon,
                amsgrad,
                weight_decay,
                grad_clip_norm,
                grad_clip_value,
            } => Self::Adam {
                lr,
                config: AdamConfig::new()
                    .with_beta_1(beta_1 as f32)
                    .with_beta_2(beta_2 as f32)
                    .with_epsilon(epsilon as f32)
                    .with_amsgrad(amsgrad)
                    .with_weight_decay(l2(weight_decay))
                    .with_grad_clipping(clipping(grad_clip_norm, grad_clip_value)),
            },
            PyOptimiser::AdamW {
                lr,
                beta_1,
                beta_2,
                epsilon,
                amsgrad,
                weight_decay,
                grad_clip_norm,
                grad_clip_value,
            } => Self::AdamW {
                lr,
                config: AdamWConfig::new()
                    .with_beta_1(beta_1 as f32)
                    .with_beta_2(beta_2 as f32)
                    .with_epsilon(epsilon as f32)
                    .with_amsgrad(amsgrad)
                    .with_weight_decay(weight_decay as f32)
                    .with_grad_clipping(clipping(grad_clip_norm, grad_clip_value)),
            },
            PyOptimiser::Sgd {
                lr,
                momentum,
                dampening,
                nesterov,
                weight_decay,
                grad_clip_norm,
                grad_clip_value,
            } => Self::Sgd {
                lr,
                config: SgdConfig::new()
                    .with_momentum(momentum.map(|momentum| {
                        MomentumConfig::new()
                            .with_momentum(momentum)
                            .with_dampening(dampening)
                            .with_nesterov(nesterov)
                    }))
                    .with_weight_decay(l2(weight_decay))
                    .with_gradient_clipping(clipping(grad_clip_norm, grad_clip_value)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_norm_takes_precedence_over_the_value() {
        let optimiser = PyOptimiser::Sgd {
            lr: 0.1,
            momentum: None,
            dampening: 0.0,
            nesterov: false,
            weight_decay: None,
            grad_clip_norm: Some(1.0),
            grad_clip_value: Some(5.0),
        };
        let json = serde_json::to_string(&OptimiserConfig::from(&optimiser)).unwrap();
        assert!(json.contains("Norm"), "{json}");
        assert!(!json.contains("Value"), "{json}");
    }

    #[test]
    fn each_class_becomes_its_optimiser_with_its_learning_rate() {
        let adamw = PyOptimiser::AdamW {
            lr: 0.25,
            beta_1: 0.8,
            beta_2: 0.9,
            epsilon: 1e-6,
            amsgrad: true,
            weight_decay: 0.01,
            grad_clip_norm: None,
            grad_clip_value: None,
        };
        let config = OptimiserConfig::from(&adamw);
        assert!(matches!(config, OptimiserConfig::AdamW { .. }));
        assert_eq!(config.lr(), 0.25);
    }
}
