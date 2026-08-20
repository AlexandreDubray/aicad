use burn::config::{Config, ConfigError};

/// Cofniguration of the neural local search
#[derive(Config, Debug)]
pub struct SolveConfig {
    /// If present, imposes a time limit in second
    #[config(default = "None")]
    pub time_limit: Option<u64>,
    /// If present, imposes a limit number of local search step
    #[config(default = "None")]
    pub iteration_limit: Option<usize>,
    #[config(default = "None")]
    pub seed: Option<u64>,
    /// Type of neural network used to produce the factorised distribution
    #[config(default = "String::from(\"consformer\")")]
    pub network_kind: String,
    /// Batch size to handle benchmarks
    #[config(default = "None")]
    pub batch_size: Option<usize>,
    /// Destroy operator
    #[config(default = "String::from(\"random\")")]
    pub destroy_kind: String,
    /// Maximum percentage of the variables changed each iteration
    #[config(default = 1.0)]
    pub destroy_fraction_max: f64,
    /// Minimum percentage of the variables changed each iteration
    #[config(default = 1.0)]
    pub destroy_fraction_min: f64,
    /// Number of epochs during which the destroy ratio goes from its maximum to minimum value
    #[config(default = 0)]
    pub mask_schedule_epochs: usize,
    /// If true, combine each variable's logits with the problem's MDDs (marginal
    /// product-of-experts) before decoding, instead of decoding each variable's logits
    /// independently
    #[config(default = false)]
    pub mdd_decode: bool,
    /// If true, decode the (possibly MDD-combined) logits stochastically
    #[config(default = false)]
    pub stochastic_decode: bool,
    /// Temperature scaling for stochastic decoding
    #[config(default = 1.0)]
    pub temperature: f64,
    /// In mdd decoding, perform gibbs sampling after block sampling
    #[config(default = true)]
    pub mdd_gibbs_cleanup: bool,
    /// If gibbs sampling is applied in mdd-decoding, how many sampling round
    #[config(default = 4)]
    pub gibbs_round: usize,
}

impl SolveConfig {
    /// Like `Config::load`, but a JSON file missing some fields (e.g. one saved before a field
    /// existed) falls back to that field's default instead of failing. `#[config(default = ...)]`
    /// only wires up `SolveConfig::new`'s optional arguments -- burn's generated `Deserialize` impl
    /// still requires every field to be present -- so this merges the file's JSON object on top of
    /// `SolveConfig::default()`'s JSON object before deserializing.
    pub fn load_lenient<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|_| ConfigError::FileNotFound(path.to_string_lossy().to_string()))?;
        let file_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|err| ConfigError::InvalidFormat(format!("{err}")))?;
        let mut merged = serde_json::to_value(Self::default())
            .map_err(|err| ConfigError::InvalidFormat(format!("{err}")))?;

        match (&mut merged, file_value) {
            (serde_json::Value::Object(default_map), serde_json::Value::Object(file_map)) => {
                default_map.extend(file_map);
            }
            _ => {
                return Err(ConfigError::InvalidFormat(
                    "expected a JSON object".to_string(),
                ));
            }
        }

        serde_json::from_value(merged).map_err(|err| ConfigError::InvalidFormat(format!("{err}")))
    }

    /// `destroy_fraction_max` is the fraction used at iteration 0 of the mask schedule,
    /// `destroy_fraction_min` is what it's annealed down to -- so `max < min` is never sensible.
    pub fn validate(&self) -> Result<(), String> {
        if self.destroy_fraction_max < self.destroy_fraction_min {
            return Err(format!(
                "destroy_fraction_max ({}) must be >= destroy_fraction_min ({})",
                self.destroy_fraction_max, self.destroy_fraction_min
            ));
        }
        Ok(())
    }
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            time_limit: None,
            iteration_limit: None,
            seed: None,
            network_kind: String::from("consformer"),
            batch_size: None,
            destroy_kind: String::from("random"),
            destroy_fraction_max: 1.0,
            destroy_fraction_min: 1.0,
            mask_schedule_epochs: 0,
            mdd_decode: false,
            stochastic_decode: false,
            temperature: 0.1,
            mdd_gibbs_cleanup: true,
            gibbs_round: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_documented_zero_config_defaults() {
        let config = SolveConfig::default();
        assert_eq!(config.network_kind, "consformer");
        assert_eq!(config.destroy_kind, "random");
        assert_eq!(config.destroy_fraction_max, 1.0);
        assert_eq!(config.destroy_fraction_min, 1.0);
        assert!(!config.mdd_decode);
        assert!(!config.stochastic_decode);
        assert_eq!(config.gibbs_round, 4);
        assert!(config.mdd_gibbs_cleanup);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_max_below_min() {
        let mut config = SolveConfig::default();
        config.destroy_fraction_max = 0.3;
        config.destroy_fraction_min = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_max_equal_to_min() {
        let mut config = SolveConfig::default();
        config.destroy_fraction_max = 0.5;
        config.destroy_fraction_min = 0.5;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn round_trips_through_json() {
        let dir =
            std::env::temp_dir().join(format!("aicad_solve_config_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("solve_config.json");

        let mut config = SolveConfig::default();
        config.destroy_kind = String::from("related");
        config.mask_schedule_epochs = 20;
        config.save(&path).expect("save should succeed");

        let loaded = SolveConfig::load(&path).expect("load should succeed");
        assert_eq!(loaded.destroy_kind, "related");
        assert_eq!(loaded.mask_schedule_epochs, 20);
        // Field(s) left at their default should still round-trip correctly.
        assert!(!loaded.mdd_decode);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let dir = std::env::temp_dir().join(format!(
            "aicad_solve_config_partial_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial_solve_config.json");
        std::fs::write(&path, r#"{"destroy_kind": "worst"}"#).unwrap();

        let loaded = SolveConfig::load_lenient(&path).expect("load should succeed");
        assert_eq!(loaded.destroy_kind, "worst");
        assert!(!loaded.mdd_decode);
        assert_eq!(loaded.batch_size, None);

        std::fs::remove_dir_all(&dir).ok();
    }
}
