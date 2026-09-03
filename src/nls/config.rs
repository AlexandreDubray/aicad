use burn::config::{Config, ConfigError};

/// Cofniguration of the neural local search
#[derive(Config, Debug)]
pub struct SolveConfig {
    #[config(default = "None")]
    pub time_limit: Option<u64>,
    #[config(default = "None")]
    pub iteration_limit: Option<usize>,
    #[config(default = "None")]
    pub seed: Option<u64>,
    #[config(default = "String::from(\"consformer\")")]
    pub network_kind: String,
    #[config(default = "None")]
    pub batch_size: Option<usize>,
    #[config(default = "String::from(\"random\")")]
    pub destroy_kind: String,
    #[config(default = 1.0)]
    pub destroy_fraction: f64,
    #[config(default = false)]
    pub stochastic_decode: bool,
    #[config(default = 1.0)]
    pub temperature: f64,
    #[config(default = "String::from(\"logits\")")]
    pub decode_kind: String,
    #[config(default = 5)]
    pub bp_iterations: usize,
    #[config(default = 0)]
    pub mdd_grouping_window_size: usize,
}

impl SolveConfig {
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
            destroy_fraction: 1.0,
            stochastic_decode: false,
            temperature: 1.0,
            decode_kind: String::from("logits"),
            bp_iterations: 5,
            mdd_grouping_window_size: 0,
        }
    }
}
