use crate::core::*;

pub struct Variable {
    domain: Vec<isize>,
    probabilities: Vec<f64>,
}

impl Variable {

    pub fn new(domain: Vec<isize>) -> Self {
        let n = domain.len();
        let probabilities = (0..n).map(|_| 1.0 / n as f64).collect::<Vec<f64>>();
        Self {
            domain,
            probabilities,
        }
    }
}

impl Variable {

    pub fn get_value(&self, value: ValueIndex) -> isize {
        self.domain[value.0]
    }

    pub fn get_probability(&self, value: ValueIndex) -> f64 {
        self.probabilities[value.0]
    }

    pub fn domain_size(&self) -> usize {
        self.domain.len()
    }
}
