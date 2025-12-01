use super::*;

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

    /// Returns the value of the domain at the given index
    pub fn get_value(&self, index: ValueIndex) -> isize {
        self.domain[index.0]
    }

    /// Returns the probability that the variable takes the value from its domain at the given
    /// index.
    pub fn get_probability(&self, index: ValueIndex) -> f64 {
        self.probabilities[index.0]
    }

    /// Returns the number of elements in the domain
    pub fn domain_size(&self) -> usize {
        self.domain.len()
    }

    /// Iterates over the domain of the variable
    pub fn iter_domain(&self) -> impl Iterator<Item = isize> {
        self.domain.iter().copied()
    }

    /// Sets the domain of the variable to the given values
    pub fn set_domain(&mut self, domain: Vec<isize>) {
        let n = domain.len();
        let probabilities = (0..n).map(|_| 1.0 / n as f64).collect::<Vec<f64>>();
        self.probabilities = probabilities;
        self.domain = domain;
    }

}
