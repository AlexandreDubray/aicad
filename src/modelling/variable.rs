use super::*;

pub struct Variable {
    domain: Vec<isize>,
    probabilities: Vec<f64>,
    constraints: Vec<ConstraintIndex>,
}

impl Variable {

    pub fn new(domain: Vec<isize>, probs: Option<Vec<f64>>) -> Self {
        let probabilities = match probs {
            Some(probabilities) => probabilities,
            None => {
                let n = domain.len();
                let p = 1.0 / (n as f64);
                vec![p; n]
            },
        };
        Self {
            domain,
            probabilities,
            constraints: vec![],
        }
    }


    /// Returns the value of the domain at the given index
    pub fn value(&self, index: ValueIndex) -> isize {
        self.domain[index.0]
    }

    /// Returns the probability that the variable takes the value from its domain at the given
    /// index.
    pub fn probability(&self, index: ValueIndex) -> f64 {
        if self.probabilities.is_empty() {
            panic!("Trying to access probabilities on unweighted variable");
        }
        self.probabilities[index.0]
    }

    pub fn set_probabilities(&mut self, probabilities: &[f64]) {
        self.probabilities = probabilities.to_owned();
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
        self.domain = domain;
        let p = 1.0 / (n as f64);
        self.probabilities = vec![p; n];
    }

    pub fn add_constraint(&mut self, constraint: ConstraintIndex) {
        self.constraints.push(constraint);
    }

    pub fn iter_constraints(&self) -> impl Iterator<Item = ConstraintIndex> {
        self.constraints.iter().copied()
    }

    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

}
