use super::*;

pub struct Variable {
    domain: Vec<isize>,
    probabilities: Option<Vec<f64>>,
    constraints: Vec<ConstraintIndex>,
}

impl Variable {

    pub fn new(domain: Vec<isize>, probabilities: Option<Vec<f64>>) -> Self {
        Self {
            domain,
            probabilities,
            constraints: vec![],
        }
    }


    /// Returns the value of the domain at the given index
    pub fn get_value(&self, index: ValueIndex) -> isize {
        self.domain[index.0]
    }

    /// Returns the probability that the variable takes the value from its domain at the given
    /// index.
    pub fn get_probability(&self, index: ValueIndex) -> f64 {
        match &self.probabilities {
            None => panic!("Trying to access probabilities on unweighted variable"),
            Some(p) => p[index.0],
        }
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
        self.domain = domain;
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
