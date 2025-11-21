use super::Constraint;
use crate::core::VariableIndex;

pub struct AllDifferent {
    variables: Vec<VariableIndex>,
}

impl AllDifferent {

    pub fn new(variables: Vec<VariableIndex>) -> Self {
        Self { variables }
    }
}

impl Constraint for AllDifferent {

    fn get_properties(&self) -> Vec<u64> {
        let nb_words = self.variables.len() / 64;
        vec![0; nb_words]
    }

    fn get_root_property(&self) -> Vec<u64> {
        let nb_words = self.variables.len() / 64;
        vec![0; nb_words]
    }

    fn get_sink_property(&self) -> Vec<u64> {
        let nb_words = self.variables.len() / 64;
        vec![0; nb_words]
    }

    fn integrate_edge(&self, property: &mut Vec<u64>, value: isize) {
    }

    fn aggregate_properties(&self, target: &mut Vec<u64>, properties: &[Vec<u64>]) {
    }
}
