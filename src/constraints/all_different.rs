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
}
