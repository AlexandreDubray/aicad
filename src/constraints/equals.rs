use crate::modelling::*;

pub struct Equals {
    x: VariableIndex,
    y: VariableIndex,
}

impl Equals {

    pub fn new(problem: &Problem, x: VariableIndex, y: VariableIndex) -> Self {
        Self { x, y }
    }

}
