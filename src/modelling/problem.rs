use crate::constraints::Constraint;
use super::*;
use super::variable::Variable;

///This structure represent a constrained optimisation problem.
#[derive(Default)]
pub struct Problem {
    /// Variables of the problem
    variables: Vec<Variable>,
    /// Constraints of the problem.
    constraints: Vec< Box<dyn Constraint>>,
    /// Order of the variables for the MDD,
    variable_ordering: Vec<usize>,
}

impl Problem {

    /// Adds a variable with the given domain to the problem and returns its index.
    pub fn add_variable(&mut self, domain: Vec<isize>) -> VariableIndex {
        let ret = VariableIndex(self.variables.len());
        self.variables.push(Variable::new(domain));
        ret
    }

    /// Adds n variables, with the same domain, to the problem and return their indexes.
    pub fn add_variables(&mut self, n: usize, domain: Vec<isize>) -> Vec<VariableIndex> {
        (0..n).map(|_| self.add_variable(domain.clone())).collect()
    }

    /// Adds a constraint to the problem and returns its index.
    pub fn add_constraint(&mut self, constraint: impl Constraint + 'static) -> ConstraintIndex {
        let ret = ConstraintIndex(self.constraints.len());
        self.constraints.push(Box::new(constraint));
        ret
    }

    /// Returns the number of variables in the problem
    pub fn number_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the number of constraints in the problem
    pub fn number_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns a pointer to the variable ordering
    pub fn variable_ordering(&self) -> &Vec<usize> {
        &self.variable_ordering
    }

    /// Returns the layer of a given variable
    pub fn variable_layer(&self, variable: VariableIndex) -> usize {
        self.variable_ordering[variable.0]
    }

    /// Sets the variable ordering for the problem
    pub fn set_variable_ordering(&mut self, ordering: Vec<usize>) {
        debug_assert!(ordering.len() == self.number_variables());
        for constraint in 0..self.constraints.len() {
            self.constraints[constraint].update_variable_ordering(&ordering);
        }
        self.variable_ordering = ordering;
    }

    /// Iterates over the constraints
    pub fn iter_constraints(&self) -> impl Iterator<Item = ConstraintIndex> {
        (0..self.constraints.len()).map(ConstraintIndex)
    }
}

impl std::ops::Index<VariableIndex> for Problem {

    type Output = Variable;

    fn index(&self, index: VariableIndex) -> &Self::Output {
        &self.variables[index.0]
    }

}

impl std::ops::IndexMut<VariableIndex> for Problem {
    fn index_mut(&mut self, index: VariableIndex) -> &mut Self::Output {
        &mut self.variables[index.0]
    }
}

impl std::ops::Index<ConstraintIndex> for Problem {

    type Output = Box<dyn Constraint>;

    fn index(&self, index: ConstraintIndex) -> &Self::Output {
        &self.constraints[index.0]
    }
}

impl std::ops::IndexMut<ConstraintIndex> for Problem {
    fn index_mut(&mut self, index: ConstraintIndex) -> &mut Self::Output {
        &mut self.constraints[index.0]
    }
}
