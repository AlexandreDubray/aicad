use super::*;

#[derive(Clone)]
pub struct Edge {
    from: NodeIndex,
    to: NodeIndex,
    assignment: isize,
    active: bool,
}

impl Edge {
    
    pub fn new(from: NodeIndex, to: NodeIndex, assignment: isize) -> Self {
        Self {
            from,
            to,
            assignment,
            active: true,
        }
    }

    pub fn from(&self) -> NodeIndex {
        self.from
    }

    pub fn to(&self) -> NodeIndex {
        self.to
    }

    pub fn assignment(&self) -> isize {
        self.assignment
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}
