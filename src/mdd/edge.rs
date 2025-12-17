use super::*;

pub struct Edge {
    layer_from: LayerIndex,
    from: NodeIndex,
    to: NodeIndex,
    assignment: isize,
    active: bool,
}

impl Edge {
    
    pub fn new(layer_from: LayerIndex,
        from: NodeIndex,
        to: NodeIndex,
        assignment: isize,
        ) -> Self {
        Self {
            layer_from,
            from,
            to,
            assignment,
            active: true,
        }
    }

    pub fn layer_from(&self) -> LayerIndex {
        self.layer_from
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
