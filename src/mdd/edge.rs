use super::*;

pub struct Edge {
    layer_from: LayerIndex,
    from: NodeIndex,
    to: NodeIndex,
    assignment: isize,
    next_parent: Option<EdgeIndex>,
    next_child: Option<EdgeIndex>,
}

impl Edge {
    
    pub fn new(layer_from: LayerIndex, from: NodeIndex, to: NodeIndex, assignment: isize, next_parent: Option<EdgeIndex>, next_child: Option<EdgeIndex>) -> Self {
        Self {
            layer_from,
            from,
            to,
            assignment,
            next_parent,
            next_child,
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

    pub fn next_parent(&self) -> Option<EdgeIndex> {
        self.next_parent
    }

    pub fn next_child(&self) -> Option<EdgeIndex> {
        self.next_child
    }
}
