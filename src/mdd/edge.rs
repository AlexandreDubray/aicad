use super::*;

pub struct Edge {
    from: NodeIndex,
    to: NodeIndex,
    value: isize,
    next_parent: Option<EdgeIndex>,
    next_child: Option<EdgeIndex>,
}

impl Edge {
    
    pub fn new(from: NodeIndex, to: NodeIndex, value: isize, next_parent: Option<EdgeIndex>, next_child: Option<EdgeIndex>) -> Self {
        Self {
            from,
            to,
            value,
            next_parent,
            next_child,
        }
    }

    pub fn from(&self) -> NodeIndex {
        self.from
    }

    pub fn to(&self) -> NodeIndex {
        self.to
    }

    pub fn value(&self) -> isize {
        self.value
    }

    pub fn next_parent(&self) -> Option<EdgeIndex> {
        self.next_parent
    }

    pub fn next_child(&self) -> Option<EdgeIndex> {
        self.next_child
    }
}
