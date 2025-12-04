use super::*;

pub struct Edge {
    layer_from: LayerIndex,
    from: NodeIndex,
    to: NodeIndex,
    assignment: isize,
    next_parent: Option<EdgeIndex>,
    prev_parent: Option<EdgeIndex>,
    next_child: Option<EdgeIndex>,
    prev_child: Option<EdgeIndex>,
    active: bool,
}

impl Edge {
    
    pub fn new(layer_from: LayerIndex,
        from: NodeIndex,
        to: NodeIndex,
        assignment: isize,
        next_parent: Option<EdgeIndex>,
        next_child: Option<EdgeIndex>,
        ) -> Self {
        Self {
            layer_from,
            from,
            to,
            assignment,
            next_parent,
            prev_parent: None,
            next_child,
            prev_child: None,
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

    pub fn next_parent(&self) -> Option<EdgeIndex> {
        self.next_parent
    }

    pub fn prev_parent(&self) -> Option<EdgeIndex> {
        self.prev_parent
    }

    pub fn next_child(&self) -> Option<EdgeIndex> {
        self.next_child
    }
    
    pub fn prev_child(&self) -> Option<EdgeIndex> {
        self.prev_child
    }

    pub fn set_prev_parent(&mut self, parent: Option<EdgeIndex>) {
        self.prev_parent = parent;
    }

    pub fn set_next_parent(&mut self, parent: Option<EdgeIndex>) {
        self.next_parent = parent;
    }

    pub fn set_prev_child(&mut self, child: Option<EdgeIndex>) {
        self.prev_child = child;
    }

    pub fn set_next_child(&mut self, child: Option<EdgeIndex>) {
        self.next_child = child;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}
