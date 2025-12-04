use super::*;

#[derive(Default)]
pub struct Node {
    layer: LayerIndex,
    index_in_layer: usize,
    first_parent: Option<EdgeIndex>,
    first_child: Option<EdgeIndex>,
    active: bool,
}

impl Node {

    pub fn new(layer: LayerIndex, index_in_layer: usize) -> Self {
        Self {
            layer,
            index_in_layer,
            first_parent: None,
            first_child: None,
            active: true,
        }
    }

    pub fn layer(&self) -> LayerIndex {
        self.layer
    }

    pub fn index_in_layer(&self) -> usize {
        self.index_in_layer
    }

    pub fn first_child(&self) -> Option<EdgeIndex> {
        self.first_child
    }

    pub fn set_first_child(&mut self, child: Option<EdgeIndex>) {
        self.first_child = child;
    }

    pub fn first_parent(&self) -> Option<EdgeIndex> {
        self.first_parent
    }

    pub fn set_first_parent(&mut self, parent: Option<EdgeIndex>) {
        self.first_parent = parent;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}
