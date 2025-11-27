use super::mdd::EdgeIndex;

#[derive(Default)]
pub struct Node {
    first_parent: Option<EdgeIndex>,
    first_child: Option<EdgeIndex>,
    top_down_properties: Vec<Vec<u64>>,
    bottom_up_properties: Vec<Vec<u64>>,
}

impl Node {

    /// Returns a reference to the top-down local properties of a constraint
    pub fn top_down_properties(&self, constraint: usize) -> &Vec<u64> {
        &self.top_down_properties[constraint]
    }

    /// Returns a mutable reference to the top-down local properties of a constraint
    pub fn top_down_properties_mut(&mut self, constraint: usize) -> &mut Vec<u64> {
        &mut self.top_down_properties[constraint]
    }

    pub fn add_top_down_property(&mut self, property: Vec<u64>) {
        self.top_down_properties.push(property);
    }

    /// Returns a reference to the bottom-up local properties of a constraint
    pub fn bottom_up_properties(&self, constraint: usize) -> &Vec<u64> {
        &self.bottom_up_properties[constraint]
    }

    /// Returns a mutable reference to the bottom-up local properties of a constraint
    pub fn bottom_up_properties_mut(&mut self, constraint: usize) -> &mut Vec<u64> {
        &mut self.bottom_up_properties[constraint]
    }

    pub fn add_bottom_up_property(&mut self, property: Vec<u64>) {
        self.bottom_up_properties.push(property);
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
}
