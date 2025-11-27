use super::Constraint;
use crate::core::{ValueIndex, VariableIndex};
use crate::core::problem::Problem;
use crate::mdd::{Mdd, NodeIndex, LayerIndex};
use rustc_hash::FxHashMap;

/// Structure for the allDifferent constraint.
///
/// References:
///     - Hoda, S., Van Hoeve, W. J., & Hooker, J. N. (2010, September). A systematic approach to MDD-based constraint programming. CP2010

pub struct AllDifferent {
    variables: Vec<VariableIndex>,
    map_bit: FxHashMap<isize, usize>,
}

impl AllDifferent {

    pub fn new(problem: &Problem, variables: Vec<VariableIndex>) -> Self {
        let mut map_bit = FxHashMap::<isize, usize>::default();
        let mut bit = 0;
        for variable in variables.iter().copied() {
            for value in problem[variable].iter_domain() {
                if !map_bit.contains_key(&value) {
                    map_bit.insert(value, bit);
                    bit += 1;
                }
            }
        }
        Self {
            variables,
            map_bit,
        }
    }
}

impl Constraint for AllDifferent {

    /// The properties of a node s for the all different consists of two bitsets:
    ///     1. A(s): the values appearing on *all* path going to/from s
    ///     2. S(s): the values appearing on *some* path going to/from s
    /// We represent these two sets in a single bitset for convenience. Let $D$ be the union of the
    /// domain of the variables in the scope of the function, then we have two bitsets of |D|/64
    /// words, or a single bitset of 2|D|/64 words.
    fn get_properties(&self) -> Vec<u64> {
        let nb_words = self.map_bit.len() / 64;
        vec![0; 2*(nb_words + 1)]
    }

    /// Update the top-down property of a given node. This implies two operations:
    ///     1. Aggregating the local property of all parents. This is done using intersection for
    ///        the A bitset and union for the S bitset
    ///     2. Incorporating the edges' values to the local properties of the parents before the
    ///        integrations. We do this as union of {value} with the A and S bitsets.
    ///
    /// Since the aggregation operations are comutative, we can do this update edge by edge.
    fn update_property_top_down(&self, mdd: &mut Mdd, layer: LayerIndex, node: NodeIndex, constraint_index: usize) {
        let mut edge_ptr = mdd[layer][node].first_parent();
        while let Some(edge) = edge_ptr {
            let parent = mdd[layer][edge].from();
            debug_assert!(mdd[layer][parent].top_down_properties(constraint_index).len() == mdd[layer][node].top_down_properties(constraint_index).len());
            debug_assert!(mdd[layer][parent].top_down_properties(constraint_index).len() % 2 == 0);

            let assignment = mdd[layer][edge].value();
            debug_assert!(self.map_bit.contains_key(&assignment));
            
            // First, we compute the position of the assignment in the bitvectors
            let bit_position = *self.map_bit.get(&assignment).unwrap();
            let value_word = bit_position / 64;
            let shift = bit_position % 64;
            // Step 1: integrates the assignment into the bitset of the node
            let nb_words = mdd[layer][parent].top_down_properties(constraint_index).len() / 2;

            let (to_rev_a, to_rev_s) = (mdd[layer][parent].top_down_properties(constraint_index)[value_word], mdd[layer][parent].top_down_properties(constraint_index)[nb_words + value_word]);
            mdd[layer][parent].top_down_properties_mut(constraint_index)[value_word] |= 1 << shift;
            mdd[layer][parent].top_down_properties_mut(constraint_index)[nb_words + value_word] |= 1 << shift;

            for word in 0..nb_words {
                let parent_word = mdd[layer][parent].top_down_properties(constraint_index)[word];
                mdd[layer][node].top_down_properties_mut(constraint_index)[word] &= parent_word;
                let parent_word = mdd[layer][parent].top_down_properties(constraint_index)[nb_words + word];
                mdd[layer][node].top_down_properties_mut(constraint_index)[nb_words + word] |= parent_word;
            }

            mdd[layer][parent].top_down_properties_mut(constraint_index)[value_word] = to_rev_a;
            mdd[layer][parent].top_down_properties_mut(constraint_index)[nb_words + value_word] = to_rev_s;
            edge_ptr = mdd[layer][edge].next_parent();
        }
    }

    fn update_property_bottom_up(&self, mdd: &mut Mdd, layer: LayerIndex, node: NodeIndex, constraint_index: usize) {
    }

    fn iter_scope(&self) -> Vec<VariableIndex> {
        self.variables.clone()
    }
}
