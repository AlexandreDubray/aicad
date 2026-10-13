//! A fixed-capacity set of small integers, stored one bit per element.
//!
//! # Meaning
//!
//! `Bitset::new(n)` holds the elements $0, \dots, n - 1$; it starts empty (all bits to 0).
//! Membership and insertion address a single element, the set operations work word by word:
//! [`union`](Bitset::union) and [`intersect`](Bitset::intersect) update `self` in place,
//! [`size_union`](Bitset::size_union) counts $|A \cup B|$ without building the union, and the
//! `*_with_and_bit` variants combine with `other` extended by one extra element. There is no
//! removal.
//!
//! Both operands of a set operation must have been created with the same `n`. Using an element
//! of $n$ or more, or combining sets of different sizes, results in a runtime error. The storage
//! is made of whole 64-bit words, so an element below the next multiple of 64 is not rejected,
//! and one beyond it panics.
//!
//! [`reset`](Bitset::reset) sets every *word* to the given value, not every element: `reset(0)`
//! empties the set and `reset(!0)` sets all bits of the last word too, including those at $n$ and
//! above.
//!
//! # How it works
//!
//! Element $e$ lives in bit $e \bmod 64$ of word $\lfloor e / 64 \rfloor$. Sizes up to 64
//! fit in one word, kept inline in the enum; larger ones use a vector of words. Both are viewed as
//! a slice of words, so every operation is the same loop over words whatever the representation.
//!
//! # Design notes
//!
//! - **One word is stored inline.** Constraints keep a bitset in the property of every node of
//!   the diagram, and they are often small (the values of a domain, the states of an automaton),
//!   so avoiding a heap allocation per node matters more than a uniform representation.
//! - **Equality and hashing look at the words.** Two bitsets are equal exactly when they have the
//!   same storage class and the same words, which is what lets properties be compared and used as
//!   merge keys.

use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq, Hash, deepsize::DeepSizeOf)]
pub(crate) enum Bitset {
    Inline(u64),
    Heap(Vec<u64>),
}

impl Bitset {
    pub(crate) fn new(n: usize) -> Self {
        let number_words = n.div_ceil(64).max(1);
        if number_words == 1 {
            Self::Inline(0)
        } else {
            Self::Heap(vec![0; number_words])
        }
    }

    fn words(&self) -> &[u64] {
        match self {
            Bitset::Inline(w) => std::slice::from_ref(w),
            Bitset::Heap(v) => v.as_slice(),
        }
    }

    fn words_mut(&mut self) -> &mut [u64] {
        match self {
            Bitset::Inline(w) => std::slice::from_mut(w),
            Bitset::Heap(v) => v.as_mut_slice(),
        }
    }

    pub(crate) fn contains(&self, element: usize) -> bool {
        let word = element / 64;
        let shift = element % 64;
        self.words()[word] & (1 << shift) != 0
    }

    pub(crate) fn insert(&mut self, element: usize) {
        let word = element / 64;
        let shift = element % 64;
        self.words_mut()[word] |= 1 << shift;
    }

    pub(crate) fn size(&self) -> usize {
        self.words()
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>() as usize
    }

    pub(crate) fn size_union(&self, other: &Bitset) -> usize {
        let other_words = other.words();
        self.words()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, word)| (word | other_words[i]).count_ones())
            .sum::<u32>() as usize
    }

    pub(crate) fn union(&mut self, other: &Bitset) {
        let other_words = other.words();
        let words = self.words_mut();
        debug_assert!(words.len() == other_words.len());
        for word in 0..words.len() {
            words[word] |= other_words[word]
        }
    }

    pub(crate) fn union_with_and_bit(&mut self, other: &Bitset, bit: usize) {
        let bit_word = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        let other_words = other.words();
        for (i, (word, other_word)) in self
            .words_mut()
            .iter_mut()
            .zip(other_words.iter())
            .enumerate()
        {
            let extra = if i == bit_word { bit_mask } else { 0 };
            *word |= other_word | extra;
        }
    }

    pub(crate) fn intersect(&mut self, other: &Bitset) {
        let other_words = other.words();
        let words = self.words_mut();
        debug_assert!(words.len() == other_words.len());
        for word in 0..words.len() {
            words[word] &= other_words[word]
        }
    }

    pub(crate) fn intersect_with_and_bit(&mut self, other: &Bitset, bit: usize) {
        let bit_word = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        let other_words = other.words();
        for (i, (word, other_word)) in self
            .words_mut()
            .iter_mut()
            .zip(other_words.iter())
            .enumerate()
        {
            let extra = if i == bit_word { bit_mask } else { 0 };
            *word &= other_word | extra;
        }
    }

    pub(crate) fn reset(&mut self, value: u64) {
        for word in self.words_mut().iter_mut() {
            *word = value;
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = u64> {
        self.words().iter().copied()
    }
}

impl std::fmt::Display for Bitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for word in self.words().iter() {
            write!(f, " {:b}", word)?;
        }
        write!(f, "")
    }
}

#[cfg(test)]
mod test_bitset {

    use super::Bitset;

    #[test]
    pub fn test_new_is_inline_for_small_n() {
        let b = Bitset::new(8);
        assert!(matches!(b, Bitset::Inline(_)));
    }

    #[test]
    pub fn test_new_is_inline_for_n_64() {
        let b = Bitset::new(64);
        assert!(matches!(b, Bitset::Inline(_)));
    }

    #[test]
    pub fn test_new_is_heap_for_n_65() {
        let b = Bitset::new(65);
        assert!(matches!(b, Bitset::Heap(_)));
    }

    #[test]
    pub fn test_new_with_zero_elements_is_inline() {
        let b = Bitset::new(0);
        assert!(matches!(b, Bitset::Inline(_)));
    }

    #[test]
    pub fn test_the_last_element_of_a_full_word_and_of_the_next_one_fit() {
        let mut inline = Bitset::new(64);
        inline.insert(63);
        assert!(inline.contains(63));
        let mut heap = Bitset::new(65);
        heap.insert(64);
        assert!(heap.contains(64));
    }

    #[test]
    pub fn test_new_is_heap_for_large_n() {
        let b = Bitset::new(200);
        assert!(matches!(b, Bitset::Heap(_)));
    }

    #[test]
    pub fn test_insert_and_contains_inline() {
        let mut b = Bitset::new(8);
        assert!(!b.contains(3));
        b.insert(3);
        assert!(b.contains(3));
        assert!(!b.contains(4));
    }

    #[test]
    pub fn test_insert_and_contains_heap_within_first_word() {
        let mut b = Bitset::new(200);
        b.insert(5);
        assert!(b.contains(5));
        assert!(!b.contains(6));
    }

    #[test]
    pub fn test_insert_and_contains_heap_crosses_word_boundary() {
        let mut b = Bitset::new(200);
        b.insert(64);
        b.insert(130);
        assert!(b.contains(64));
        assert!(b.contains(130));
        assert!(!b.contains(63));
        assert!(!b.contains(65));
    }

    #[test]
    pub fn test_size_inline() {
        let mut b = Bitset::new(8);
        assert_eq!(b.size(), 0);
        b.insert(0);
        b.insert(5);
        b.insert(7);
        assert_eq!(b.size(), 3);
    }

    #[test]
    pub fn test_size_heap() {
        let mut b = Bitset::new(200);
        b.insert(1);
        b.insert(64);
        b.insert(150);
        assert_eq!(b.size(), 3);
    }

    #[test]
    pub fn test_size_union_inline() {
        let mut a = Bitset::new(8);
        let mut b = Bitset::new(8);
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        assert_eq!(a.size_union(&b), 3);
    }

    #[test]
    pub fn test_size_union_heap() {
        let mut a = Bitset::new(200);
        let mut b = Bitset::new(200);
        a.insert(1);
        a.insert(150);
        b.insert(150);
        b.insert(160);
        assert_eq!(a.size_union(&b), 3);
    }

    #[test]
    pub fn test_union_inline() {
        let mut a = Bitset::new(8);
        let mut b = Bitset::new(8);
        a.insert(1);
        b.insert(2);
        a.union(&b);
        assert!(a.contains(1));
        assert!(a.contains(2));
        assert_eq!(a.size(), 2);
    }

    #[test]
    pub fn test_union_heap() {
        let mut a = Bitset::new(200);
        let mut b = Bitset::new(200);
        a.insert(10);
        b.insert(140);
        a.union(&b);
        assert!(a.contains(10));
        assert!(a.contains(140));
        assert_eq!(a.size(), 2);
    }

    #[test]
    pub fn test_intersect_inline() {
        let mut a = Bitset::new(8);
        let mut b = Bitset::new(8);
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        a.intersect(&b);
        assert!(!a.contains(1));
        assert!(a.contains(2));
        assert!(!a.contains(3));
        assert_eq!(a.size(), 1);
    }

    #[test]
    pub fn test_intersect_heap() {
        let mut a = Bitset::new(200);
        let mut b = Bitset::new(200);
        a.insert(10);
        a.insert(140);
        b.insert(140);
        b.insert(190);
        a.intersect(&b);
        assert!(!a.contains(10));
        assert!(a.contains(140));
        assert!(!a.contains(190));
    }

    #[test]
    pub fn test_union_with_and_bit_inline() {
        let mut a = Bitset::new(8);
        let b = Bitset::new(8);
        a.union_with_and_bit(&b, 5);
        assert!(a.contains(5));
        assert_eq!(a.size(), 1);
    }

    #[test]
    pub fn test_union_with_and_bit_heap() {
        let mut a = Bitset::new(200);
        let mut b = Bitset::new(200);
        b.insert(10);
        a.union_with_and_bit(&b, 140);
        assert!(a.contains(10));
        assert!(a.contains(140));
        assert_eq!(a.size(), 2);
    }

    #[test]
    pub fn test_intersect_with_and_bit_inline() {
        let mut a = Bitset::new(8);
        a.insert(3);
        a.insert(5);
        let b = Bitset::new(8);
        a.intersect_with_and_bit(&b, 5);
        assert!(!a.contains(3));
        assert!(a.contains(5));
    }

    #[test]
    pub fn test_intersect_with_and_bit_heap() {
        let mut a = Bitset::new(200);
        a.insert(10);
        a.insert(140);
        let b = Bitset::new(200);
        a.intersect_with_and_bit(&b, 140);
        assert!(!a.contains(10));
        assert!(a.contains(140));
    }

    #[test]
    pub fn test_reset_inline() {
        let mut b = Bitset::new(8);
        b.insert(3);
        b.reset(0);
        assert!(!b.contains(3));
        assert_eq!(b.size(), 0);
    }

    #[test]
    pub fn test_reset_heap() {
        let mut b = Bitset::new(200);
        b.insert(10);
        b.insert(140);
        b.reset(0);
        assert_eq!(b.size(), 0);
    }

    #[test]
    pub fn test_iter_word_count_inline() {
        let b = Bitset::new(8);
        assert_eq!(b.iter().count(), 1);
    }

    #[test]
    pub fn test_iter_word_count_heap() {
        let b = Bitset::new(200);
        assert_eq!(b.iter().count(), 4);
    }

    #[test]
    pub fn test_clone_and_eq() {
        let mut a = Bitset::new(200);
        a.insert(10);
        a.insert(140);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    pub fn test_display_does_not_panic() {
        let mut b = Bitset::new(200);
        b.insert(10);
        let _ = format!("{}", b);
    }
}
