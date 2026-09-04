use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq, Hash, deepsize::DeepSizeOf)]
pub enum Bitset {
    Inline(u64),
    Heap(Vec<u64>),
}

impl Bitset {
    pub fn new(n: usize) -> Self {
        let number_words = (n / 64) + 1;
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

    pub fn contains(&self, element: usize) -> bool {
        let word = element / 64;
        let shift = element % 64;
        self.words()[word] & (1 << shift) != 0
    }

    pub fn insert(&mut self, element: usize) {
        let word = element / 64;
        let shift = element % 64;
        self.words_mut()[word] |= 1 << shift;
    }

    pub fn _remove(&mut self, element: usize) {
        let word = element / 64;
        let shift = element % 64;
        self.words_mut()[word] &= !(1 << shift);
    }

    pub fn size(&self) -> usize {
        self.words().iter().map(|word| word.count_ones()).sum::<u32>() as usize
    }

    pub fn size_union(&self, other: &Bitset) -> usize {
        let other_words = other.words();
        self.words()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, word)| (word | other_words[i]).count_ones())
            .sum::<u32>() as usize
    }

    pub fn union(&mut self, other: &Bitset) {
        let other_words = other.words();
        let words = self.words_mut();
        debug_assert!(words.len() == other_words.len());
        for word in 0..words.len() {
            words[word] |= other_words[word]
        }
    }

    pub fn union_with_and_bit(&mut self, other: &Bitset, bit: usize) {
        let bit_word = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        let other_words = other.words();
        for (i, (word, other_word)) in self.words_mut().iter_mut().zip(other_words.iter()).enumerate() {
            let extra = if i == bit_word { bit_mask } else { 0 };
            *word |= other_word | extra;
        }
    }

    pub fn intersect(&mut self, other: &Bitset) {
        let other_words = other.words();
        let words = self.words_mut();
        debug_assert!(words.len() == other_words.len());
        for word in 0..words.len() {
            words[word] &= other_words[word]
        }
    }

    pub fn intersect_with_and_bit(&mut self, other: &Bitset, bit: usize) {
        let bit_word = bit / 64;
        let bit_mask = 1u64 << (bit % 64);
        let other_words = other.words();
        for (i, (word, other_word)) in self.words_mut().iter_mut().zip(other_words.iter()).enumerate() {
            let extra = if i == bit_word { bit_mask } else { 0 };
            *word &= other_word | extra;
        }
    }

    pub fn reset(&mut self, value: u64) {
        for word in self.words_mut().iter_mut() {
            *word = value;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> {
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
    pub fn test_new_is_inline_for_n_63() {
        let b = Bitset::new(63);
        assert!(matches!(b, Bitset::Inline(_)));
    }

    #[test]
    pub fn test_new_is_heap_for_n_64() {
        let b = Bitset::new(64);
        assert!(matches!(b, Bitset::Heap(_)));
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
    pub fn test_remove_inline() {
        let mut b = Bitset::new(8);
        b.insert(2);
        b.insert(4);
        b._remove(2);
        assert!(!b.contains(2));
        assert!(b.contains(4));
    }

    #[test]
    pub fn test_remove_heap() {
        let mut b = Bitset::new(200);
        b.insert(70);
        b.insert(140);
        b._remove(70);
        assert!(!b.contains(70));
        assert!(b.contains(140));
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
