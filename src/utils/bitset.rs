use std::hash::Hash;

#[derive(Clone, PartialEq, Eq, Hash, deepsize::DeepSizeOf)]
pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {

    /// Creates a new bitset for storing n contiguous integers (starting from 0)
    pub fn new(n: usize) -> Self {
        //debug_assert!(n > 0);
        let number_words = (n / 64) + 1;
        Self {
            words: vec![0; number_words],
        }
    }

    pub fn contains(&self, element: usize) -> bool {
        let word = element / 64;
        let shift = element % 64;
        self.words[word] & (1 << shift) != 0
    }

    pub fn insert(&mut self, element: usize) {
        let word = element / 64;
        let shift = element % 64;
        self.words[word] |= 1 << shift;
    }

    pub fn remove(&mut self, element: usize) {
        let word = element / 64;
        let shift = element % 64;
        self.words[word] &= !(1 << shift);
    }

    pub fn size(&self) -> usize {
        self.words.iter().map(|word| word.count_ones()).sum::<u32>() as usize
    }

    pub fn size_union(&self, other: &Bitset) -> usize {
        self.words.iter().copied().enumerate().map(|(i, word)| (word | other.words[i]).count_ones()).sum::<u32>() as usize
    }

    pub fn union(&mut self, other: &Bitset) {
        debug_assert!(self.words.len() == other.words.len());
        for word in 0..self.words.len() {
            self.words[word] |= other.words[word]
        }
    }

    pub fn intersect(&mut self, other: &Bitset) {
        debug_assert!(self.words.len() == other.words.len());
        for word in 0..self.words.len() {
            self.words[word] &= other.words[word]
        }
    }

    pub fn reset(&mut self, value: u64) {
        for word in 0..self.words.len() {
            self.words[word] = value;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> {
        self.words.iter().copied()
    }
}

impl std::fmt::Display for Bitset {

    fn fmt(&self, f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for word in self.words.iter() {
            write!(f, " {:b}", word)?;
        }
        write!(f, "")
    }
}
