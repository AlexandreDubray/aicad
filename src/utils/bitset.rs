use rustc_hash::FxHashMap;
use std::hash::{Hasher, Hash};

pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {

    /// Creates a new bitset for storing n contiguous integers (starting from 0)
    pub fn new(n: usize) -> Self {
        //debug_assert!(n > 0);
        let number_words = (n / 64).max(1);
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

}

impl Hash for Bitset {
    fn hash<T: Hasher>(&self, state: &mut T) {
        self.words.hash(state);
    }
}

pub struct SparseBitset<T: Eq + Hash + Copy> {
    plain: Bitset,
    map: FxHashMap<T, usize>,
}

impl<T: Eq + Hash + Copy> SparseBitset<T> {

    pub fn new(elements: impl Iterator<Item = T>) -> Self {
        let mut map = FxHashMap::<T, usize>::default();
        for (bit, element) in elements.enumerate() {
            map.insert(element, bit);
        }
        Self {
            plain: Bitset::new(map.len()),
            map,
        }
    }


    pub fn contains(&self, element: T) -> bool {
        let element = *self.map.get(&element).unwrap();
        self.plain.contains(element)
    }

    pub fn insert(&mut self, element: T) {
        let element = *self.map.get(&element).unwrap();
        self.plain.insert(element);
    }

    pub fn remove(&mut self, element: T) {
        let element = *self.map.get(&element).unwrap();
        self.plain.remove(element);
    }

    pub fn size(&self) -> usize {
        self.plain.size()
    }

    pub fn size_union(&self, other: &SparseBitset<T>) -> usize {
        self.plain.size_union(&other.plain)
    }

    pub fn union(&mut self, other: &SparseBitset<T>) {
        self.plain.union(&other.plain);
    }

    pub fn interesect(&mut self, other: &SparseBitset<T>) {
        self.plain.intersect(&other.plain);
    }

    pub fn reset(&mut self, value: u64) {
        self.plain.reset(value);
    }
}

impl<T: Eq + Hash + Copy> Hash for SparseBitset<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.plain.hash(state);
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
impl<T: Eq + Hash + Copy> std::fmt::Display for SparseBitset<T> {

    fn fmt(&self, f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.plain)
    }
}
