#[derive(Clone, Copy, Debug)]
pub enum MergeHeuristic {
    LessRelaxed,
    MostLikely,
    StateSimilarity,
}

impl MergeHeuristic {
    pub(crate) fn bucket_merge(&self) -> bool {
        match self {
            Self::LessRelaxed => false,
            Self::MostLikely => false,
            Self::StateSimilarity => true,
        }
    }
}
