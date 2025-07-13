use vector::{Index, Vector};

pub struct AppoxNearestNeighor {
    pub index: Index<512>,
    pub vectors: Vec<Vector<512>>, // D = dimensions
    pub metadata: Vec<Metadata>,
}

impl AppoxNearestNeighor {
    pub fn build(vectors: &[Vector<512>], metadata: &[Metadata]) -> Self {
        assert_eq!(
            vectors.len(),
            metadata.len(),
            "vectors and metadata must have same length"
        );
        let index = Index::build(vectors, 1, 1, 42);
        Self {
            index,
            vectors: vectors.to_vec(),
            metadata: metadata.to_vec(),
        }
    }
    pub fn query<'a>(
        &'a self,
        query: &Vector<512>,
        k: i32,
    ) -> Vec<ApproxNearNeighResult<'a, Metadata>> {
        self.index
            .search(&self.vectors, query, k as usize)
            .into_iter()
            .map(|(idx, dist)| ApproxNearNeighResult::new(&self.metadata[idx], dist))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Metadata {
    pub file: String,
    pub code: String,
}

#[derive(Clone, Debug)]
pub struct ApproxNearNeighResult<'a, Metadata> {
    pub metadata: &'a Metadata,
    pub distance: f32,
}

impl<'a, M> ApproxNearNeighResult<'a, M> {
    pub fn new(metadata: &'a M, distance: f32) -> Self {
        Self { metadata, distance }
    }
}
