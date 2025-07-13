use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;

pub struct MyEmbedder {
    inner: EAEmbedder,
}

const DEFAULT_MODEL_ID: &str = "jinaai/jina-embeddings-v2-small-en";
impl MyEmbedder {
    pub fn new(
    ) -> Result<Self> {
        
     
        let jina_embedder = JinaEmbedder::new(DEFAULT_MODEL_ID, None, None).map_err(|e| 
            anyhow::anyhow!("Failed to load JinaEmbedder model '{}': {}. Ensure the model exists and network is available if downloading.", DEFAULT_MODEL_ID, e)
        )?;
        let inner = EAEmbedder::Text(TextEmbedder::Jina(Box::new(jina_embedder)));
        Ok(Self { inner })
    }

    pub async fn embed(&self, text: &str) -> Result<[f32; 512]> {
        let mut results = self.embed_batch(&[text], Some(1)).await?;
        if results.is_empty() {
            Err(anyhow::anyhow!("Embedding failed for text"))
        } else {
            Ok(results.remove(0))
        }
    }

    pub async fn embed_batch(&self, texts: &[&str], batch_size: Option<usize>) -> Result<Vec<[f32; 512]>> {
        let results = self.inner.embed(texts, batch_size, None).await?;
        let mut embeddings_array_vec = Vec::with_capacity(results.len());

        for embedding_result in results {
            let vector = embedding_result.to_dense()?; // vector is Vec<f32>
            if vector.len() != 512 { 
                
                return Err(anyhow::anyhow!(
                    "Embedding size mismatch: expected 512 for Jina v2, got {}",
                    vector.len()
                ));
            }
            let mut arr = [0.0_f32; 512];
            arr.copy_from_slice(&vector);
            embeddings_array_vec.push(arr);
        }
        Ok(embeddings_array_vec)
    }
}
