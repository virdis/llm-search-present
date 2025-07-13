use crate::{
    approx_nearest_neigh::{self, Metadata},
    rerank::Reranker,
};
use anyhow::anyhow;
use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use ollama_rs::Ollama;
use std::pin::Pin;
use std::sync::Arc;
use vector::Vector;

use crate::{approx_nearest_neigh::AppoxNearestNeighor, myembedder::MyEmbedder};

pub struct HypotheticalDoc<'a> {
    pub local_client: Ollama,
    pub embedder: Arc<MyEmbedder>,
    pub approx_near_neigh: &'a AppoxNearestNeighor,
    pub chunk_size: usize,
}

impl<'a> HypotheticalDoc<'a> {
    pub fn new(
        local_client: Ollama,
        embedder: Arc<MyEmbedder>,
        approx_nearest_neigh: &'a AppoxNearestNeighor,
        chunk_size: usize,
    ) -> Self {
        Self {
            local_client,
            embedder,
            approx_near_neigh: approx_nearest_neigh,
            chunk_size,
        }
    }

    pub async fn generate_hypothetical_document(&self, query: &str) -> anyhow::Result<String> {
        let prompt = format!(
            "Generate a hypothetical Golang code snippet or document that would answer the following query as if it existed in a codebase. The generated document must fit within {} characters.\n\nQuery: {}\n\nHypothetical Document:",
            self.chunk_size, query
        );
        let system = "You are a Golang code generator. Given a query, generate a Golang code snippet or document that would answer it. The output must not exceed the specified chunk size.";

        let mut stream = self.explain_code_stream(&prompt, Some(system)).await?;
        let mut full_doc = String::new();
        while let Some(chunk_result) = stream.next().await {
            full_doc.push_str(&chunk_result?);
        }
        if full_doc.is_empty() {
            Err(anyhow!(
                "Hypothetical document generation returned no content."
            ))
        } else {
            Ok(full_doc)
        }
    }

    // No re-ranking
    pub async fn retrieve(&self, query: &str, k: usize) -> anyhow::Result<HypotheticalResponse> {
        let hypothetical_document = self.generate_hypothetical_document(query).await?;
        let mut results = self.similarity_search(&hypothetical_document, k).await?;
        let answer_stream = self.synthesize_answer_stream(query, &results).await?;

        Ok(HypotheticalResponse {
            answer_stream,
            code_refs: results,
        })
    }

    pub async fn similarity_search(
        &self,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<HypotheticalResult>> {
        let embedding = self.embedder.embed(query).await?;
        let embedding_ref: &Vector<512> =
            unsafe { &*(&embedding as *const [f32; 512] as *const [f32; 512]) };
        let results = self.approx_near_neigh.query(embedding_ref, k as i32);
        let hyde_results = results
            .into_iter()
            .enumerate()
            .map(|(idx, res)| HypotheticalResult {
                index: idx,
                distance: res.distance,
                meta: res.metadata.clone(),
            })
            .collect();
        Ok(hyde_results)
    }

    pub async fn synthesize_answer_stream(
        &self,
        query: &str,
        code_refs: &[HypotheticalResult],
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>>
    {
        let context_snippets: Vec<String> = code_refs
            .iter()
            .map(|res| format!("File: {}\nCode:\n{}\n", res.meta.file, res.meta.code))
            .collect();
        let llm_prompt = format!(
            "Given the following user query:\n{}\n\nand these relevant code snippets:\n{}\n\nProvide a detailed answer, referencing the code where appropriate.",
            query,
            context_snippets.join("\n---\n")
        );
        self.explain_code_stream(&llm_prompt, None).await
    }

    pub async fn explain_code_stream(
        &self,
        code: &str,
        system_prompt: Option<&str>,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>>
    {
        let prompt = format!("Explain the following Golang code in detail:\n\n{}", code);
        let system = system_prompt
            .unwrap_or("You are a Golnag expert. Explain the code clearly and concisely.");

        //     curl http://localhost:8080/api/generate -d '{
        //     "model": "qwen2.5:7b",
        //     "prompt": "Write a golang function to add 2 numbers",
        //     "stream": false
        // }'
        //
        // let mut res: GenerationResponseStream = ollama
        //     .generate_stream(GenerationRequest::new("llama2:latest".to_string(), PROMPT))
        //     .await
        //     .unwrap();
        //
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct HypotheticalResult {
    pub index: usize,
    pub distance: f32,
    pub meta: Metadata,
}

pub struct HypotheticalResponse {
    pub answer_stream: Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>,
    pub code_refs: Vec<HypotheticalResult>,
}
