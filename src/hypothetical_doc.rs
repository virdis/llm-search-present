use crate::approx_nearest_neigh::{self, Metadata};
use anyhow::anyhow;
use futures_util::{Stream, StreamExt};
use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, MessageRole, request::ChatMessageRequest},
};
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::{self, AsyncWriteExt};
use vector::Vector;

use crate::{approx_nearest_neigh::AppoxNearestNeighor, myembedder::MyEmbedder};

// HyDe
pub struct HypotheticalDoc<'a> {
    pub embedder: MyEmbedder,
    pub approx_near_neigh: &'a AppoxNearestNeighor,
    pub chunk_size: usize,
}

impl<'a> HypotheticalDoc<'a> {
    pub fn new(
        embedder: MyEmbedder,
        approx_nearest_neigh: &'a AppoxNearestNeighor,
        chunk_size: usize,
    ) -> Self {
        Self {
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

        let stream = self.explain_code_stream(&prompt, Some(system)).await?;
        let mut full_doc = String::new();
        full_doc = stream;
        if full_doc.is_empty() {
            Err(anyhow!(
                "Hypothetical document generation returned no content."
            ))
        } else {
            Ok(full_doc)
        }
    }

    // No re-ranking, but Jina model supports re-ranking
    pub async fn retrieve(&self, query: &str, k: usize) -> anyhow::Result<HypotheticalResponse> {
        let hypothetical_document = self.generate_hypothetical_document(query).await?;
        let results = self.similarity_search(&hypothetical_document, k).await?;
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
    ) -> anyhow::Result<String, anyhow::Error> {
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
    ) -> anyhow::Result<String, anyhow::Error> {
        let prompt = format!("Explain the following Golang code in detail:\n\n{}", code);
        let system = system_prompt
            .unwrap_or("You are a Golang expert. Explain the code clearly and concisely.");

        let system_message = ChatMessage::new(MessageRole::System, system.to_string());
        let user_message = ChatMessage::new(MessageRole::User, prompt.to_string());

        let chat_message_req = ChatMessageRequest::new(
            "qwen2.5-coder:32b".to_string(),
            vec![system_message, user_message],
        );
        // Default Setup
        let ollama = Ollama::new("http://localhost", 11434);

        let mut response = ollama
            .send_chat_messages_stream(chat_message_req)
            .await
            .unwrap();
        let mut answer: Vec<String> = vec![];
        let mut stdout = io::stdout();
        println!("Printing the streaming response to std::io");
        while let Some(res) = response.next().await {
            let response = res.unwrap();
            stdout.write_all(&response.message.content.as_bytes());
            stdout.flush();
            answer.push(response.message.content);
        }
        Ok(answer.join(""))
    }
}

#[derive(Debug, Clone)]
pub struct HypotheticalResult {
    pub index: usize,
    pub distance: f32,
    pub meta: Metadata,
}

pub struct HypotheticalResponse {
    pub answer_stream: String,
    pub code_refs: Vec<HypotheticalResult>,
}
