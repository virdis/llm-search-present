use std::sync::Arc;

use anyhow::Result;
use approx_nearest_neigh::{AppoxNearestNeighor, Metadata};
use code_splitter::{CharCounter, Splitter};
use futures::future::join_all;
use futures_util::StreamExt;
use hypothetical_doc::HypotheticalDoc;
use ignore::WalkBuilder;
use myembedder::MyEmbedder;

mod approx_nearest_neigh;
mod hypothetical_doc;
mod myembedder;

const GO_LANG_SRC: &str = "./data";

pub fn chunk_repo() -> Result<Vec<(String, String)>> {
    println!("Starting chunking for repo: {}", GO_LANG_SRC);
    let lang = tree_sitter_go::language();
    let splitter = Splitter::new(lang, CharCounter)
        .expect("Failed to load tree-sitter language")
        .with_max_size(1000);
    let mut out = Vec::new();
    let mut total_files = 0;
    let mut skipped_files = 0;
    let mut total_chunks = 0;
    for entry in WalkBuilder::new(GO_LANG_SRC)
        .standard_filters(true)
        .build()
        .filter_map(Result::ok)
    {
        let path = entry.path();

        if !path.is_file() {
            continue;
        }
        // TODO: Update this filter when multi-language support is added.
        if !path.extension().map_or(false, |e| e == "go") {
            println!("Skipping non golang source file: {}", path.display());
            skipped_files += 1;
            continue;
        }
        total_files += 1;
        print!("Processing file: {}", path.display());
        let code = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                println!("Failed to read file {}: {}", path.display(), e);
                skipped_files += 1;
                continue;
            }
        };
        let code_bytes = code.as_bytes();
        let chunks = match splitter.split(code_bytes) {
            Ok(c) => c,
            Err(e) => {
                println!("Failed to split file {}: {}", path.display(), e);
                skipped_files += 1;
                continue;
            }
        };
        let mut file_chunk_count = 0;
        for chunk in chunks {
            let start = chunk.range.start_byte;
            let end = chunk.range.end_byte;
            let snippet = match std::str::from_utf8(&code_bytes[start..end]) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    println!("Invalid UTF-8 in file {}: {}", path.display(), e);
                    continue;
                }
            };
            out.push((path.display().to_string(), snippet));
            file_chunk_count += 1;
        }
        total_chunks += file_chunk_count;
        println!(
            "File {}: {} chunks generated",
            path.display(),
            file_chunk_count
        );
    }
    println!(
        "Chunking complete. Processed {} Golang files, skipped {} files. Total chunks: {}.",
        total_files, skipped_files, total_chunks
    );
    Ok(out)
}

async fn build_index(embedder: &myembedder::MyEmbedder) -> Result<AppoxNearestNeighor> {
    let chunks = chunk_repo();

    let mut vecs = Vec::new();
    let mut metas = Vec::new();

    let app_batch_size = 32;

    let mut processing_futures = Vec::new();

    for (batch_idx, chunk_batch_slice) in chunks?.chunks(app_batch_size).enumerate() {
        let owned_chunk_batch: Vec<(String, String)> = chunk_batch_slice.to_vec();

        let future = async move {
            let texts_to_embed: Vec<String> = owned_chunk_batch
                .iter()
                .map(|(_, code_snippet)| code_snippet.clone())
                .collect();

            let result = if texts_to_embed.is_empty() {
                Ok((owned_chunk_batch, Vec::new()))
            } else {
                let text_slices: Vec<&str> = texts_to_embed.iter().map(AsRef::as_ref).collect();
                match embedder
                    .embed_batch(&text_slices, Some(text_slices.len()))
                    .await
                {
                    Ok(embeddings) => Ok((owned_chunk_batch, embeddings)),
                    Err(e) => Err(anyhow::anyhow!("Error in batch {}: {}", batch_idx, e)),
                }
            };

            result
        };
        processing_futures.push(future);
    }

    let all_results = join_all(processing_futures).await;

    for result_item in all_results {
        match result_item {
            Ok((original_chunk_batch, embedding_arrays)) => {
                if !embedding_arrays.is_empty() {
                    for (embedding_idx, embedding_array) in embedding_arrays.iter().enumerate() {
                        let (file_path, code_snippet) = &original_chunk_batch[embedding_idx];
                        vecs.push(vector::Vector::<512>::from(*embedding_array));
                        metas.push(Metadata {
                            file: file_path.clone(),
                            code: code_snippet.clone(),
                        });
                    }
                }
            }
            Err(e) => {
                println!("Failed to process a batch of embeddings: {}", e);
                // It's important to decide error strategy: continue with partial embeddings or fail hard.
                // For now, propagating the first error encountered.
                return Err(e.into());
            }
        }
    }

    println!(
        "Embeddings generation phase complete. {} embeddings collected.",
        vecs.len()
    );

    if vecs.is_empty() {
        println!("No embeddings were generated. Index will be empty.");
    }

    let ann_instance: AppoxNearestNeighor =
        approx_nearest_neigh::AppoxNearestNeighor::build(&vecs, &metas);

    Ok(ann_instance)
}

// Skip re-ranking
async fn search_docs(
    embedder: MyEmbedder,
    ann_index: &AppoxNearestNeighor,
    query_string: &str,
    k: usize,
) -> Result<()> {
    let hyde = HypotheticalDoc::new(embedder, ann_index, 1000);

    println!("Start searching docs...");

    let start_time = std::time::Instant::now();

    println!(
        "Retrieving results for query: '{}' with k={}",
        query_string, k
    );

    let hits_result = hyde.retrieve(query_string, k).await;

    let duration = start_time.elapsed();

    match hits_result {
        Ok(hits) => {
            println!("Answer generated in {:.2?}:", duration);
            println!("Answer Details \n {:?}", hits.answer_stream);
            Ok(())
        }
        Err(e) => {
            eprintln!("Error retrieving results: {}", e);
            Err(e)
        }
    }
}

// Instead of hard coding take arguments from the stdio

#[tokio::main]
async fn main() -> Result<()> {
    let embedder = MyEmbedder::new().unwrap();
    let approx_nearest_neigh = build_index(&embedder).await.unwrap();
    let query_string = "Find code that adds two numbers";
    let res = search_docs(embedder, &approx_nearest_neigh, query_string, 1000)
        .await
        .unwrap();
    Ok(())
}
