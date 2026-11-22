//! Creates a sparse, metadata-only GGUF file for demonstrations.
//!
//! The output is not a runnable language model. Its apparent size is used only to demonstrate
//! DivideIT's planner without downloading or redistributing model weights.

use std::env;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("work/divideit-demo.gguf"));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(&output)?;
    file.write_all(b"GGUF")?;
    file.write_all(&3_u32.to_le_bytes())?;
    file.write_all(&291_u64.to_le_bytes())?;
    file.write_all(&8_u64.to_le_bytes())?;

    text_metadata(&mut file, "general.name", "DivideIT Demo 7B")?;
    text_metadata(&mut file, "general.architecture", "llama")?;
    u32_metadata(&mut file, "general.file_type", 15)?;
    u32_metadata(&mut file, "llama.block_count", 32)?;
    u32_metadata(&mut file, "llama.context_length", 8192)?;
    u32_metadata(&mut file, "llama.embedding_length", 4096)?;
    u32_metadata(&mut file, "llama.attention.head_count", 32)?;
    u32_metadata(&mut file, "llama.attention.head_count_kv", 8)?;

    // Sparse extension: realistic apparent size, without any copyrighted model weights.
    file.seek(SeekFrom::Start(4_500_000_000 - 1))?;
    file.write_all(&[0])?;
    println!("Created demo metadata file: {}", output.display());
    println!("This is not a runnable model and should not be committed.");
    Ok(())
}

fn string(file: &mut File, value: &str) -> std::io::Result<()> {
    file.write_all(&(value.len() as u64).to_le_bytes())?;
    file.write_all(value.as_bytes())
}

fn text_metadata(file: &mut File, key: &str, value: &str) -> std::io::Result<()> {
    string(file, key)?;
    file.write_all(&8_u32.to_le_bytes())?;
    string(file, value)
}

fn u32_metadata(file: &mut File, key: &str, value: u32) -> std::io::Result<()> {
    string(file, key)?;
    file.write_all(&4_u32.to_le_bytes())?;
    file.write_all(&value.to_le_bytes())
}
