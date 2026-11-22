use divideit::gguf::{self, ModelInfo};
use divideit::planner::{self, Plan};
use divideit::size::{format_bytes, parse_size};
use divideit::system::{self, SystemInfo};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const LOGO: &str = r#"
  ____  _       _     _      ___ _____
 |  _ \(_)_   _(_) __| | ___|_ _|_   _|
 | | | | \ \ / / |/ _` |/ _ \| |  | |
 | |_| | |\ V /| | (_| |  __/| |  | |
 |____/|_| \_/ |_|\__,_|\___|___| |_|
"#;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let command = arguments.next().unwrap_or_else(|| "help".into());
    match command.as_str() {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("divideit {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "inspect-system" => {
            reject_extra(arguments)?;
            let system = system::inspect()?;
            print_system(&system);
            Ok(())
        }
        "inspect-model" => {
            let path = required_path(arguments.next(), "inspect-model")?;
            reject_extra(arguments)?;
            let model = gguf::inspect(&path)?;
            print_model(&path, &model);
            Ok(())
        }
        "plan" | "analyze" => {
            let path = required_path(arguments.next(), "plan")?;
            let memory_limit = parse_plan_options(arguments)?;
            let model = gguf::inspect(&path)?;
            let system = system::inspect()?;
            let plan = planner::create_plan(&model, &system, memory_limit);
            print_plan(&path, &model, &system, &plan);
            Ok(())
        }
        "scan" => {
            let directory = required_path(arguments.next(), "scan")?;
            let memory_limit = parse_plan_options(arguments)?;
            let system = system::inspect()?;
            let models = scan_models(&directory)?;
            print_scan(&directory, &models, &system, memory_limit);
            Ok(())
        }
        unknown => Err(format!(
            "unknown command '{unknown}'. Run 'divideit help' to see available commands."
        )),
    }
}

fn required_path(value: Option<String>, command: &str) -> Result<PathBuf, String> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("{command} requires a path to a GGUF model"))
}

fn reject_extra(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    if let Some(extra) = arguments.next() {
        Err(format!("unexpected argument: {extra}"))
    } else {
        Ok(())
    }
}

fn parse_plan_options(mut arguments: impl Iterator<Item = String>) -> Result<Option<u64>, String> {
    let mut memory_limit = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--memory-limit" | "-m" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--memory-limit requires a value such as 8GB".to_string())?;
                memory_limit = Some(parse_size(&value).map_err(|error| error.to_string())?);
            }
            unknown => return Err(format!("unexpected option: {unknown}")),
        }
    }
    Ok(memory_limit)
}

fn print_help() {
    println!("{LOGO}");
    println!("Run larger local LLMs on smaller machines.\n");
    println!("USAGE:");
    println!("  divideit <COMMAND> [OPTIONS]\n");
    println!("COMMANDS:");
    println!("  inspect-system             Show detected RAM, CPU, and GPU resources");
    println!("  inspect-model <MODEL>      Read GGUF metadata without loading tensors");
    println!("  plan <MODEL> [OPTIONS]     Recommend memory-aware execution profiles");
    println!("  scan <DIRECTORY> [OPTIONS] Compare every GGUF model in a directory");
    println!("  version                    Print the DivideIT version");
    println!("  help                       Print this help\n");
    println!("PLAN OPTIONS:");
    println!("  -m, --memory-limit <SIZE>  Maximum host RAM, for example 8GB or 6144MB\n");
    println!("EXAMPLE:");
    println!("  divideit plan .\\models\\qwen2.5-7b-q4_k_m.gguf --memory-limit 8GB");
}

fn print_system(system: &SystemInfo) {
    println!("{LOGO}");
    println!("SYSTEM PROFILE\n");
    row("Operating system", &system.operating_system);
    row("Logical CPUs", &system.logical_cpus.to_string());
    row("Total RAM", &format_bytes(system.total_memory));
    row("Available RAM", &format_bytes(system.available_memory));
    match &system.gpu {
        Some(gpu) => {
            row("GPU", &gpu.name);
            row("GPU memory", &format_bytes(gpu.total_memory));
            row(
                "Available VRAM",
                &gpu.available_memory
                    .map(format_bytes)
                    .unwrap_or_else(|| "unknown".into()),
            );
            row("GPU detection", &gpu.source);
        }
        None => row("GPU", "not detected; CPU-only planning will be used"),
    }
}

fn print_model(path: &Path, model: &ModelInfo) {
    println!("{LOGO}");
    println!("MODEL PROFILE\n");
    row("File", &path.display().to_string());
    row("Name", model.name.as_deref().unwrap_or("unknown"));
    row(
        "Architecture",
        model.architecture.as_deref().unwrap_or("unknown"),
    );
    row("GGUF version", &model.version.to_string());
    row("Quantization", &gguf::quantization_name(model.file_type));
    row("File size", &format_bytes(model.file_size));
    row("Tensors", &model.tensor_count.to_string());
    optional_row("Transformer blocks", model.block_count);
    optional_row("Training context", model.context_length);
    optional_row("Embedding length", model.embedding_length);
    optional_row("Attention heads", model.attention_heads);
    optional_row("KV heads", model.kv_heads);
    row("Estimate confidence", planner::confidence(model));
}

fn print_plan(path: &Path, model: &ModelInfo, system: &SystemInfo, plan: &Plan) {
    println!("{LOGO}");
    println!("MEMORY PLAN\n");
    row(
        "Model",
        model.name.as_deref().unwrap_or(&path.display().to_string()),
    );
    row(
        "Architecture",
        model.architecture.as_deref().unwrap_or("unknown"),
    );
    row("Quantization", &gguf::quantization_name(model.file_type));
    row("Model size", &format_bytes(model.file_size));
    row("Available RAM", &format_bytes(system.available_memory));
    row("Planning budget", &format_bytes(plan.memory_budget));
    row("Estimate confidence", planner::confidence(model));
    println!();
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>12} {:>12} {:>13}",
        "PROFILE", "CONTEXT", "BATCH", "GPU LAY", "EST. RAM", "EST. VRAM", "STATUS"
    );
    println!("{}", "-".repeat(82));
    for profile in &plan.profiles {
        println!(
            "{:<14} {:>8} {:>8} {:>8} {:>12} {:>12} {:>13}",
            profile.kind,
            profile.context_length,
            profile.batch_size,
            profile.gpu_layers,
            format_bytes(profile.estimated_ram),
            format_bytes(profile.estimated_vram),
            planner::budget_status(profile),
        );
        println!("  KV cache: {}", profile.kv_cache);
    }

    println!();
    if let Some(recommended) = planner::recommended_profile(plan) {
        println!("RECOMMENDED: {} profile", recommended.kind);
        println!(
            "llama-server --model \"{}\" --ctx-size {} --batch-size {} --n-gpu-layers {} --cache-type-k {} --cache-type-v {}",
            path.display(),
            recommended.context_length,
            recommended.batch_size,
            recommended.gpu_layers,
            recommended.kv_cache.to_ascii_lowercase(),
            recommended.kv_cache.to_ascii_lowercase(),
        );
    } else {
        println!("NO SAFE PROFILE: use a smaller or more heavily quantized model.");
        println!(
            "Estimated practical minimum: {}",
            format_bytes(planner::minimum_practical_budget(model))
        );
    }

    if !plan.warnings.is_empty() {
        println!("\nNOTES:");
        for warning in &plan.warnings {
            println!("- {warning}");
        }
    }
    println!("\nEstimates are planning guidance; verify them with runtime measurements.");
}

fn scan_models(directory: &Path) -> Result<Vec<(PathBuf, ModelInfo)>, String> {
    if !directory.is_dir() {
        return Err(format!("'{}' is not a directory", directory.display()));
    }
    let mut pending = vec![directory.to_path_buf()];
    let mut models = Vec::new();
    while let Some(current) = pending.pop() {
        let entries = fs::read_dir(&current)
            .map_err(|error| format!("could not read '{}': {error}", current.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|error| format!("could not read directory entry: {error}"))?;
            let file_type = entry.file_type().map_err(|error| {
                format!("could not inspect '{}': {error}", entry.path().display())
            })?;
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !file_type.is_file()
                || !entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
            {
                continue;
            }
            if let Ok(model) = gguf::inspect(&entry.path()) {
                models.push((entry.path(), model));
            }
        }
    }
    models.sort_by(|left, right| right.1.file_size.cmp(&left.1.file_size));
    Ok(models)
}

fn print_scan(
    directory: &Path,
    models: &[(PathBuf, ModelInfo)],
    system: &SystemInfo,
    memory_limit: Option<u64>,
) {
    println!("{LOGO}");
    println!("MODEL FIT REPORT\n");
    row("Directory", &directory.display().to_string());
    row("Models found", &models.len().to_string());
    row(
        "Memory budget",
        &format_bytes(memory_limit.unwrap_or(system.available_memory * 85 / 100)),
    );
    println!();
    if models.is_empty() {
        println!("No readable .gguf models were found.");
        return;
    }

    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "MODEL", "SIZE", "QUANT", "CONTEXT", "EST. RAM", "RESULT"
    );
    println!("{}", "-".repeat(88));
    for (path, model) in models {
        let plan = planner::create_plan(model, system, memory_limit);
        let recommended = planner::recommended_profile(&plan);
        let name = model
            .name
            .as_deref()
            .or_else(|| path.file_stem().and_then(|value| value.to_str()))
            .unwrap_or("unknown");
        let clipped_name = if name.chars().count() > 28 {
            format!("{}...", name.chars().take(25).collect::<String>())
        } else {
            name.to_string()
        };
        match recommended {
            Some(profile) => println!(
                "{:<28} {:>10} {:>10} {:>10} {:>12} {:>12}",
                clipped_name,
                format_bytes(model.file_size),
                gguf::quantization_name(model.file_type),
                profile.context_length,
                format_bytes(profile.estimated_ram),
                "FITS",
            ),
            None => println!(
                "{:<28} {:>10} {:>10} {:>10} {:>12} {:>12}",
                clipped_name,
                format_bytes(model.file_size),
                gguf::quantization_name(model.file_type),
                "-",
                format_bytes(planner::minimum_practical_budget(model)),
                "TOO LARGE",
            ),
        }
    }
    println!("\nUse 'divideit plan <MODEL>' for the full explanation and runtime command.");
}

fn row(label: &str, value: &str) {
    println!("{label:<20} {value}");
}

fn optional_row(label: &str, value: Option<u64>) {
    row(
        label,
        &value
            .map(|number| number.to_string())
            .unwrap_or_else(|| "unknown".into()),
    );
}
