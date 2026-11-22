use crate::gguf::ModelInfo;
use crate::size::{GIB, MIB};
use crate::system::SystemInfo;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum ProfileKind {
    LowMemory,
    Balanced,
    Performance,
}

impl fmt::Display for ProfileKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowMemory => formatter.write_str("LOW MEMORY"),
            Self::Balanced => formatter.write_str("BALANCED"),
            Self::Performance => formatter.write_str("PERFORMANCE"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    pub kind: ProfileKind,
    pub context_length: u64,
    pub batch_size: u64,
    pub kv_cache: &'static str,
    pub gpu_layers: u64,
    pub estimated_ram: u64,
    pub estimated_vram: u64,
    pub fits_memory_budget: bool,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub memory_budget: u64,
    pub profiles: Vec<ExecutionProfile>,
    pub warnings: Vec<String>,
}

pub fn create_plan(model: &ModelInfo, system: &SystemInfo, requested_budget: Option<u64>) -> Plan {
    let memory_budget = requested_budget.unwrap_or(system.available_memory * 85 / 100);
    let mut warnings = Vec::new();

    if model.block_count.is_none() {
        warnings.push(
            "model does not declare a transformer block count; GPU offload is disabled".into(),
        );
    }
    if model.embedding_length.is_none() || model.kv_heads.is_none() {
        warnings.push(
            "model lacks complete attention metadata; KV-cache estimates use a conservative fallback"
                .into(),
        );
    }
    if system
        .gpu
        .as_ref()
        .and_then(|gpu| gpu.available_memory)
        .is_none()
        && system.gpu.is_some()
    {
        warnings.push(
            "GPU free memory is unavailable; VRAM planning uses a conservative estimate".into(),
        );
    }

    let maximum_gpu_layers = estimate_gpu_layers(model, system);
    let maximum_context = model.context_length.unwrap_or(4096).max(512);
    let specifications = [
        (
            ProfileKind::LowMemory,
            2048_u64,
            64_u64,
            "Q4",
            0.625_f64,
            0.75_f64,
        ),
        (ProfileKind::Balanced, 4096, 256, "Q8", 1.125, 0.90),
        (ProfileKind::Performance, 8192, 512, "F16", 2.0, 1.0),
    ];

    let profiles = specifications
        .into_iter()
        .map(
            |(kind, context, batch, cache_name, cache_bytes, gpu_fraction)| {
                let gpu_layers = (maximum_gpu_layers as f64 * gpu_fraction).floor() as u64;
                fit_profile(
                    kind,
                    model,
                    memory_budget,
                    context.min(maximum_context),
                    batch,
                    cache_name,
                    cache_bytes,
                    gpu_layers,
                )
            },
        )
        .collect();

    Plan {
        memory_budget,
        profiles,
        warnings,
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_profile(
    kind: ProfileKind,
    model: &ModelInfo,
    memory_budget: u64,
    target_context: u64,
    batch_size: u64,
    kv_cache: &'static str,
    kv_bytes_per_element: f64,
    gpu_layers: u64,
) -> ExecutionProfile {
    let mut context = target_context.max(512);
    loop {
        let (ram, vram) =
            estimate_memory(model, context, batch_size, kv_bytes_per_element, gpu_layers);
        if ram <= memory_budget || context <= 512 {
            return ExecutionProfile {
                kind,
                context_length: context,
                batch_size,
                kv_cache,
                gpu_layers,
                estimated_ram: ram,
                estimated_vram: vram,
                fits_memory_budget: ram <= memory_budget,
            };
        }
        context = (context / 2).max(512);
    }
}

fn estimate_gpu_layers(model: &ModelInfo, system: &SystemInfo) -> u64 {
    let blocks = model.block_count.unwrap_or(0);
    let Some(gpu) = &system.gpu else {
        return 0;
    };
    if blocks == 0 {
        return 0;
    }
    let available = gpu.available_memory.unwrap_or(gpu.total_memory * 70 / 100);
    let reserve = (available / 10).max(512 * MIB);
    let usable = available.saturating_sub(reserve);
    let layer_weight = model.file_size / (blocks + 2);
    if layer_weight == 0 {
        return 0;
    }
    (usable / layer_weight).min(blocks)
}

fn estimate_memory(
    model: &ModelInfo,
    context: u64,
    batch_size: u64,
    kv_bytes_per_element: f64,
    gpu_layers: u64,
) -> (u64, u64) {
    let blocks = model.block_count.unwrap_or(0);
    let offloaded_fraction = if blocks == 0 {
        0.0
    } else {
        gpu_layers.min(blocks) as f64 / (blocks + 2) as f64
    };
    let gpu_weights = (model.file_size as f64 * offloaded_fraction) as u64;
    let cpu_weights = model.file_size.saturating_sub(gpu_weights);

    let embedding = model.embedding_length.unwrap_or(4096);
    let heads = model.attention_heads.unwrap_or(32).max(1);
    let kv_heads = model.kv_heads.unwrap_or(heads).max(1);
    let head_dimension = (embedding / heads).max(1);
    let cache_elements = 2_u64
        .saturating_mul(blocks.max(1))
        .saturating_mul(kv_heads)
        .saturating_mul(head_dimension)
        .saturating_mul(context);
    let kv_cache_bytes = (cache_elements as f64 * kv_bytes_per_element) as u64;
    let gpu_cache_fraction = if blocks == 0 {
        0.0
    } else {
        gpu_layers.min(blocks) as f64 / blocks as f64
    };
    let gpu_cache = (kv_cache_bytes as f64 * gpu_cache_fraction) as u64;
    let cpu_cache = kv_cache_bytes.saturating_sub(gpu_cache);

    let activation_buffer = (batch_size
        .saturating_mul(embedding)
        .saturating_mul(4)
        .saturating_mul(6))
    .max(256 * MIB);
    let runtime_overhead = 256 * MIB;
    let ram = cpu_weights
        .saturating_add(cpu_cache)
        .saturating_add(activation_buffer)
        .saturating_add(runtime_overhead);
    let vram = gpu_weights
        .saturating_add(gpu_cache)
        .saturating_add(if gpu_layers > 0 { 256 * MIB } else { 0 });
    (ram, vram)
}

pub fn confidence(model: &ModelInfo) -> &'static str {
    if model.block_count.is_some()
        && model.embedding_length.is_some()
        && model.attention_heads.is_some()
        && model.kv_heads.is_some()
    {
        "high"
    } else if model.block_count.is_some() && model.embedding_length.is_some() {
        "medium"
    } else {
        "low"
    }
}

pub fn recommended_profile(plan: &Plan) -> Option<&ExecutionProfile> {
    plan.profiles
        .iter()
        .find(|profile| matches!(profile.kind, ProfileKind::Balanced) && profile.fits_memory_budget)
        .or_else(|| {
            plan.profiles
                .iter()
                .find(|profile| profile.fits_memory_budget)
        })
}

pub fn budget_status(profile: &ExecutionProfile) -> &'static str {
    if profile.fits_memory_budget {
        "FITS"
    } else {
        "OVER BUDGET"
    }
}

pub fn minimum_practical_budget(model: &ModelInfo) -> u64 {
    model.file_size.saturating_add(GIB / 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::GpuInfo;

    fn model() -> ModelInfo {
        ModelInfo {
            name: Some("test".into()),
            architecture: Some("llama".into()),
            version: 3,
            tensor_count: 291,
            metadata_count: 24,
            file_size: 4 * GIB,
            file_type: Some(15),
            block_count: Some(32),
            context_length: Some(8192),
            embedding_length: Some(4096),
            attention_heads: Some(32),
            kv_heads: Some(8),
        }
    }

    fn system() -> SystemInfo {
        SystemInfo {
            operating_system: "test".into(),
            logical_cpus: 8,
            total_memory: 16 * GIB,
            available_memory: 12 * GIB,
            gpu: Some(GpuInfo {
                name: "test GPU".into(),
                total_memory: 6 * GIB,
                available_memory: Some(5 * GIB),
                source: "test".into(),
            }),
        }
    }

    #[test]
    fn creates_three_profiles_and_recommends_balanced() {
        let plan = create_plan(&model(), &system(), Some(8 * GIB));
        assert_eq!(plan.profiles.len(), 3);
        assert!(matches!(
            recommended_profile(&plan).unwrap().kind,
            ProfileKind::Balanced
        ));
        assert!(plan.profiles.iter().all(|profile| profile.gpu_layers <= 32));
    }

    #[test]
    fn reduces_context_to_fit_a_small_budget() {
        let plan = create_plan(&model(), &system(), Some(2 * GIB));
        assert!(plan
            .profiles
            .iter()
            .all(|profile| profile.context_length <= 8192));
        assert!(plan
            .profiles
            .iter()
            .any(|profile| profile.context_length == 512));
    }
}
