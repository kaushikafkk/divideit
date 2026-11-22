use crate::size::KIB;
use std::process::Command;
use std::thread;

#[cfg(target_os = "linux")]
use std::fs;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub total_memory: u64,
    pub available_memory: Option<u64>,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub operating_system: String,
    pub logical_cpus: usize,
    pub total_memory: u64,
    pub available_memory: u64,
    pub gpu: Option<GpuInfo>,
}

pub fn inspect() -> Result<SystemInfo, String> {
    let (total_memory, available_memory) = memory_info()?;
    Ok(SystemInfo {
        operating_system: std::env::consts::OS.to_string(),
        logical_cpus: thread::available_parallelism().map_or(1, |count| count.get()),
        total_memory,
        available_memory,
        gpu: gpu_info(),
    })
}

fn gpu_info() -> Option<GpuInfo> {
    nvidia_gpu_info().or_else(platform_gpu_info)
}

fn nvidia_gpu_info() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .to_string();
    let mut values = line.split(',').map(str::trim);
    let name = values.next()?.to_string();
    let total_mib: u64 = values.next()?.parse().ok()?;
    let free_mib: u64 = values.next()?.parse().ok()?;
    Some(GpuInfo {
        name,
        total_memory: total_mib * 1024 * KIB,
        available_memory: Some(free_mib * 1024 * KIB),
        source: "nvidia-smi".into(),
    })
}

#[cfg(target_os = "windows")]
fn memory_info() -> Result<(u64, u64), String> {
    let script = "$os=Get-CimInstance Win32_OperatingSystem; ".to_string()
        + "Write-Output \"$($os.TotalVisibleMemorySize),$($os.FreePhysicalMemory)\"";
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .map_err(|error| format!("failed to query Windows memory: {error}"))?;
    if !output.status.success() {
        return Err("Windows memory query failed".into());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.trim();
    let (total, free) = line
        .split_once(',')
        .ok_or_else(|| format!("unexpected Windows memory response: {line}"))?;
    let total_kib: u64 = total
        .trim()
        .parse()
        .map_err(|_| "invalid total-memory value returned by Windows".to_string())?;
    let free_kib: u64 = free
        .trim()
        .parse()
        .map_err(|_| "invalid free-memory value returned by Windows".to_string())?;
    Ok((total_kib * KIB, free_kib * KIB))
}

#[cfg(target_os = "windows")]
fn platform_gpu_info() -> Option<GpuInfo> {
    let script = "Get-CimInstance Win32_VideoController | ".to_string()
        + "Sort-Object AdapterRAM -Descending | Select-Object -First 1 | "
        + "ForEach-Object { Write-Output \"$($_.Name)|$($_.AdapterRAM)\" }";
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (name, memory) = text.trim().split_once('|')?;
    let total_memory = memory.trim().parse().ok()?;
    Some(GpuInfo {
        name: name.trim().to_string(),
        total_memory,
        available_memory: None,
        source: "Windows video controller (approximate)".into(),
    })
}

#[cfg(target_os = "linux")]
fn memory_info() -> Result<(u64, u64), String> {
    let contents = fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("failed to read /proc/meminfo: {error}"))?;
    let mut total = None;
    let mut available = None;
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("MemTotal:") {
            total = parse_linux_kib(value);
        } else if let Some(value) = line.strip_prefix("MemAvailable:") {
            available = parse_linux_kib(value);
        }
    }
    match (total, available) {
        (Some(total), Some(available)) => Ok((total, available)),
        _ => Err("/proc/meminfo did not contain expected memory fields".into()),
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_kib(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .map(|v| v * KIB)
}

#[cfg(target_os = "linux")]
fn platform_gpu_info() -> Option<GpuInfo> {
    None
}

#[cfg(target_os = "macos")]
fn memory_info() -> Result<(u64, u64), String> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .map_err(|error| format!("failed to query macOS memory: {error}"))?;
    let total: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .map_err(|_| "invalid memory value returned by macOS".to_string())?;
    // `vm_stat` accounting differs across releases; use a conservative planning value.
    Ok((total, total * 3 / 4))
}

#[cfg(target_os = "macos")]
fn platform_gpu_info() -> Option<GpuInfo> {
    None
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn memory_info() -> Result<(u64, u64), String> {
    Err("system memory inspection is unsupported on this operating system".into())
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn platform_gpu_info() -> Option<GpuInfo> {
    None
}
