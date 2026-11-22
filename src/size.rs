use std::fmt;

pub const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;
pub const GIB: u64 = 1024 * MIB;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeParseError(pub String);

impl fmt::Display for SizeParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

pub fn parse_size(value: &str) -> Result<u64, SizeParseError> {
    let normalized = value.trim().to_ascii_lowercase().replace(' ', "");
    if normalized.is_empty() {
        return Err(SizeParseError("memory size cannot be empty".into()));
    }

    let split_at = normalized
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(normalized.len());
    let (number, suffix) = normalized.split_at(split_at);
    let number: f64 = number
        .parse()
        .map_err(|_| SizeParseError(format!("invalid memory size: {value}")))?;
    if !number.is_finite() || number < 0.0 {
        return Err(SizeParseError(format!("invalid memory size: {value}")));
    }

    let multiplier = match suffix {
        "" | "b" => 1,
        "k" | "kb" | "kib" => KIB,
        "m" | "mb" | "mib" => MIB,
        "g" | "gb" | "gib" => GIB,
        "t" | "tb" | "tib" => 1024 * GIB,
        _ => {
            return Err(SizeParseError(format!(
                "unknown size suffix '{suffix}'; use MB, GB, MiB, or GiB"
            )))
        }
    };

    let bytes = number * multiplier as f64;
    if bytes > u64::MAX as f64 {
        return Err(SizeParseError("memory size is too large".into()));
    }
    Ok(bytes.round() as u64)
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_sizes() {
        assert_eq!(parse_size("8GB").unwrap(), 8 * GIB);
        assert_eq!(parse_size("1.5 GiB").unwrap(), GIB + 512 * MIB);
        assert_eq!(parse_size("256m").unwrap(), 256 * MIB);
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(format_bytes(8 * GIB), "8.00 GiB");
        assert_eq!(format_bytes(512 * MIB), "512.0 MiB");
    }
}
