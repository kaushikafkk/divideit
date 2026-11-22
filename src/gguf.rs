use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const MAX_STRING_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 100_000_000;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub file_size: u64,
    pub file_type: Option<u64>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub attention_heads: Option<u64>,
    pub kv_heads: Option<u64>,
}

#[derive(Debug, Clone)]
enum ScalarValue {
    Text(String),
    Unsigned(u64),
    Signed(i64),
    Other,
}

impl ScalarValue {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            Self::Signed(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            _ => None,
        }
    }
}

pub fn inspect(path: &Path) -> Result<ModelInfo, String> {
    let file = File::open(path)
        .map_err(|error| format!("could not open '{}': {error}", path.display()))?;
    let file_size = file
        .metadata()
        .map_err(|error| format!("could not inspect '{}': {error}", path.display()))?
        .len();
    parse_reader(BufReader::new(file), file_size)
}

fn parse_reader<R: Read>(mut reader: R, file_size: u64) -> Result<ModelInfo, String> {
    let mut magic = [0_u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| "file is too short to be a GGUF model".to_string())?;
    if &magic != b"GGUF" {
        return Err("invalid GGUF signature".into());
    }

    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(format!(
            "unsupported GGUF version {version}; DivideIT currently supports versions 2 and 3"
        ));
    }
    let tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;
    if metadata_count > 1_000_000 {
        return Err("GGUF metadata count is unreasonably large".into());
    }

    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let value = read_value(&mut reader, value_type, 0)?;
        metadata.insert(key, value);
    }

    let architecture = text_value(&metadata, "general.architecture");
    let architecture_prefix = architecture.as_deref().unwrap_or("llama");
    let prefixed = |suffix: &str| format!("{architecture_prefix}.{suffix}");
    let block_count = number_value(&metadata, &prefixed("block_count"));
    let context_length = number_value(&metadata, &prefixed("context_length"));
    let embedding_length = number_value(&metadata, &prefixed("embedding_length"));
    let attention_heads = number_value(&metadata, &prefixed("attention.head_count"));
    let kv_heads = number_value(&metadata, &prefixed("attention.head_count_kv"));

    Ok(ModelInfo {
        name: text_value(&metadata, "general.name"),
        architecture,
        version,
        tensor_count,
        metadata_count,
        file_size,
        file_type: number_value(&metadata, "general.file_type"),
        block_count,
        context_length,
        embedding_length,
        attention_heads,
        kv_heads,
    })
}

fn text_value(metadata: &HashMap<String, ScalarValue>, key: &str) -> Option<String> {
    metadata.get(key)?.as_text().map(str::to_string)
}

fn number_value(metadata: &HashMap<String, ScalarValue>, key: &str) -> Option<u64> {
    metadata.get(key)?.as_u64()
}

fn read_value<R: Read>(
    reader: &mut R,
    value_type: u32,
    depth: usize,
) -> Result<ScalarValue, String> {
    if depth > 4 {
        return Err("GGUF metadata arrays are nested too deeply".into());
    }
    match value_type {
        0 => Ok(ScalarValue::Unsigned(read_u8(reader)? as u64)),
        1 => Ok(ScalarValue::Signed(read_i8(reader)? as i64)),
        2 => Ok(ScalarValue::Unsigned(read_u16(reader)? as u64)),
        3 => Ok(ScalarValue::Signed(read_i16(reader)? as i64)),
        4 => Ok(ScalarValue::Unsigned(read_u32(reader)? as u64)),
        5 => Ok(ScalarValue::Signed(read_i32(reader)? as i64)),
        6 => {
            read_f32(reader)?;
            Ok(ScalarValue::Other)
        }
        7 => {
            read_u8(reader)?;
            Ok(ScalarValue::Other)
        }
        8 => Ok(ScalarValue::Text(read_string(reader)?)),
        9 => {
            let element_type = read_u32(reader)?;
            let length = read_u64(reader)?;
            if length > MAX_ARRAY_ITEMS {
                return Err("GGUF metadata array is unreasonably large".into());
            }
            for _ in 0..length {
                read_value(reader, element_type, depth + 1)?;
            }
            Ok(ScalarValue::Other)
        }
        10 => Ok(ScalarValue::Unsigned(read_u64(reader)?)),
        11 => Ok(ScalarValue::Signed(read_i64(reader)?)),
        12 => {
            read_f64(reader)?;
            Ok(ScalarValue::Other)
        }
        other => Err(format!("unsupported GGUF metadata value type {other}")),
    }
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let length = read_u64(reader)?;
    if length > MAX_STRING_BYTES {
        return Err("GGUF string is unreasonably large".into());
    }
    let mut bytes = vec![0_u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| "unexpected end of GGUF string".to_string())?;
    String::from_utf8(bytes).map_err(|_| "GGUF metadata contains invalid UTF-8".into())
}

macro_rules! read_number {
    ($name:ident, $type:ty, $bytes:expr) => {
        fn $name<R: Read>(reader: &mut R) -> Result<$type, String> {
            let mut buffer = [0_u8; $bytes];
            reader
                .read_exact(&mut buffer)
                .map_err(|_| "unexpected end of GGUF metadata".to_string())?;
            Ok(<$type>::from_le_bytes(buffer))
        }
    };
}

read_number!(read_u16, u16, 2);
read_number!(read_i16, i16, 2);
read_number!(read_u32, u32, 4);
read_number!(read_i32, i32, 4);
read_number!(read_f32, f32, 4);
read_number!(read_u64, u64, 8);
read_number!(read_i64, i64, 8);
read_number!(read_f64, f64, 8);

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, String> {
    let mut buffer = [0_u8; 1];
    reader
        .read_exact(&mut buffer)
        .map_err(|_| "unexpected end of GGUF metadata".to_string())?;
    Ok(buffer[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8, String> {
    Ok(read_u8(reader)? as i8)
}

pub fn quantization_name(file_type: Option<u64>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(2) => "Q4_0".into(),
        Some(3) => "Q4_1".into(),
        Some(7) => "Q8_0".into(),
        Some(8) => "Q5_0".into(),
        Some(9) => "Q5_1".into(),
        Some(10) => "Q2_K".into(),
        Some(11) => "Q3_K_S".into(),
        Some(12) => "Q3_K_M".into(),
        Some(13) => "Q3_K_L".into(),
        Some(14) => "Q4_K_S".into(),
        Some(15) => "Q4_K_M".into(),
        Some(16) => "Q5_K_S".into(),
        Some(17) => "Q5_K_M".into(),
        Some(18) => "Q6_K".into(),
        Some(19) => "IQ2_XXS".into(),
        Some(20) => "IQ2_XS".into(),
        Some(value) => format!("GGUF type {value}"),
        None => "unknown".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_text_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(bytes, value);
    }

    fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn parses_minimal_version_three_model() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        push_text_metadata(&mut bytes, "general.name", "DivideIT Demo");
        push_text_metadata(&mut bytes, "general.architecture", "llama");
        push_u32_metadata(&mut bytes, "general.file_type", 15);
        push_u32_metadata(&mut bytes, "llama.block_count", 32);
        push_u32_metadata(&mut bytes, "llama.context_length", 8192);
        push_u32_metadata(&mut bytes, "llama.embedding_length", 4096);
        push_u32_metadata(&mut bytes, "llama.attention.head_count_kv", 8);

        let result = parse_reader(Cursor::new(&bytes), 4_500_000_000).unwrap();
        assert_eq!(result.name.as_deref(), Some("DivideIT Demo"));
        assert_eq!(result.block_count, Some(32));
        assert_eq!(result.context_length, Some(8192));
        assert_eq!(quantization_name(result.file_type), "Q4_K_M");
    }

    #[test]
    fn rejects_non_gguf_files() {
        let error = parse_reader(Cursor::new(b"NOPE"), 4).unwrap_err();
        assert_eq!(error, "invalid GGUF signature");
    }
}
