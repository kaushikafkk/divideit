//! Core library for DivideIT.
//!
//! The crate intentionally has no third-party dependencies in its first release.
//! It can inspect a system, parse GGUF metadata, and create memory-aware execution
//! profiles without loading model tensors into memory.

pub mod gguf;
pub mod planner;
pub mod size;
pub mod system;
