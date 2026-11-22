<div align="center">

# DivideIT

**Run larger local LLMs on smaller machines.**

A dependency-free Rust CLI that inspects GGUF models and recommends memory-aware
llama.cpp configurations for the hardware you actually have.

![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-MVP-brightgreen)

</div>

## Why DivideIT?

Running a local model is often a cycle of guessing flags, crashing with an out-of-memory error,
and trying again. DivideIT turns that guesswork into an explicit plan.

It reads the model metadata without loading tensors, detects available RAM and VRAM, estimates
weight and KV-cache memory, and produces three useful profiles:

- **Low memory** for constrained machines.
- **Balanced** for everyday local use.
- **Performance** for maximum throughput.

## See it in action

### Hardware inspection

![DivideIT system profile](docs/assets/system-profile.png)

### Memory-aware execution plan

![DivideIT memory plan](docs/assets/memory-plan.png)

> DivideIT does not claim that dense LLM layers can run independently. Every generated token still
> traverses every transformer block. The project reduces memory pressure through informed model
> placement, context sizing, memory mapping, and KV-cache precision.

## Quick start

```bash
git clone https://github.com/kaushikafkk/divideit.git
cd divideit
cargo build --release
```

Inspect your machine:

```bash
divideit inspect-system
```

Inspect a GGUF model without loading its tensors:

```bash
divideit inspect-model ./models/qwen2.5-7b-q4_k_m.gguf
```

Create a plan with an 8 GiB host-memory budget:

```bash
divideit plan ./models/qwen2.5-7b-q4_k_m.gguf --memory-limit 8GB
```

DivideIT emits a reproducible `llama-server` command for the recommended profile.

## What it calculates

```text
estimated host RAM = CPU weights + CPU KV cache + activation buffer + runtime margin
estimated VRAM     = GPU weights + GPU KV cache + backend margin
```

KV-cache size is derived from transformer blocks, attention heads, KV heads, embedding length,
context length, and cache precision. If metadata is incomplete, DivideIT reports lower confidence
and uses a conservative fallback.

## Commands

| Command | Purpose |
|---|---|
| `divideit inspect-system` | Detect CPU, RAM, GPU, and VRAM |
| `divideit inspect-model <path>` | Parse GGUF metadata safely |
| `divideit plan <path>` | Generate execution profiles |
| `divideit plan <path> -m 8GB` | Plan against a fixed host-memory budget |
| `divideit scan <directory> -m 8GB` | Compare every local GGUF against one budget |

## How this differs from Colibrì

[Colibrì](https://github.com/JustVugg/colibri) is an experimental inference engine specialized for
streaming the experts of the enormous GLM-5.2 Mixture-of-Experts model from storage. DivideIT is not
an inference engine and does not reproduce that work. It is a model-agnostic preflight, comparison,
and configuration layer for GGUF runtimes. Its `scan` command can compare a collection of models
before any of them are loaded, while its profiles explain the memory trade-offs they make.

## Design goals

- **Honest estimates:** show confidence and limitations.
- **Tiny footprint:** no third-party Rust dependencies in v0.1.
- **Safe parsing:** validate external lengths before allocating memory.
- **Reproducible advice:** emit concrete runtime arguments.
- **Measurable progress:** compare predictions with real runtime telemetry in v0.2.

See [architecture](docs/ARCHITECTURE.md) and the [roadmap](docs/ROADMAP.md) for implementation
details and planned runtime monitoring.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The unit tests build an in-memory GGUF fixture, so no model download is required.

## Project status

Version 0.1 is a working planning MVP. It does not yet start or monitor llama.cpp. That runtime
controller is deliberately the next milestone so the current estimates can be tested before
automatic tuning is introduced.

## Contributing

Bug reports and benchmark results from different hardware are especially useful. Please include
your operating system, model quantization, available RAM/VRAM, predicted usage, and measured peak
usage.

## License

[MIT](LICENSE)
