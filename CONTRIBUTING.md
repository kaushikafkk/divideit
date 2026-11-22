# Contributing to DivideIT

Thanks for helping make local LLM inference more predictable.

## Good first contributions

- Test GGUF architectures that are not covered yet.
- Improve AMD, Intel, or Apple GPU discovery.
- Compare predicted and measured memory on real hardware.
- Add malformed-GGUF parser tests.
- Improve explanations without hiding uncertainty.

## Development checks

Run all checks before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

DivideIT 0.1 deliberately uses only the Rust standard library. Please discuss new dependencies in
an issue before adding one.

## Benchmark reports

Useful benchmark reports include:

- Operating system and CPU
- Total and available RAM
- GPU and VRAM
- Exact model and quantization
- Context and batch size
- DivideIT's prediction
- Measured peak RAM and VRAM
- Time to first token and tokens per second

Do not upload model weights or other files whose license does not permit redistribution.

## Pull requests

Keep changes focused, add tests for behavioral changes, and document any new estimation assumptions.
The planner must label uncertain results rather than presenting guesses as exact measurements.

