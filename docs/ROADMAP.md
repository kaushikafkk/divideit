# Roadmap

## 0.1 — Memory planner

- [x] Cross-platform system memory inspection
- [x] GGUF v2/v3 metadata parsing
- [x] KV-cache memory estimates
- [x] GPU-layer planning
- [x] Three execution profiles
- [x] Reproducible llama.cpp command generation
- [x] Recursive model-library fit report

## 0.2 — Runtime controller

- [ ] Discover or configure a local llama.cpp installation
- [ ] Launch and stop `llama-server`
- [ ] Record peak RAM, VRAM, time to first token, and tokens per second
- [ ] Detect unsafe memory pressure and stop gracefully
- [ ] Persist benchmark results in a local database

## 0.3 — Autotuner

- [ ] Run short controlled benchmarks across safe configurations
- [ ] Rank configurations by speed, memory, and stability
- [ ] Reuse known-good profiles for a machine/model pair
- [ ] Add AMD, Intel, and Apple GPU-specific discovery

## Research track

- On-demand expert caching for Mixture-of-Experts models
- Multi-GPU placement
- Remote worker execution
- Predictive memory models learned from local benchmark history

