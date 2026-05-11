.PHONY: build build-cuda check check-cuda test test-cuda kernels bench bench-python bench-compare docker deploy-provision deploy-push deploy-bench deploy-teardown a100-bench validate validate-full validate-kernels loc clean

# The active workspace lives in v3/. All cargo targets `cd v3`
# first; `cargo build --release` from the repo root has no
# Cargo.toml to find. The binary produced is `rvllm-server` in the
# `rvllm-serve` crate. Older targets used a `rvllm` crate that no
# longer exists.

# Local development (mock-gpu, Mac / Linux without CUDA)
build:
	cd v3 && cargo build --release --bin rvllm-server

# CUDA build (Linux + NVIDIA GPU)
build-cuda:
	cd v3 && cargo build --release --features cuda --bin rvllm-server

# GB10 build (DGX Spark, sm_121 + cuda)
build-gb10:
	cd v3 && cargo build --release --features cuda,gb10 --bin rvllm-server

# Check workspace compiles (no CUDA). Excludes runtime/bench/serve
# which require cuda transitively; the non-cuda DAG is what this
# target enforces. Use `make check-cuda` for the full path.
check:
	cd v3 && cargo check --workspace --exclude rvllm-runtime --exclude rvllm-bench --exclude rvllm-serve

# Check the full workspace compiles with CUDA features.
check-cuda:
	cd v3 && cargo check --workspace --features rvllm-runtime/cuda

# Compile .cu kernels to .ptx (requires nvcc)
kernels:
	bash kernels/build.sh sm_121

# Run all tests (no-CUDA crates only — most coverage lives there).
test:
	cd v3 && cargo test --workspace --exclude rvllm-runtime --exclude rvllm-bench --exclude rvllm-serve

# Tests for the CUDA-feature path (needs a real GPU + matching driver).
test-cuda:
	cd v3 && cargo test --features cuda -p rvllm-serve --lib
	cd v3 && cargo test --features cuda -p rvllm-runtime --lib

# Run benchmarks (Rust)
bench:
	cargo bench --package rvllm-bench --bench sampling_bench

# Run Python benchmarks
bench-python:
	python3 benches/bench_python.py

# Compare Rust vs Python benchmarks
bench-compare: bench bench-python
	python3 benches/compare.py

# Local build validation (no GPU required, runs in Docker)
validate:
	bash scripts/validate-local.sh

# Full validation including release binary
validate-full:
	bash scripts/validate-local.sh --full

# Kernel-only validation (fastest)
validate-kernels:
	bash scripts/validate-local.sh --kernels

# Build Docker image
docker:
	bash scripts/build-docker.sh

# Deploy to vast.ai A100
deploy-provision:
	bash deploy/vastai-provision.sh

deploy-push:
	bash deploy/vastai-deploy.sh

deploy-bench:
	bash deploy/vastai-benchmark.sh

deploy-teardown:
	bash deploy/vastai-teardown.sh

# Full A100 benchmark pipeline
a100-bench: deploy-provision deploy-push deploy-bench

# Count lines of code
loc:
	@find crates -name "*.rs" | xargs wc -l | tail -1
	@echo "CUDA kernels:"
	@find kernels -name "*.cu" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "  0 lines"

# Clean
clean:
	cargo clean
	rm -f benches/python_results.json
	rm -f deploy/results_*.json
