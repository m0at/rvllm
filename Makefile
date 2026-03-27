.PHONY: build build-cuda check check-cuda test test-cuda kernels bench bench-python bench-compare docker deploy-provision deploy-push deploy-bench deploy-teardown a100-bench loc clean

# Local development (Mac, mock-gpu)
build:
	cargo build --release -p rvllm-server

# CUDA build (Linux + NVIDIA GPU)
build-cuda:
	cargo build --release --features cuda -p rvllm-server

# Check workspace compiles (mock-gpu, Mac)
check:
	cargo check --workspace

# Check workspace compiles with CUDA features (needs cudarc, OK to fail without CUDA toolkit)
check-cuda:
	cargo check --workspace --features rvllm-server/cuda

# Compile .cu kernels to .ptx (requires nvcc)
kernels:
	bash kernels/build.sh

# Run all tests
test:
	cargo test --workspace

# Run tests with CUDA features
test-cuda:
	cargo test --workspace --features rvllm-server/cuda

# Run benchmarks (Rust)
bench:
	cargo bench --package rvllm-bench --bench sampling_bench

# Run Python benchmarks
bench-python:
	python3 benches/bench_python.py

# Compare Rust vs Python benchmarks
bench-compare: bench bench-python
	python3 benches/compare.py

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
