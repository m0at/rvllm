# Multi-stage build for rvllm on CUDA
# Stage 1: Build the Rust binary with CUDA support
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

# Install Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

# Build with CUDA support, release mode. The active workspace is
# v3/; the binary is `rvllm-server` (in the rvllm-serve crate).
RUN cd v3 && cargo build --release --features cuda --bin rvllm-server 2>&1 | tail -20

# Stage 2: Runtime image (smaller)
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/v3/target/release/rvllm-server /usr/local/bin/rvllm-server
COPY --from=builder /build/kernels/sm_121/*.ptx /usr/local/share/rvllm/kernels/sm_121/
COPY --from=builder /build/kernels/sm_121/manifest.json /usr/local/share/rvllm/kernels/sm_121/manifest.json

# Default port
EXPOSE 8010

ENV RVLLM_KERNEL_DIR=/usr/local/share/rvllm/kernels
ENV RUST_LOG=info

ENTRYPOINT ["rvllm-server"]
# The current CLI takes `--model-dir` (path to the HF model directory)
# and binds via `--bind ADDR:PORT`. There is no `serve` subcommand.
CMD ["--model-dir", "/models/default", "--bind", "0.0.0.0:8010"]
