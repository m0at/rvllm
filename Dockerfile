# Multi-stage build for rvllm on CUDA
# Stage 1: Build the Rust binary with CUDA support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

# Build with CUDA support, release mode
RUN cargo build --release --features cuda -p rvllm-server 2>&1 | tail -20

# Stage 2: Runtime image (smaller)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/rvllm /usr/local/bin/rvllm
COPY --from=builder /build/kernels/*.ptx /usr/local/share/rvllm/kernels/

# Default port
EXPOSE 8000
# Metrics port
EXPOSE 9090

ENV RVLLM_KERNEL_DIR=/usr/local/share/rvllm/kernels
ENV RUST_LOG=info

ENTRYPOINT ["rvllm"]
CMD ["serve", "--model", "/models/default", "--host", "0.0.0.0", "--port", "8000"]
