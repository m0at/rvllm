#!/bin/bash
set -euo pipefail
cd /Users/andy/rvllm
SHA=$(git rev-parse --short HEAD)
echo "$SHA" > /tmp/rvllm_sha.txt
echo "SHA=$SHA"
git archive --format=tar.gz --prefix=rvllm/ HEAD -o "/tmp/rvllm-${SHA}.tar.gz"
ls -la "/tmp/rvllm-${SHA}.tar.gz"
