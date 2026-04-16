#!/bin/bash
# download_kernels_hf.sh -- Download pre-compiled kernels from HuggingFace.
# Replaces kernel compilation on the GPU box.
#
# Nukes any existing kernels/sm_XX/ dir and downloads fresh from HF.
#
# Usage:
#   bash deploy/download_kernels_hf.sh              # auto-detect arch, use REVISION SHA
#   bash deploy/download_kernels_hf.sh sm_90 abc123 # explicit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ARCH=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')}
ARCH="sm_${ARCH#sm_}"
SHA=${2:-$(cat "$REPO_DIR/REVISION" 2>/dev/null || git -C "$REPO_DIR" rev-parse HEAD)}

HF_REPO="and-y/rvllm-kernels"
KERNEL_DIR="$REPO_DIR/kernels/$ARCH"

echo "Downloading kernels from HF: $HF_REPO"
echo "  SHA:  $SHA"
echo "  Arch: $ARCH"

# Nuke existing -- never trust what's on the box
rm -rf "$KERNEL_DIR"
mkdir -p "$KERNEL_DIR"

python3 - "$HF_REPO" "$KERNEL_DIR" "$SHA" "$ARCH" <<'PYEOF'
import sys, os
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

hf_repo = sys.argv[1]
kernel_dir = sys.argv[2]
sha = sys.argv[3]
arch = sys.argv[4]

api = HfApi()
prefix = f"{sha}/{arch}/"

# List files under this SHA+arch
all_files = api.list_repo_files(hf_repo)
matching = [f for f in all_files if f.startswith(prefix) and not "/obj/" in f]

if not matching:
    # Fall back to latest/<arch>/ when exact SHA not found (kernel .cu files unchanged)
    fallback = f"latest/{arch}/"
    matching = [f for f in all_files if f.startswith(fallback) and not "/obj/" in f]
    if matching:
        prefix = fallback
        print(f"Exact SHA not found, using latest/{arch}/ ({len(matching)} files)")
    else:
        print(f"FATAL: No kernels found at {sha}/{arch}/ or latest/{arch}/ in {hf_repo}")
        print(f"Available: {set('/'.join(f.split('/')[:2]) for f in all_files)}")
        sys.exit(1)

print(f"Found {len(matching)} kernel files")

for f in matching:
    local_name = f[len(prefix):]  # strip prefix to get filename
    if "/" in local_name:
        continue  # skip subdirs like obj/
    dest = os.path.join(kernel_dir, local_name)
    downloaded = hf_hub_download(hf_repo, f, local_dir=kernel_dir, local_dir_use_symlinks=False)
    # hf_hub_download puts files in subdirs, move to flat
    if os.path.exists(downloaded) and downloaded != dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(downloaded, dest)
    print(f"  {local_name}")

# Verify we got the essentials
ptx_count = len([f for f in os.listdir(kernel_dir) if f.endswith('.ptx')])
so_exists = os.path.exists(os.path.join(kernel_dir, 'libcutlass_kernels.so'))
print(f"\nDownloaded: {ptx_count} PTX, SO={'yes' if so_exists else 'NO'}")

if ptx_count == 0:
    print("FATAL: No PTX files downloaded")
    sys.exit(1)
if not so_exists:
    print("FATAL: No libcutlass_kernels.so downloaded")
    sys.exit(1)
PYEOF

echo "Kernels ready at $KERNEL_DIR"
