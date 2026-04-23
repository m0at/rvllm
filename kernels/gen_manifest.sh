#!/bin/bash
# Generate manifest.json for a kernel output directory.
#
# Usage: ./gen_manifest.sh <arch_dir> <revision>
#   arch_dir: e.g. kernels/sm_121/ — must contain *.ptx files
#   revision: git SHA (or "dev" for local builds)
#
# Emits <arch_dir>/manifest.json with:
#   { revision, arch, entries: { <stem>: { path, sha256, bytes } } }
#
# The logical name is the PTX file stem (e.g. fp8_gemv.ptx → "fp8_gemv").
# This matches how `KernelLoader::load_ptx(name)` looks up artifacts.

set -e

ARCH_DIR="${1:?usage: $0 <arch_dir> <revision>}"
REVISION="${2:?usage: $0 <arch_dir> <revision>}"

if [ ! -d "$ARCH_DIR" ]; then
    echo "gen_manifest: $ARCH_DIR is not a directory" >&2
    exit 1
fi

ARCH_NAME="$(basename "$ARCH_DIR")"
OUT="$ARCH_DIR/manifest.json"

python3 - "$ARCH_DIR" "$ARCH_NAME" "$REVISION" "$OUT" <<'PY'
import hashlib, json, os, sys
arch_dir, arch_name, revision, out = sys.argv[1:5]
entries = {}
for fn in sorted(os.listdir(arch_dir)):
    if not fn.endswith(".ptx"):
        continue
    path = os.path.join(arch_dir, fn)
    with open(path, "rb") as f:
        data = f.read()
    stem = fn[:-len(".ptx")]
    entries[stem] = {
        "path": fn,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
doc = {"revision": revision, "arch": arch_name, "entries": entries}
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"  manifest: {out} ({len(entries)} entries)")
PY
