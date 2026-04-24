#!/usr/bin/env python3
"""Probe safetensors header in the exact way gemma4_arch.rs::detect_weight_prefix does."""
import sys, struct, json
p = sys.argv[1]
with open(p, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
keys = [k for k in header if k != "__metadata__"]
print(f"total header entries: {len(keys)}")
# Rust iterates hashmap keys, order is unspecified but let's see first 20
print("first 20 header keys (file order):")
for k in keys[:20]:
    print(" ", k)

# What detect_from_keys would find
for k in keys:
    if k.startswith("model.language_model."):
        print(f"\nDETECT: model.language_model (found key: {k})")
        break
    if k.startswith("language_model."):
        print(f"\nDETECT: language_model (found key: {k})")
        break
else:
    print("\nDETECT: fallback 'model'")
