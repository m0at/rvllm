#!/usr/bin/env python3
"""Patch build_fa3.sh to include fa3_combine_hdim256.cu in SRCS."""
from pathlib import Path
p = Path("/workspace/runs/2c6bbd0fc/rvllm/kernels/build_fa3.sh")
s = p.read_text()

NEW_LINE = '    "${SCRIPT_DIR}/fa3_combine_hdim256.cu"\n'
ANCHOR   = '    "${FA3_DIR}/flash_prepare_scheduler.cu"\n)\n'
REPLACE  = '    "${FA3_DIR}/flash_prepare_scheduler.cu"\n' + NEW_LINE + ')\n'

# Undo any prior bad inserts
for junk in ['    "/fa3_combine_hdim256.cu"\n',
             '    "\\/fa3_combine_hdim256.cu"\n']:
    s = s.replace(junk, "")

if "fa3_combine_hdim256.cu" not in s:
    if ANCHOR not in s:
        raise SystemExit("anchor not found in build_fa3.sh")
    s = s.replace(ANCHOR, REPLACE, 1)
    p.write_text(s)
    print("patched")
else:
    print("already patched")
