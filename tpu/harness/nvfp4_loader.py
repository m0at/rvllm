#!/usr/bin/env python3
"""NVFP4 modelopt safetensors loader for MiniMax-M2.7-NVFP4.

Reads safetensors shards produced by NVIDIA modelopt NVFP4 quantization.
NVFP4 weights are stored as:
  <base>.weight_packed   uint8  shape (rows, cols/2)  -- two FP4 values per byte
  <base>.weight_scale    uint8  shape (rows, cols/16) -- FP8 E4M3 scale per 16 elements

Tensors whose names match the config `quantization_config.ignore` patterns stay
as plain bf16 (attention, router gates, embeddings, lm_head, MTP heads).

Dequant at load (path B) is delegated to the Zig SIMD library via ctypes.
"""
import argparse
import ctypes
import fnmatch
import glob
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    from safetensors import safe_open
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

try:
    import ml_dtypes
    HAVE_ML_DTYPES = True
except Exception:
    HAVE_ML_DTYPES = False


# ---- FFI to librvllm_zig ----------------------------------------------------

def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..'))


def _find_zig_lib():
    env = os.environ.get('RVLLM_ZIG_LIB', '').strip()
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(
                "RVLLM_ZIG_LIB=%s does not exist. Build it with "
                "`cd zig && zig build -Doptimize=ReleaseFast`." % env)
        return env
    root = _repo_root()
    candidates = [
        os.path.join(root, 'zig', 'zig-out', 'lib', 'librvllm_zig.so'),
        os.path.join(root, 'zig', 'zig-out', 'lib', 'librvllm_zig.dylib'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "librvllm_zig not found. Set RVLLM_ZIG_LIB or build with "
        "`cd zig && zig build -Doptimize=ReleaseFast`. Looked in: %s"
        % ", ".join(candidates))


_ZIG = None


def _load_zig():
    global _ZIG
    if _ZIG is not None:
        return _ZIG
    lib_path = _find_zig_lib()
    lib = ctypes.CDLL(lib_path)

    lib.rvllm_nvfp4_to_int8.restype = ctypes.c_int32
    lib.rvllm_nvfp4_to_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,   # packed_ptr, packed_len
        ctypes.c_void_p, ctypes.c_size_t,   # scales_ptr, scales_len
        ctypes.c_size_t, ctypes.c_size_t,   # rows, cols
        ctypes.c_void_p,                     # out_i8_ptr
        ctypes.c_void_p,                     # out_row_scales_ptr
        ctypes.c_size_t,                     # n_threads
    ]

    lib.rvllm_nvfp4_to_bf16.restype = ctypes.c_int32
    lib.rvllm_nvfp4_to_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p,                     # out_bf16_ptr (u16)
        ctypes.c_size_t,                     # n_threads
    ]

    _ZIG = lib
    return lib


# ---- dataclass --------------------------------------------------------------

@dataclass
class NvFp4Tensor:
    name: str
    shape: tuple
    packed: np.ndarray
    scales: np.ndarray
    group_size: int = 16
    global_scale: float = 1.0
    input_scale: float = 1.0


# ---- in-house safetensors reader (fallback) --------------------------------

class _MmapShard:
    """Minimal in-house safetensors reader: parses JSON header + mmaps tensor bytes."""

    def __init__(self, path):
        self.path = path
        f = open(path, 'rb')
        self._f = f
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        self.header = json.loads(header_bytes)
        self.data_start = 8 + header_len
        f.seek(0, os.SEEK_END)
        self.file_size = f.tell()
        self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return [k for k in self.header.keys() if k != '__metadata__']

    def get_tensor(self, name):
        info = self.header[name]
        shape = tuple(info['shape'])
        dtype_str = info['dtype']
        s, e = info['data_offsets']
        off = self.data_start + s
        nbytes = e - s
        raw = np.frombuffer(self._mm, dtype=np.uint8, count=nbytes, offset=off)
        return _view_as_dtype(raw, dtype_str, shape)

    def get_info(self, name):
        return self.header[name]

    def close(self):
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


def _view_as_dtype(raw_u8, dtype_str, shape):
    d = dtype_str.upper()
    if d in ('U8', 'UINT8'):
        return raw_u8.reshape(shape)
    if d in ('I8', 'INT8'):
        return raw_u8.view(np.int8).reshape(shape)
    if d in ('BF16', 'BFLOAT16'):
        return raw_u8.view(np.uint16).reshape(shape)
    if d in ('F16', 'FLOAT16'):
        return raw_u8.view(np.float16).reshape(shape)
    if d in ('F32', 'FLOAT32'):
        return raw_u8.view(np.float32).reshape(shape)
    if d in ('F8_E4M3', 'FLOAT8_E4M3FN', 'F8E4M3'):
        return raw_u8.reshape(shape)  # raw uint8 bits
    if d in ('I32', 'INT32'):
        return raw_u8.view(np.int32).reshape(shape)
    if d in ('I64', 'INT64'):
        return raw_u8.view(np.int64).reshape(shape)
    raise ValueError("unsupported safetensors dtype: %s" % dtype_str)


# ---- reader -----------------------------------------------------------------

class ModeloptSafetensorsReader:
    """Read a modelopt-NVFP4 safetensors model directory.

    Opens all shards lazily. `is_nvfp4(name)` returns True when `name` plus its
    `.weight_packed` / `.weight_scale` twins are present and name matches no
    ignore pattern. `read_nvfp4` returns an NvFp4Tensor; `read_bf16` returns a
    uint16 (bf16-bits) ndarray.
    """

    WEIGHT_SUFFIX = '.weight'
    PACKED_SUFFIX = '.weight_packed'
    SCALE_SUFFIX = '.weight_scale'
    SCALE2_SUFFIX = '.weight_scale_2'
    INPUT_SCALE_SUFFIX = '.input_scale'

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.config = self._load_config()
        qcfg = self.config.get('quantization_config', {}) or {}
        self.ignore_patterns = list(qcfg.get('ignore', []) or [])
        weights_cfg = qcfg.get('weights', {}) or {}
        self.group_size = int(weights_cfg.get('group_size', 16))

        self._shard_paths = self._discover_shards()
        if not self._shard_paths:
            raise FileNotFoundError(
                "no *.safetensors shards found in %s" % model_dir)

        # name -> shard_path
        self._name_to_shard = {}
        self._shards = {}  # path -> handle
        for p in self._shard_paths:
            keys = self._peek_keys(p)
            for k in keys:
                self._name_to_shard[k] = p

    # ---- config -----------------------------------------------------------

    def _load_config(self):
        path = os.path.join(self.model_dir, 'config.json')
        if not os.path.exists(path):
            raise FileNotFoundError("config.json not found in %s" % self.model_dir)
        with open(path) as f:
            return json.load(f)

    # ---- shard discovery --------------------------------------------------

    def _discover_shards(self):
        idx = os.path.join(self.model_dir, 'model.safetensors.index.json')
        if os.path.exists(idx):
            with open(idx) as f:
                index = json.load(f)
            names = sorted(set(index['weight_map'].values()))
            return [os.path.join(self.model_dir, n) for n in names]
        found = sorted(glob.glob(os.path.join(self.model_dir, '*.safetensors')))
        return found

    def _peek_keys(self, path):
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
        return [k for k in header.keys() if k != '__metadata__']

    def _open_shard(self, path):
        h = self._shards.get(path)
        if h is not None:
            return h
        if HAVE_SAFETENSORS:
            h = safe_open(path, framework='numpy')
        else:
            h = _MmapShard(path)
        self._shards[path] = h
        return h

    # ---- matching ---------------------------------------------------------

    def _matches_ignore(self, base_name):
        for pat in self.ignore_patterns:
            if fnmatch.fnmatchcase(base_name, pat):
                return True
            if fnmatch.fnmatchcase(base_name + '.weight', pat):
                return True
        return False

    def list_tensors(self):
        return sorted(self._name_to_shard.keys())

    def iter_shards(self):
        for p in self._shard_paths:
            yield p, self._open_shard(p)

    # ---- nvfp4 detection ---------------------------------------------------

    def is_nvfp4(self, name):
        """True when `name` is the logical (unquantized) tensor name whose
        on-disk form is one of the supported NVFP4 layouts.

        Accepts either the bare base (e.g. `model.layers.0.mlp.gate_proj`) or
        the full `<base>.weight` name, consistent with modelopt layouts.

        Supported layouts:
          A (modelopt 4-tensor, real schema):
              <base>.weight          uint8  (rows, cols/2) -- packed FP4 pairs
              <base>.weight_scale    uint8  (rows, cols/16) -- FP8 E4M3 block scale
              <base>.weight_scale_2  float32 scalar -- per-tensor global scale
              <base>.input_scale     float32 scalar -- per-tensor activation scale
          B (legacy pair):
              <base>.weight_packed + <base>.weight_scale
        """
        base = name[:-len(self.WEIGHT_SUFFIX)] if name.endswith(self.WEIGHT_SUFFIX) else name
        if self._matches_ignore(base):
            return False
        scale = base + self.SCALE_SUFFIX
        # Layout A: <base>.weight is uint8 packed + <base>.weight_scale present.
        alt_weight = base + self.WEIGHT_SUFFIX
        if alt_weight in self._name_to_shard and scale in self._name_to_shard:
            info = self._tensor_info(alt_weight)
            dtype_str = info['dtype'].upper()
            if dtype_str in ('U8', 'UINT8'):
                return True
        # Layout B (fallback): explicit weight_packed + weight_scale.
        packed = base + self.PACKED_SUFFIX
        if packed in self._name_to_shard and scale in self._name_to_shard:
            return True
        return False

    def _tensor_info(self, name):
        path = self._name_to_shard[name]
        h = self._open_shard(path)
        if HAVE_SAFETENSORS:
            # safe_open has no public header access; fall back to re-reading header
            with open(path, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_len))
            return header[name]
        return h.get_info(name)

    def _read_raw(self, name):
        if name not in self._name_to_shard:
            raise KeyError("tensor not found in any shard: %s" % name)
        path = self._name_to_shard[name]
        h = self._open_shard(path)
        if HAVE_SAFETENSORS:
            return h.get_tensor(name)
        return h.get_tensor(name)

    # ---- public reads -----------------------------------------------------

    def read_nvfp4(self, name):
        base = name[:-len(self.WEIGHT_SUFFIX)] if name.endswith(self.WEIGHT_SUFFIX) else name
        scale_name = base + self.SCALE_SUFFIX
        scale2_name = base + self.SCALE2_SUFFIX
        input_scale_name = base + self.INPUT_SCALE_SUFFIX

        # Prefer layout A (modelopt 4-tensor): <base>.weight as uint8 packed.
        alt = base + self.WEIGHT_SUFFIX
        packed_name = None
        if alt in self._name_to_shard:
            info = self._tensor_info(alt)
            if info['dtype'].upper() in ('U8', 'UINT8'):
                packed_name = alt
        if packed_name is None:
            legacy = base + self.PACKED_SUFFIX
            if legacy in self._name_to_shard:
                packed_name = legacy
        if packed_name is None:
            raise KeyError(
                "NVFP4 packed tensor not found: %s or %s"
                % (base + self.WEIGHT_SUFFIX, base + self.PACKED_SUFFIX))
        if scale_name not in self._name_to_shard:
            raise KeyError("NVFP4 scale tensor not found: %s" % scale_name)

        packed = np.ascontiguousarray(self._read_raw(packed_name)).view(np.uint8)
        scales = np.ascontiguousarray(self._read_raw(scale_name)).view(np.uint8)
        if packed.ndim != 2 or scales.ndim != 2:
            raise ValueError(
                "expected 2D packed and scales, got %s and %s for %s"
                % (packed.shape, scales.shape, base))
        rows, packed_cols = packed.shape
        cols = packed_cols * 2
        if scales.shape != (rows, cols // self.group_size):
            raise ValueError(
                "scale shape %s does not match rows=%d cols/group=%d for %s"
                % (scales.shape, rows, cols // self.group_size, base))

        global_scale = 1.0
        if scale2_name in self._name_to_shard:
            raw = self._read_raw(scale2_name)
            arr = np.asarray(raw).astype(np.float32).reshape(-1)
            if arr.size != 1:
                raise ValueError(
                    "weight_scale_2 expected scalar, got shape %s for %s"
                    % (arr.shape, base))
            global_scale = float(arr[0])

        input_scale = 1.0
        if input_scale_name in self._name_to_shard:
            raw = self._read_raw(input_scale_name)
            arr = np.asarray(raw).astype(np.float32).reshape(-1)
            if arr.size != 1:
                raise ValueError(
                    "input_scale expected scalar, got shape %s for %s"
                    % (arr.shape, base))
            input_scale = float(arr[0])

        return NvFp4Tensor(
            name=base,
            shape=(rows, cols),
            packed=packed,
            scales=scales,
            group_size=self.group_size,
            global_scale=global_scale,
            input_scale=input_scale,
        )

    def read_bf16(self, name):
        """Return tensor as uint16 (bf16 bits) ndarray. Up-casts f16/f32 to bf16."""
        raw = self._read_raw(name)
        info = self._tensor_info(name)
        dtype_str = info['dtype'].upper()
        if dtype_str in ('BF16', 'BFLOAT16'):
            return np.ascontiguousarray(raw).view(np.uint16)
        if dtype_str in ('F16', 'FLOAT16'):
            return _f32_to_bf16_bits(np.asarray(raw, dtype=np.float32))
        if dtype_str in ('F32', 'FLOAT32'):
            return _f32_to_bf16_bits(np.asarray(raw, dtype=np.float32))
        raise ValueError(
            "read_bf16 called on non-float tensor %s (dtype=%s)" % (name, dtype_str))


def _f32_to_bf16_bits(a_f32):
    """Round-to-nearest-even f32 -> bf16 bits (uint16), per agent 3 contract."""
    u32 = a_f32.astype(np.float32).view(np.uint32)
    # RNE: add 0x7FFF + ((u32 >> 16) & 1) before truncation
    lsb = (u32 >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u32 + bias) >> 16
    return rounded.astype(np.uint16).reshape(a_f32.shape)


# ---- host-side dequant via Zig ---------------------------------------------

def dequant_nvfp4_to_int8_cpu(t, n_threads=0):
    """NVFP4 -> int8 per-row via librvllm_zig SIMD kernels.

    Returns (int8_matrix [rows, cols], row_scales_f32 [rows]).
    """
    lib = _load_zig()
    rows, cols = t.shape
    packed = np.ascontiguousarray(t.packed, dtype=np.uint8)
    scales = np.ascontiguousarray(t.scales, dtype=np.uint8)
    out = np.empty((rows, cols), dtype=np.int8)
    row_scales = np.empty((rows,), dtype=np.float32)
    rc = lib.rvllm_nvfp4_to_int8(
        packed.ctypes.data, ctypes.c_size_t(packed.nbytes),
        scales.ctypes.data, ctypes.c_size_t(scales.nbytes),
        ctypes.c_size_t(rows), ctypes.c_size_t(cols),
        out.ctypes.data,
        row_scales.ctypes.data,
        ctypes.c_size_t(int(n_threads)),
    )
    if rc != 0:
        raise RuntimeError(
            "rvllm_nvfp4_to_int8 failed with rc=%d for tensor %s" % (rc, t.name))
    # Fold per-tensor modelopt global scale into the per-row scales.
    if t.global_scale != 1.0:
        row_scales = (row_scales * np.float32(t.global_scale)).astype(np.float32)
    return out, row_scales


def dequant_nvfp4_to_bf16_cpu(t, n_threads=0):
    """NVFP4 -> bf16 via librvllm_zig SIMD kernels.

    Returns an ndarray of shape (rows, cols) and dtype ml_dtypes.bfloat16
    (when available) backed by the same uint16 buffer.
    """
    lib = _load_zig()
    rows, cols = t.shape
    packed = np.ascontiguousarray(t.packed, dtype=np.uint8)
    scales = np.ascontiguousarray(t.scales, dtype=np.uint8)
    out_u16 = np.empty((rows, cols), dtype=np.uint16)
    rc = lib.rvllm_nvfp4_to_bf16(
        packed.ctypes.data, ctypes.c_size_t(packed.nbytes),
        scales.ctypes.data, ctypes.c_size_t(scales.nbytes),
        ctypes.c_size_t(rows), ctypes.c_size_t(cols),
        out_u16.ctypes.data,
        ctypes.c_size_t(int(n_threads)),
    )
    if rc != 0:
        raise RuntimeError(
            "rvllm_nvfp4_to_bf16 failed with rc=%d for tensor %s" % (rc, t.name))
    if t.global_scale != 1.0:
        if not HAVE_ML_DTYPES:
            raise RuntimeError(
                "ml_dtypes required to fold modelopt global_scale for %s" % t.name)
        bf16_view = out_u16.view(ml_dtypes.bfloat16)
        scaled_f32 = bf16_view.astype(np.float32) * np.float32(t.global_scale)
        return scaled_f32.astype(ml_dtypes.bfloat16)
    if HAVE_ML_DTYPES:
        return out_u16.view(ml_dtypes.bfloat16)
    return out_u16


# ---- smoke test -------------------------------------------------------------

def _smoke(model_dir):
    r = ModeloptSafetensorsReader(model_dir)
    names = r.list_tensors()
    print("model_dir:   %s" % model_dir)
    print("shards:      %d" % len(r._shard_paths))
    print("tensors:     %d" % len(names))
    print("ignore_pats: %s" % r.ignore_patterns)
    print("group_size:  %d" % r.group_size)

    first_nvfp4 = None
    first_bf16 = None
    for n in names:
        if n.endswith(r.PACKED_SUFFIX):
            base = n[:-len(r.PACKED_SUFFIX)]
            if first_nvfp4 is None and r.is_nvfp4(base):
                first_nvfp4 = base
        else:
            if first_bf16 is None and not n.endswith(r.SCALE_SUFFIX):
                info = r._tensor_info(n)
                dt = info['dtype'].upper()
                if dt in ('BF16', 'BFLOAT16', 'F16', 'FLOAT16', 'F32', 'FLOAT32'):
                    first_bf16 = n
        if first_nvfp4 and first_bf16:
            break

    if first_nvfp4 is not None:
        t = r.read_nvfp4(first_nvfp4)
        print("first NVFP4: name=%s shape=%s packed=%s scales=%s"
              % (t.name, t.shape, t.packed.shape, t.scales.shape))
    else:
        print("first NVFP4: <none found>")

    if first_bf16 is not None:
        info = r._tensor_info(first_bf16)
        print("first bf16:  name=%s shape=%s dtype=%s"
              % (first_bf16, tuple(info['shape']), info['dtype']))
    else:
        print("first bf16:  <none found>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    args = ap.parse_args()
    _smoke(args.model_dir)


if __name__ == '__main__':
    main()
