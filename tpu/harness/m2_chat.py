"""MiniMax-M2 chat template + tokenizer wrapper.

Owner: agent 14 (M2 swarm). Loads tokenizer.json, prefers Zig BPE via ctypes
(agent 5's c_abi) when RVLLM_ZIG_LIB is set, otherwise uses the `tokenizers`
library. Applies the official MiniMax-M2 chat template (jinja from
tokenizer_config.json if present, hardcoded otherwise).
"""

import ctypes
import json
import os
import sys


BOS_ID = 1
EOS_ID = 2

DEFAULT_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "<|im_start|>assistant\n"
)


class _HFTokenizerAdapter:
    def __init__(self, tok):
        self._tok = tok
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(list(ids))


class _ZigBpeAdapter:
    def __init__(self, lib, handle, hf_fallback):
        self._lib = lib
        self._handle = handle
        self._hf = hf_fallback  # for decode, which Zig may not expose yet
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        cap = max(4096, len(text_bytes) * 4)
        out = (ctypes.c_uint32 * cap)()
        out_len = ctypes.c_size_t(0)
        rc = self._lib.rvllm_bpe_encode(
            self._handle,
            ctypes.c_char_p(text_bytes), ctypes.c_size_t(len(text_bytes)),
            out, ctypes.c_size_t(cap), ctypes.byref(out_len),
        )
        if rc != 0:
            raise RuntimeError(f"rvllm_bpe_encode failed rc={rc}")
        return [int(out[i]) for i in range(out_len.value)]

    def decode(self, ids):
        return self._hf.decode(list(ids))


def _try_zig_bpe(model_dir, hf_fallback):
    lib_path = os.environ.get("RVLLM_ZIG_LIB")
    if not lib_path:
        return None
    if not os.path.exists(lib_path):
        print(f"RVLLM_ZIG_LIB={lib_path} not found; using tokenizers lib",
              file=sys.stderr)
        return None
    try:
        lib = ctypes.CDLL(lib_path)
        if not hasattr(lib, "rvllm_bpe_init") or not hasattr(lib, "rvllm_bpe_encode"):
            print("zig lib missing rvllm_bpe_* exports; using tokenizers lib",
                  file=sys.stderr)
            return None
        lib.rvllm_bpe_init.restype = ctypes.c_void_p
        lib.rvllm_bpe_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.rvllm_bpe_encode.restype = ctypes.c_int32
        lib.rvllm_bpe_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        tok_path = os.path.join(model_dir, "tokenizer.json").encode("utf-8")
        handle = lib.rvllm_bpe_init(
            ctypes.c_char_p(tok_path), ctypes.c_size_t(len(tok_path)))
        if not handle:
            print("rvllm_bpe_init returned null; using tokenizers lib",
                  file=sys.stderr)
            return None
        print(f"using Zig BPE via {lib_path}", file=sys.stderr)
        return _ZigBpeAdapter(lib, handle, hf_fallback)
    except (OSError, AttributeError) as e:
        print(f"zig BPE init failed ({e}); using tokenizers lib",
              file=sys.stderr)
        return None


def load_tokenizer_m2(model_dir):
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"no tokenizer.json in {model_dir}")
    from tokenizers import Tokenizer
    hf = _HFTokenizerAdapter(Tokenizer.from_file(tok_path))

    zig = _try_zig_bpe(model_dir, hf)
    if zig is not None:
        return zig
    print("using HuggingFace tokenizers backend", file=sys.stderr)
    return hf


def _load_jinja_template(model_dir):
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    tpl = cfg.get("chat_template")
    if isinstance(tpl, list):
        for entry in tpl:
            if isinstance(entry, dict) and entry.get("name") == "default":
                return entry.get("template")
        return None
    return tpl


def _render_hardcoded(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def apply_chat_template(messages, model_dir=None):
    template = _load_jinja_template(model_dir) if model_dir else None
    if template is None:
        return _render_hardcoded(messages)
    try:
        import jinja2
    except ImportError:
        print("jinja2 not installed; using hardcoded template",
              file=sys.stderr)
        return _render_hardcoded(messages)
    env = jinja2.Environment(
        trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
    tmpl = env.from_string(template)
    return tmpl.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token="",
        eos_token="",
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="dir containing tokenizer.json")
    args = ap.parse_args()

    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    print("=== chat template output ===")
    rendered = apply_chat_template(sample_messages, model_dir=args.model_dir)
    print(rendered)
    print("=== end template ===")

    if args.model_dir is None:
        print("(no --model-dir given; skipping tokenizer smoke)")
        sys.exit(0)

    tok = load_tokenizer_m2(args.model_dir)
    print(f"bos_token_id={tok.bos_token_id} eos_token_id={tok.eos_token_id}")
    sample = "Hello, world!"
    ids = tok.encode(sample)
    print(f"encode({sample!r}) -> {ids}")
    back = tok.decode(ids)
    print(f"decode(ids) -> {back!r}")
    ids_rendered = tok.encode(rendered)
    print(f"rendered template token count: {len(ids_rendered)}")
