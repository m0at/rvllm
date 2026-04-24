#!/usr/bin/env python3
"""Benchmark torch.compile DecomposeK on Gemma4 31B decode GEMM shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    k: int
    n: int
    description: str


SHAPES: dict[str, ShapeSpec] = {
    "o_sliding": ShapeSpec(
        name="o_sliding",
        k=8192,
        n=5376,
        description="Gemma4 sliding-layer O-proj decode",
    ),
    "o_global": ShapeSpec(
        name="o_global",
        k=16384,
        n=5376,
        description="Gemma4 global-layer O-proj decode",
    ),
    "down": ShapeSpec(
        name="down",
        k=21504,
        n=5376,
        description="Gemma4 FFN down-proj decode",
    ),
}


@dataclass
class TrialResult:
    impl: str
    variant: str
    workload: str
    shape: str
    batch: int
    k: int
    n: int
    splits: int | None
    dtype: str
    device: str
    compile_mode: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    max_abs_diff: float | None
    max_rel_diff: float | None
    l2_rel_diff: float | None


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        if device == "cpu":
            return torch.float32
        return torch.float16
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[requested]


def resolve_compile_mode(requested: str, device: str) -> str | None:
    if requested == "none":
        return None
    if requested == "auto":
        return "max-autotune" if device == "cuda" else "default"
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_baseline(workload: str):
    if workload == "gemm_only":
        def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
            return x @ w_kn
        return fn
    if workload == "epilogue":
        def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
            return (x @ w_kn) * chscale + residual
        return fn
    raise ValueError(f"unknown workload: {workload}")


def make_reference(workload: str):
    if workload == "gemm_only":
        def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
            return x.float() @ w_kn.float()
        return fn
    if workload == "epilogue":
        def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
            return (x.float() @ w_kn.float()) * chscale.float() + residual.float()
        return fn
    raise ValueError(f"unknown workload: {workload}")


def make_decomposek(workload: str, splits: int, variant: str):
    if variant == "native":
        if workload == "gemm_only":
            def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
                k_per = x.shape[1] // splits
                x_bmm = x.unflatten(1, (splits, k_per)).transpose(0, 1)
                w_bmm = w_kn.unflatten(0, (splits, k_per))
                return torch.bmm(x_bmm, w_bmm).sum(dim=0)
            return fn
        if workload == "epilogue":
            def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
                k_per = x.shape[1] // splits
                x_bmm = x.unflatten(1, (splits, k_per)).transpose(0, 1)
                w_bmm = w_kn.unflatten(0, (splits, k_per))
                return torch.bmm(x_bmm, w_bmm).sum(dim=0) * chscale + residual
            return fn
    if variant == "sum_fp32":
        if workload == "gemm_only":
            def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
                k_per = x.shape[1] // splits
                x_bmm = x.unflatten(1, (splits, k_per)).transpose(0, 1)
                w_bmm = w_kn.unflatten(0, (splits, k_per))
                return torch.bmm(x_bmm, w_bmm).float().sum(dim=0)
            return fn
        if workload == "epilogue":
            def fn(x: torch.Tensor, w_kn: torch.Tensor, residual: torch.Tensor, chscale: torch.Tensor) -> torch.Tensor:
                k_per = x.shape[1] // splits
                x_bmm = x.unflatten(1, (splits, k_per)).transpose(0, 1)
                w_bmm = w_kn.unflatten(0, (splits, k_per))
                reduced = torch.bmm(x_bmm, w_bmm).float().sum(dim=0)
                return reduced * chscale.float() + residual.float()
            return fn
    raise ValueError(f"unknown workload/variant: {workload}/{variant}")


def compile_fn(fn, compile_mode: str | None):
    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode)


def benchmark_fn(
    fn,
    x: torch.Tensor,
    w_kn: torch.Tensor,
    residual: torch.Tensor,
    chscale: torch.Tensor,
    device: str,
    warmup: int,
    iters: int,
) -> tuple[list[float], torch.Tensor]:
    out = fn(x, w_kn, residual, chscale)
    synchronize(device)
    for _ in range(warmup):
        out = fn(x, w_kn, residual, chscale)
        synchronize(device)
    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(x, w_kn, residual, chscale)
        synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms, out


def summarize_result(
    *,
    impl: str,
    variant: str,
    workload: str,
    spec: ShapeSpec,
    batch: int,
    splits: int | None,
    dtype: torch.dtype,
    device: str,
    compile_mode: str | None,
    times_ms: list[float],
    output: torch.Tensor,
    reference: torch.Tensor | None,
) -> TrialResult:
    max_abs_diff = None
    max_rel_diff = None
    l2_rel_diff = None
    if reference is not None:
        diff = (output - reference).abs().float()
        max_abs_diff = float(diff.max().item())
        ref_abs = reference.abs().float().clamp_min(1e-2)
        max_rel_diff = float((diff / ref_abs).max().item())
        l2_rel_diff = float(diff.norm().item() / reference.float().norm().clamp_min(1e-6).item())
    return TrialResult(
        impl=impl,
        variant=variant,
        workload=workload,
        shape=spec.name,
        batch=batch,
        k=spec.k,
        n=spec.n,
        splits=splits,
        dtype=str(dtype).replace("torch.", ""),
        device=device,
        compile_mode=compile_mode or "none",
        median_ms=round(statistics.median(times_ms), 4),
        mean_ms=round(statistics.fmean(times_ms), 4),
        min_ms=round(min(times_ms), 4),
        max_ms=round(max(times_ms), 4),
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        l2_rel_diff=l2_rel_diff,
    )


def run_case(
    *,
    spec: ShapeSpec,
    batch: int,
    workload: str,
    splits_list: list[int],
    decomposek_modes: list[str],
    device: str,
    dtype: torch.dtype,
    compile_mode: str | None,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    scale = 1.0 / math.sqrt(spec.k)
    x = torch.randn(batch, spec.k, device=device, dtype=dtype) * scale
    w_kn = torch.randn(spec.k, spec.n, device=device, dtype=dtype) * scale
    residual = torch.randn(batch, spec.n, device=device, dtype=dtype) * scale
    chscale = torch.randn(spec.n, device=device, dtype=dtype).abs_() + 0.5

    with torch.no_grad():
        reference_out = make_reference(workload)(x, w_kn, residual, chscale)
        synchronize(device)
        baseline = compile_fn(make_baseline(workload), compile_mode)
        baseline_times, baseline_out = benchmark_fn(
            baseline, x, w_kn, residual, chscale, device, warmup, iters
        )
        baseline_result = summarize_result(
            impl="baseline",
            variant="native",
            workload=workload,
            spec=spec,
            batch=batch,
            splits=None,
            dtype=dtype,
            device=device,
            compile_mode=compile_mode,
            times_ms=baseline_times,
            output=baseline_out,
            reference=reference_out,
        )

        decomposek_results: list[TrialResult] = []
        for variant in decomposek_modes:
            for splits in splits_list:
                if spec.k % splits != 0:
                    continue
                decomposek = compile_fn(make_decomposek(workload, splits, variant), compile_mode)
                decomp_times, decomp_out = benchmark_fn(
                    decomposek, x, w_kn, residual, chscale, device, warmup, iters
                )
                decomposek_results.append(
                    summarize_result(
                        impl="decomposek",
                        variant=variant,
                        workload=workload,
                        spec=spec,
                        batch=batch,
                        splits=splits,
                        dtype=dtype,
                        device=device,
                        compile_mode=compile_mode,
                        times_ms=decomp_times,
                        output=decomp_out,
                        reference=reference_out,
                    )
                )

    best = min(decomposek_results, key=lambda item: item.median_ms) if decomposek_results else None
    speedup = None
    if best is not None:
        speedup = round(baseline_result.median_ms / best.median_ms, 4)

    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "shape": asdict(spec),
        "batch": batch,
        "workload": workload,
        "reference": "float32_eager",
        "baseline": asdict(baseline_result),
        "best_decomposek": asdict(best) if best is not None else None,
        "speedup_vs_baseline": speedup,
        "all_decomposek": [asdict(item) for item in decomposek_results],
    }


def print_summary(case: dict[str, object]) -> None:
    shape = case["shape"]
    baseline = case["baseline"]
    best = case["best_decomposek"]
    speedup = case["speedup_vs_baseline"]
    print(
        f"{shape['name']:>10}  batch={case['batch']:<2}  workload={case['workload']:<9}  "
        f"baseline={baseline['median_ms']:>8.4f} ms",
        end="",
    )
    if best is None:
        print("  no valid split")
        return
    print(
        f"  best_variant={best['variant']:<8} split={best['splits']:<3}  "
        f"decomposek={best['median_ms']:>8.4f} ms  "
        f"speedup={speedup:>6.3f}x  "
        f"max_abs_diff={best['max_abs_diff']:.4e}  "
        f"max_rel_diff={best['max_rel_diff']:.4e}  "
        f"l2_rel_diff={best['l2_rel_diff']:.4e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="o_sliding,o_global,down",
        help=f"comma-separated shape presets: {','.join(SHAPES)}",
    )
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--splits", default="2,4,8,16,32")
    parser.add_argument(
        "--decomposek-modes",
        default="native,sum_fp32",
        help="comma-separated: native, sum_fp32",
    )
    parser.add_argument(
        "--workloads",
        default="epilogue",
        help="comma-separated: epilogue, gemm_only",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["auto", "default", "reduce-overhead", "max-autotune", "none"],
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    compile_mode = resolve_compile_mode(args.compile_mode, device)
    shape_names = parse_csv_strings(args.shapes)
    batches = parse_csv_ints(args.batches)
    splits_list = parse_csv_ints(args.splits)
    decomposek_modes = parse_csv_strings(args.decomposek_modes)
    workloads = parse_csv_strings(args.workloads)

    for shape_name in shape_names:
        if shape_name not in SHAPES:
            raise SystemExit(f"unknown shape: {shape_name}")

    print(
        f"device={device} dtype={str(dtype).replace('torch.', '')} "
        f"compile_mode={compile_mode or 'none'} workloads={','.join(workloads)}"
    )
    print("Shapes:")
    for shape_name in shape_names:
        spec = SHAPES[shape_name]
        print(f"  {spec.name}: Mx{spec.k} @ {spec.k}x{spec.n}  ({spec.description})")
    print("")

    cases: list[dict[str, object]] = []
    for workload in workloads:
        for shape_name in shape_names:
            spec = SHAPES[shape_name]
            for batch in batches:
                case = run_case(
                    spec=spec,
                    batch=batch,
                    workload=workload,
                    splits_list=splits_list,
                    decomposek_modes=decomposek_modes,
                    device=device,
                    dtype=dtype,
                    compile_mode=compile_mode,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                cases.append(case)
                print_summary(case)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "compile_mode": compile_mode or "none",
            "decomposek_modes": decomposek_modes,
            "warmup": args.warmup,
            "iters": args.iters,
            "cases": cases,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
