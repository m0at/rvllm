const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // Root library module. Aggregates all submodules used by the static lib,
    // shared lib, tests, and binary driver.
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library (pure Zig, no libc).
    const lib = b.addLibrary(.{
        .name = "rvllm_zig",
        .root_module = lib_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // Shared library exposing the C ABI surface to Python ctypes.
    const shared_mod = b.createModule(.{
        .root_source_file = b.path("src/c_abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    const shared_lib = b.addLibrary(.{
        .name = "rvllm_zig",
        .root_module = shared_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(shared_lib);

    // Unit tests against the root module.
    const test_exe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Dequant binary driver: safetensors shard iteration + NVFP4 -> int8/bf16.
    const dequant_exe = b.addExecutable(.{
        .name = "rvllm_dequant",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/rvllm_dequant_bin.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(dequant_exe);
    const run_dequant = b.addRunArtifact(dequant_exe);
    if (b.args) |args| run_dequant.addArgs(args);
    const dequant_step = b.step("rvllm_dequant", "Run rvllm_dequant driver");
    dequant_step.dependOn(&run_dequant.step);

    // Benchmark (existing).
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
