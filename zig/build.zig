const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // Static library (pure Zig, no libc).
    const lib = b.addStaticLibrary(.{
        .name = "rvllm_zig",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Shared library exposing the C ABI surface to Python ctypes.
    const shared_lib = b.addSharedLibrary(.{
        .name = "rvllm_zig",
        .root_source_file = b.path("src/c_abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(shared_lib);

    // Unit tests against the root module.
    const test_exe = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Dequant binary driver: safetensors shard iteration + NVFP4 -> int8/bf16.
    const dequant_exe = b.addExecutable(.{
        .name = "rvllm_dequant",
        .root_source_file = b.path("src/rvllm_dequant_bin.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    b.installArtifact(dequant_exe);
    const run_dequant = b.addRunArtifact(dequant_exe);
    if (b.args) |args| run_dequant.addArgs(args);
    const dequant_step = b.step("rvllm_dequant", "Run rvllm_dequant driver");
    dequant_step.dependOn(&run_dequant.step);

    // Benchmark (existing).
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
