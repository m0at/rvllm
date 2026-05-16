// Enforces the crate DAG from v3/IMPL_PLAN.md section 1.1.
// Parses every crates/*/Cargo.toml and asserts only allowed rvllm-* edges exist.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

// (crate, allowed-rvllm-deps). Ordered bottom-up. Any edge not in this set fails.
fn allowed_deps() -> HashMap<&'static str, HashSet<&'static str>> {
    fn s(list: &[&'static str]) -> HashSet<&'static str> {
        list.iter().copied().collect()
    }
    let mut m = HashMap::new();
    m.insert("rvllm-core", s(&[]));
    m.insert("rvllm-mem", s(&["rvllm-core"]));
    m.insert("rvllm-kernels", s(&["rvllm-core", "rvllm-mem"]));
    m.insert(
        "rvllm-cutlass",
        s(&["rvllm-core", "rvllm-mem", "rvllm-kernels"]),
    );
    m.insert(
        "rvllm-attention",
        s(&["rvllm-core", "rvllm-mem", "rvllm-kernels"]),
    );
    m.insert(
        "rvllm-fused",
        s(&["rvllm-core", "rvllm-mem", "rvllm-kernels"]),
    );
    m.insert("rvllm-metadata", s(&["rvllm-core", "rvllm-mem"]));
    m.insert(
        "rvllm-graph",
        s(&["rvllm-core", "rvllm-mem", "rvllm-metadata"]),
    );
    m.insert("rvllm-loader", s(&["rvllm-core", "rvllm-mem"]));
    m.insert(
        "rvllm-sampling",
        s(&["rvllm-core", "rvllm-mem", "rvllm-fused"]),
    );
    m.insert(
        "rvllm-runtime",
        s(&[
            "rvllm-core",
            "rvllm-mem",
            "rvllm-kernels",
            "rvllm-cutlass",
            "rvllm-attention",
            "rvllm-fused",
            "rvllm-metadata",
            "rvllm-graph",
            "rvllm-loader",
            "rvllm-sampling",
        ]),
    );
    m.insert("rvllm-serve", s(&["rvllm-core", "rvllm-runtime"]));
    m.insert("rvllm-bench", s(&["rvllm-core", "rvllm-runtime"]));
    m.insert("rvllm-deploy", s(&["rvllm-core"]));
    m.insert("rvllm-invariants", s(&[]));
    m
}

fn crates_dir() -> PathBuf {
    // tests run from crate root (rvllm-invariants); crates are siblings.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn parse_rvllm_deps(cargo_toml: &str) -> HashSet<String> {
    // Catches three Cargo.toml forms:
    //   [dependencies]
    //   rvllm-foo = ...            (flat table entry)
    //   [dependencies.rvllm-foo]   (inline subtable)
    //   [dev-dependencies.rvllm-foo]
    // Avoids pulling in a TOML crate so this test has zero deps.
    let mut deps = HashSet::new();
    let mut in_deps = false;
    for line in cargo_toml.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            // Tables: [dependencies] / [dev-dependencies] / [build-dependencies]
            // Subtables: [dependencies.NAME] / [dev-dependencies.NAME] / [build-dependencies.NAME]
            let (section, subkey) = match rest.split_once('.') {
                Some((s, k)) => (s, Some(k)),
                None => (rest, None),
            };
            let is_deps = matches!(
                section,
                "dependencies" | "dev-dependencies" | "build-dependencies"
            );
            in_deps = is_deps && subkey.is_none();
            if is_deps {
                if let Some(k) = subkey {
                    if k.starts_with("rvllm-") {
                        deps.insert(k.to_string());
                    }
                }
            }
            continue;
        }
        if !in_deps {
            continue;
        }
        if let Some(eq) = t.find('=') {
            let name = t[..eq].trim();
            if name.starts_with("rvllm-") {
                deps.insert(name.to_string());
            }
        }
    }
    deps
}

#[test]
fn dag_matches_impl_plan() {
    let allowed = allowed_deps();
    let dir = crates_dir();
    let mut violations = Vec::new();

    for entry in fs::read_dir(&dir).expect("read crates/") {
        let entry = entry.expect("read entry");
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let cargo_toml = path.join("Cargo.toml");
        if !cargo_toml.exists() {
            continue;
        }
        let body = fs::read_to_string(&cargo_toml).expect("read Cargo.toml");
        let deps = parse_rvllm_deps(&body);

        let Some(allowed_set) = allowed.get(name.as_str()) else {
            violations.push(format!("unknown crate '{name}' (add to allowed_deps)"));
            continue;
        };
        for dep in &deps {
            if !allowed_set.contains(dep.as_str()) {
                violations.push(format!(
                    "crate '{name}' depends on '{dep}' (not permitted by DAG)"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "DAG violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_declared_crate_exists() {
    let allowed = allowed_deps();
    let dir = crates_dir();
    let mut missing = Vec::new();
    for name in allowed.keys() {
        if !dir.join(name).join("Cargo.toml").exists() {
            missing.push(*name);
        }
    }
    assert!(missing.is_empty(), "declared but missing: {missing:?}");
}
