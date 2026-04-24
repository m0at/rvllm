// Headless swarm driver. Runs the controller with no UI, dispatches a
// configurable set of goals against the mock backend, and exits when
// every goal has reached a terminal status.
//
// Usage:
//   swarm-cli --goal "port FP8 tile"
//   swarm-cli --goals-file goals.txt --timeout-s 120

use std::time::{Duration, Instant};

use rvllm_swarm_egui::controller::spawn;
use rvllm_swarm_egui::detect_repo_root;
use rvllm_swarm_egui::ipc::Cmd;
use rvllm_swarm_egui::state::{BackendKind, ExecutionMode, GoalStatus};

struct Args {
    goals: Vec<String>,
    broadcasts: Vec<String>,
    timeout_s: u64,
    n_live: Option<usize>,
    mode: Option<ExecutionMode>,
    decode_batch_target: Option<usize>,
}

fn parse_args() -> Args {
    let mut goals = Vec::<String>::new();
    let mut broadcasts = Vec::<String>::new();
    let mut timeout_s: u64 = 120;
    let mut n_live: Option<usize> = None;
    let mut mode: Option<ExecutionMode> = None;
    let mut decode_batch_target: Option<usize> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--goal" => {
                if let Some(v) = it.next() {
                    goals.push(v);
                }
            }
            "--broadcast" => {
                if let Some(v) = it.next() {
                    broadcasts.push(v);
                }
            }
            "--goals-file" => {
                if let Some(p) = it.next() {
                    if let Ok(s) = std::fs::read_to_string(&p) {
                        for line in s.lines() {
                            let t = line.trim();
                            if !t.is_empty() && !t.starts_with('#') {
                                goals.push(t.to_owned());
                            }
                        }
                    } else {
                        eprintln!("could not read --goals-file {p}");
                    }
                }
            }
            "--timeout-s" => {
                if let Some(v) = it.next() {
                    timeout_s = v.parse().unwrap_or(120);
                }
            }
            "--n-live" => {
                if let Some(v) = it.next() {
                    n_live = v.parse().ok();
                }
            }
            "--mode" => {
                if let Some(v) = it.next() {
                    mode = match v.as_str() {
                        "operator-30" | "operator" | "30" => Some(ExecutionMode::Operator30),
                        "saturator" | "saturate" | "throughput" | "512" => {
                            Some(ExecutionMode::Saturator)
                        }
                        _ => {
                            eprintln!("unknown --mode {v}");
                            print_usage();
                            std::process::exit(2);
                        }
                    };
                }
            }
            "--decode-batch" => {
                if let Some(v) = it.next() {
                    decode_batch_target = v.parse().ok();
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg {a}");
                print_usage();
                std::process::exit(2);
            }
        }
    }
    if goals.is_empty() && broadcasts.is_empty() {
        goals.push("write a haiku about worktrees".into());
    }
    Args {
        goals,
        broadcasts,
        timeout_s,
        n_live,
        mode,
        decode_batch_target,
    }
}

fn print_usage() {
    eprintln!(
        "swarm-cli: headless driver for rvllm-swarm-egui\n\
         \n\
         --goal <text>         submit a goal (repeatable)\n\
         --broadcast <text>    single-submit fanout to all 30 agents\n\
         --goals-file <path>   one goal per line\n\
         --timeout-s <N>       max wall seconds (default 120)\n\
         --n-live <N>          scheduler live slots\n\
         --mode <name>         operator-30 or saturator\n\
         --decode-batch <N>    target on-card decode batch (1..512)\n"
    );
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args = parse_args();
    let repo_root = detect_repo_root();
    let backend_kind = if cfg!(feature = "cuda") {
        BackendKind::Rvllm
    } else {
        BackendKind::Mock
    };
    let h = spawn(repo_root, backend_kind);

    if let Some(n) = args.n_live {
        let _ = h.cmd_tx.send(Cmd::AdjustScheduler { n_live: n });
    }
    if args.mode.is_some() || args.decode_batch_target.is_some() {
        let mode = args.mode.unwrap_or_else(|| {
            let g = h.state.read();
            g.settings.scheduler.execution_mode
        });
        let decode_batch_target = args
            .decode_batch_target
            .unwrap_or_else(|| mode.decode_batch_target());
        let _ = h.cmd_tx.send(Cmd::SetExecutionMode {
            mode,
            decode_batch_target,
        });
    }

    for g in &args.goals {
        let _ = h.cmd_tx.send(Cmd::SubmitGoal { text: g.clone() });
    }
    for g in &args.broadcasts {
        let _ = h.cmd_tx.send(Cmd::SubmitBroadcast { text: g.clone() });
    }

    let deadline = Instant::now() + Duration::from_secs(args.timeout_s);
    loop {
        if Instant::now() >= deadline {
            eprintln!("swarm-cli: timeout after {}s", args.timeout_s);
            std::process::exit(1);
        }
        let done = {
            let g = h.state.read();
            g.tasks.goals.values().all(|goal| {
                matches!(
                    goal.status,
                    GoalStatus::Done | GoalStatus::Failed | GoalStatus::Cancelled
                )
            }) && !g.tasks.goals.is_empty()
        };
        if done {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    // Final summary.
    let g = h.state.read();
    let counts = g.tasks.global_counts();
    println!(
        "{{\"goals\":{},\"tasks\":{{\"total\":{},\"done\":{},\"failed\":{}}}}}",
        g.tasks.goals.len(),
        counts.total,
        counts.done,
        counts.failed
    );
}
