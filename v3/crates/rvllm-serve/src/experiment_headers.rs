#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentLabel {
    pub key: &'static str,
    pub header_name: &'static str,
    pub value: String,
}

struct LabelSpec {
    key: &'static str,
    header_name: &'static str,
    env_names: &'static [&'static str],
}

const LABEL_SPECS: &[LabelSpec] = &[
    LabelSpec {
        key: "weight_path",
        header_name: "x-rvllm-experiment-weight-path",
        env_names: &[
            "RVLLM_EXPERIMENT_WEIGHT",
            "RVLLM_EXPERIMENT_QUANT_PATH",
            "RVLLM_QUANT_PATH",
            "RVLLM_W4A8",
            "RVLLM_AWQ_METADATA_ONLY",
        ],
    },
    LabelSpec {
        key: "kv_path",
        header_name: "x-rvllm-experiment-kv-path",
        env_names: &[
            "RVLLM_EXPERIMENT_KV",
            "RVLLM_EXPERIMENT_KV_PATH",
            "RVLLM_KV_PATH",
            "RVLLM_ROTORQUANT",
            "RVLLM_F16_KV",
        ],
    },
    LabelSpec {
        key: "attention_path",
        header_name: "x-rvllm-experiment-attention-path",
        env_names: &[
            "RVLLM_EXPERIMENT_ATTENTION",
            "RVLLM_EXPERIMENT_ATTENTION_PATH",
            "RVLLM_ATTENTION_PATH",
            "RVLLM_FA3",
            "RVLLM_FA2_FALLBACK",
        ],
    },
    LabelSpec {
        key: "architecture_policy",
        header_name: "x-rvllm-experiment-architecture-policy",
        env_names: &[
            "RVLLM_EXPERIMENT_ARCH",
            "RVLLM_EXPERIMENT_ARCHITECTURE_POLICY",
            "RVLLM_ARCHITECTURE_POLICY",
            "RVLLM_EXPERIMENT_ARCH_POLICY",
            "RVLLM_ARCH_POLICY",
            "RVLLM_FORCE_SM75_COMPAT",
            "RVLLM_FORCE_HOPPER",
        ],
    },
    LabelSpec {
        key: "validation_mode",
        header_name: "x-rvllm-experiment-validation-mode",
        env_names: &["RVLLM_EXPERIMENT_VALIDATION"],
    },
    LabelSpec {
        key: "revision",
        header_name: "x-rvllm-experiment-revision",
        env_names: &[
            "RVLLM_EXPERIMENT_REVISION",
            "RVLLM_GIT_REVISION",
            "RVLLM_REVISION",
            "GIT_REVISION",
            "REVISION",
        ],
    },
];

impl ExperimentLabel {
    pub fn header_pair(&self) -> (&'static str, String) {
        (self.header_name, header_safe_value(&self.value))
    }

    pub fn metadata_pair(&self) -> String {
        format!("{}={}", self.key, quoted_value(&self.value))
    }
}

pub fn from_env() -> Vec<ExperimentLabel> {
    labels_from(|name| std::env::var(name).ok())
}

pub fn header_pairs(labels: &[ExperimentLabel]) -> Vec<(&'static str, String)> {
    labels.iter().map(ExperimentLabel::header_pair).collect()
}

pub fn metadata_value(labels: &[ExperimentLabel]) -> String {
    labels
        .iter()
        .map(ExperimentLabel::metadata_pair)
        .collect::<Vec<_>>()
        .join("; ")
}

fn labels_from(mut get_env: impl FnMut(&str) -> Option<String>) -> Vec<ExperimentLabel> {
    LABEL_SPECS
        .iter()
        .filter_map(|spec| {
            spec.env_names.iter().find_map(|name| {
                get_env(name).and_then(|value| {
                    let value = clean_env_value(&value)?;
                    Some(ExperimentLabel {
                        key: spec.key,
                        header_name: spec.header_name,
                        value,
                    })
                })
            })
        })
        .collect()
}

fn clean_env_value(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| header_safe_value(value))
}

fn header_safe_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch == '\t' || ch.is_control() {
                ' '
            } else {
                ch
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn quoted_value(value: &str) -> String {
    let value = header_safe_value(value);
    if value.bytes().all(is_token_byte) {
        return value;
    }

    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for ch in value.chars() {
        if ch == '\\' || ch == '"' {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}

fn is_token_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
            | b'!'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
            | b'/'
            | b':'
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn from_env_emits_known_labels_in_stable_order() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_label_env();
        std::env::set_var("RVLLM_EXPERIMENT_WEIGHT", "w4a8-awq");
        std::env::set_var("RVLLM_EXPERIMENT_KV", "fp8");
        std::env::set_var("RVLLM_EXPERIMENT_ATTENTION", "fa3");
        std::env::set_var("RVLLM_EXPERIMENT_ARCH", "force-hopper");
        std::env::set_var("RVLLM_EXPERIMENT_VALIDATION", "chat");
        std::env::set_var("RVLLM_EXPERIMENT_REVISION", "abc123");

        let labels = from_env();

        assert_eq!(
            labels,
            vec![
                ExperimentLabel {
                    key: "weight_path",
                    header_name: "x-rvllm-experiment-weight-path",
                    value: "w4a8-awq".into(),
                },
                ExperimentLabel {
                    key: "kv_path",
                    header_name: "x-rvllm-experiment-kv-path",
                    value: "fp8".into(),
                },
                ExperimentLabel {
                    key: "attention_path",
                    header_name: "x-rvllm-experiment-attention-path",
                    value: "fa3".into(),
                },
                ExperimentLabel {
                    key: "architecture_policy",
                    header_name: "x-rvllm-experiment-architecture-policy",
                    value: "force-hopper".into(),
                },
                ExperimentLabel {
                    key: "validation_mode",
                    header_name: "x-rvllm-experiment-validation-mode",
                    value: "chat".into(),
                },
                ExperimentLabel {
                    key: "revision",
                    header_name: "x-rvllm-experiment-revision",
                    value: "abc123".into(),
                },
            ]
        );
        clear_label_env();
    }

    #[test]
    fn from_env_uses_aliases_and_skips_empty_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_label_env();
        std::env::set_var("RVLLM_QUANT_PATH", " ");
        std::env::set_var("RVLLM_KV_PATH", "paged");
        std::env::set_var("RVLLM_ATTENTION_PATH", "flash");
        std::env::set_var("RVLLM_ARCH_POLICY", "policy-a");
        std::env::set_var("REVISION", "def456");

        let labels = from_env();

        assert_eq!(
            labels.iter().map(|label| label.key).collect::<Vec<_>>(),
            vec![
                "kv_path",
                "attention_path",
                "architecture_policy",
                "revision"
            ]
        );
        assert_eq!(labels[0].value, "paged");
        assert_eq!(labels[3].value, "def456");
        clear_label_env();
    }

    #[test]
    fn explicit_env_names_win_over_aliases() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_label_env();
        std::env::set_var("RVLLM_EXPERIMENT_REVISION", "new");
        std::env::set_var("REVISION", "old");
        std::env::set_var("RVLLM_EXPERIMENT_ARCH", "explicit");
        std::env::set_var("RVLLM_ARCH_POLICY", "alias");

        let labels = from_env();

        assert_eq!(
            labels,
            vec![
                ExperimentLabel {
                    key: "architecture_policy",
                    header_name: "x-rvllm-experiment-architecture-policy",
                    value: "explicit".into(),
                },
                ExperimentLabel {
                    key: "revision",
                    header_name: "x-rvllm-experiment-revision",
                    value: "new".into(),
                },
            ]
        );
        clear_label_env();
    }

    #[test]
    fn formatting_helpers_clean_header_values() {
        let label = ExperimentLabel {
            key: "weight_path",
            header_name: "x-rvllm-experiment-weight-path",
            value: " awq\r\nx-bad: yes\tv2 ".into(),
        };

        assert_eq!(
            label.header_pair(),
            ("x-rvllm-experiment-weight-path", "awq x-bad: yes v2".into())
        );
        assert_eq!(label.metadata_pair(), "weight_path=\"awq x-bad: yes v2\"");
    }

    #[test]
    fn metadata_value_joins_labels() {
        let labels = vec![
            ExperimentLabel {
                key: "kv_path",
                header_name: "x-rvllm-experiment-kv-path",
                value: "fp8-kv".into(),
            },
            ExperimentLabel {
                key: "revision",
                header_name: "x-rvllm-experiment-revision",
                value: "abc 123".into(),
            },
        ];

        assert_eq!(
            metadata_value(&labels),
            "kv_path=fp8-kv; revision=\"abc 123\""
        );
        assert_eq!(
            header_pairs(&labels),
            vec![
                ("x-rvllm-experiment-kv-path", "fp8-kv".into()),
                ("x-rvllm-experiment-revision", "abc 123".into()),
            ]
        );
    }

    fn clear_label_env() {
        for spec in LABEL_SPECS {
            for name in spec.env_names {
                std::env::remove_var(name);
            }
        }
    }
}
