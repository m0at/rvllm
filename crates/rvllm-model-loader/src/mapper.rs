/// Maps HuggingFace weight names to an internal naming convention.
#[derive(Debug)]
pub struct WeightMapper {
    model_type: String,
}

impl WeightMapper {
    pub fn new(model_type: &str) -> Self {
        Self {
            model_type: model_type.to_string(),
        }
    }

    /// Remap a HuggingFace weight name to the internal convention.
    ///
    /// The mapping normalizes common patterns:
    /// - `model.layers.N.` -> `layers.N.`
    /// - `model.embed_tokens` -> `embed_tokens`
    /// - `model.norm` -> `norm`
    /// - `lm_head` stays as-is
    /// - Attention projection renaming per model type
    pub fn map_name(&self, hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Strip common `model.` prefix used in HF checkpoints
        if name.starts_with("model.") {
            name = name["model.".len()..].to_string();
        }

        // Model-specific attention head remapping
        match self.model_type.as_str() {
            "gemma" | "gemma2" => {
                name = name.replace("self_attn.q_proj", "attn.q");
                name = name.replace("self_attn.k_proj", "attn.k");
                name = name.replace("self_attn.v_proj", "attn.v");
                name = name.replace("self_attn.o_proj", "attn.o");
                name = name.replace("mlp.gate_proj", "ffn.gate");
                name = name.replace("mlp.up_proj", "ffn.up");
                name = name.replace("mlp.down_proj", "ffn.down");
                name = name.replace("input_layernorm", "attn_norm");
                name = name.replace("post_attention_layernorm", "post_attn_norm");
                name = name.replace("pre_feedforward_layernorm", "pre_ffn_norm");
                name = name.replace("post_feedforward_layernorm", "post_ffn_norm");
            }
            "llama" | "mistral" | "qwen2" => {
                name = name.replace("self_attn.q_proj", "attn.q");
                name = name.replace("self_attn.k_proj", "attn.k");
                name = name.replace("self_attn.v_proj", "attn.v");
                name = name.replace("self_attn.o_proj", "attn.o");
                name = name.replace("mlp.gate_proj", "ffn.gate");
                name = name.replace("mlp.up_proj", "ffn.up");
                name = name.replace("mlp.down_proj", "ffn.down");
                name = name.replace("input_layernorm", "attn_norm");
                name = name.replace("post_attention_layernorm", "ffn_norm");
            }
            "phi" | "phi3" => {
                // Phi-2 uses `ln` for the shared layernorm; Phi-3 uses
                // `input_layernorm` / `post_attention_layernorm` like Llama.
                // Attention projections follow the standard naming.
                name = name.replace("self_attn.q_proj", "attn.q");
                name = name.replace("self_attn.k_proj", "attn.k");
                name = name.replace("self_attn.v_proj", "attn.v");
                name = name.replace("self_attn.o_proj", "attn.o");
                name = name.replace("self_attn.q_layernorm", "attn.q_ln");
                name = name.replace("self_attn.k_layernorm", "attn.k_ln");
                name = name.replace("mlp.gate_proj", "ffn.gate");
                name = name.replace("mlp.gate_up_proj", "ffn.gate_up");
                name = name.replace("mlp.up_proj", "ffn.up");
                name = name.replace("mlp.down_proj", "ffn.down");
                name = name.replace("input_layernorm", "attn_norm");
                name = name.replace("post_attention_layernorm", "ffn_norm");
                // Phi-2 shared layernorm
                name = name.replace(".ln.", ".shared_norm.");
                name = name.replace("final_layernorm", "norm");
            }
            "gpt2" | "gpt_neox" => {
                name = name.replace("attn.c_attn", "attn.qkv");
                name = name.replace("attn.c_proj", "attn.o");
                name = name.replace("mlp.c_fc", "ffn.up");
                name = name.replace("mlp.c_proj", "ffn.down");
                name = name.replace("ln_1", "attn_norm");
                name = name.replace("ln_2", "ffn_norm");
            }
            _ => {
                // Unknown model type: pass through with only the model. prefix stripped
            }
        }

        name
    }

    pub fn model_type(&self) -> &str {
        &self.model_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_mapping() {
        let mapper = WeightMapper::new("llama");
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.5.mlp.gate_proj.weight"),
            "layers.5.ffn.gate.weight"
        );
        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(mapper.map_name("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn gpt2_mapping() {
        let mapper = WeightMapper::new("gpt2");
        assert_eq!(
            mapper.map_name("model.layers.0.attn.c_attn.weight"),
            "layers.0.attn.qkv.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.ln_1.weight"),
            "layers.0.attn_norm.weight"
        );
    }

    #[test]
    fn unknown_model_passthrough() {
        let mapper = WeightMapper::new("custom_arch");
        assert_eq!(
            mapper.map_name("model.layers.0.foo.weight"),
            "layers.0.foo.weight"
        );
    }

    #[test]
    fn no_model_prefix() {
        let mapper = WeightMapper::new("llama");
        assert_eq!(
            mapper.map_name("layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
    }

    #[test]
    fn model_type_accessor() {
        let mapper = WeightMapper::new("mistral");
        assert_eq!(mapper.model_type(), "mistral");
    }

    #[test]
    fn phi_mapping() {
        let mapper = WeightMapper::new("phi");
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_layernorm.weight"),
            "layers.0.attn.q_ln.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.mlp.gate_up_proj.weight"),
            "layers.0.ffn.gate_up.weight"
        );
    }

    #[test]
    fn phi3_mapping() {
        let mapper = WeightMapper::new("phi3");
        assert_eq!(
            mapper.map_name("model.layers.0.input_layernorm.weight"),
            "layers.0.attn_norm.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.mlp.down_proj.weight"),
            "layers.0.ffn.down.weight"
        );
    }

    #[test]
    fn gemma_mapping() {
        let mapper = WeightMapper::new("gemma");
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.ffn.gate.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.input_layernorm.weight"),
            "layers.0.attn_norm.weight"
        );
    }

    #[test]
    fn gemma2_mapping() {
        let mapper = WeightMapper::new("gemma2");
        assert_eq!(
            mapper.map_name("model.layers.0.pre_feedforward_layernorm.weight"),
            "layers.0.pre_ffn_norm.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.post_feedforward_layernorm.weight"),
            "layers.0.post_ffn_norm.weight"
        );
        assert_eq!(
            mapper.map_name("model.layers.0.post_attention_layernorm.weight"),
            "layers.0.post_attn_norm.weight"
        );
    }
}
