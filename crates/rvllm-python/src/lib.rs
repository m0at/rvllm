#![allow(dead_code)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rvllm_core::prelude::SamplingParams;
use rvllm_sampling::sampler::{Sampler as RustSampler, SamplerOutput as RustSamplerOutput};

// ---------------------------------------------------------------------------
// SamplerOutput
// ---------------------------------------------------------------------------

#[pyclass(name = "SamplerOutput")]
#[derive(Clone)]
struct PySamplerOutput {
    #[pyo3(get)]
    token_id: u32,
    #[pyo3(get)]
    logprob: f32,
    #[pyo3(get)]
    top_logprobs: Vec<(u32, f32)>,
}

#[pymethods]
impl PySamplerOutput {
    fn __repr__(&self) -> String {
        format!(
            "SamplerOutput(token_id={}, logprob={:.4}, top_logprobs={})",
            self.token_id,
            self.logprob,
            self.top_logprobs.len()
        )
    }
}

impl From<RustSamplerOutput> for PySamplerOutput {
    fn from(o: RustSamplerOutput) -> Self {
        Self {
            token_id: o.token_id,
            logprob: o.logprob,
            top_logprobs: o.top_logprobs,
        }
    }
}

// ---------------------------------------------------------------------------
// SamplingParams
// ---------------------------------------------------------------------------

#[pyclass(name = "SamplingParams")]
#[derive(Clone)]
struct PySamplingParams {
    #[pyo3(get, set)]
    temperature: f32,
    #[pyo3(get, set)]
    top_p: f32,
    #[pyo3(get, set)]
    top_k: u32,
    #[pyo3(get, set)]
    min_p: f32,
    #[pyo3(get, set)]
    repetition_penalty: f32,
    #[pyo3(get, set)]
    frequency_penalty: f32,
    #[pyo3(get, set)]
    presence_penalty: f32,
    #[pyo3(get, set)]
    max_tokens: usize,
    #[pyo3(get, set)]
    seed: Option<u64>,
    #[pyo3(get, set)]
    logprobs: Option<usize>,
    #[pyo3(get, set)]
    length_penalty: f32,
    #[pyo3(get, set)]
    early_stopping: bool,
}

#[pymethods]
impl PySamplingParams {
    #[new]
    #[pyo3(signature = (
        temperature = 1.0,
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
        repetition_penalty = 1.0,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        max_tokens = 256,
        seed = None,
        logprobs = None,
        length_penalty = 1.0,
        early_stopping = false,
    ))]
    fn new(
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        max_tokens: usize,
        seed: Option<u64>,
        logprobs: Option<usize>,
        length_penalty: f32,
        early_stopping: bool,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            max_tokens,
            seed,
            logprobs,
            length_penalty,
            early_stopping,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={}, top_p={}, top_k={}, min_p={})",
            self.temperature, self.top_p, self.top_k, self.min_p
        )
    }
}

impl PySamplingParams {
    pub fn to_rust(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            max_tokens: self.max_tokens,
            seed: self.seed,
            logprobs: self.logprobs,
            length_penalty: self.length_penalty,
            early_stopping: self.early_stopping,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

#[pyclass(name = "Sampler")]
struct PySampler {
    inner: RustSampler,
}

#[pymethods]
impl PySampler {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustSampler::new(),
        }
    }

    #[pyo3(signature = (logits, temperature = 1.0, top_k = 0, top_p = 1.0, min_p = 0.0, seed = None, logprobs = None))]
    fn sample(
        &self,
        logits: Vec<f32>,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: Option<u64>,
        logprobs: Option<usize>,
    ) -> PyResult<PySamplerOutput> {
        let vocab_size = logits.len();
        let params = SamplingParams {
            temperature,
            top_p,
            top_k,
            min_p,
            logprobs,
            seed,
            ..Default::default()
        };
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        self.inner
            .sample(&logits, vocab_size, &params, &[], &mut rng)
            .map(PySamplerOutput::from)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> &'static str {
        "Sampler()"
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: rvllm_tokenizer::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_pretrained(model: &str) -> PyResult<Self> {
        let inner = rvllm_tokenizer::Tokenizer::from_pretrained(model)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner
            .encode(text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

// ---------------------------------------------------------------------------
// EngineConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "EngineConfig")]
#[derive(Clone)]
struct PyEngineConfig {
    #[pyo3(get, set)]
    model: String,
    #[pyo3(get, set)]
    max_model_len: usize,
    #[pyo3(get, set)]
    gpu_memory_utilization: f32,
    #[pyo3(get, set)]
    dtype: String,
    #[pyo3(get, set)]
    tensor_parallel_size: usize,
}

#[pymethods]
impl PyEngineConfig {
    #[new]
    #[pyo3(signature = (
        model = String::new(),
        max_model_len = 2048,
        gpu_memory_utilization = 0.90,
        dtype = "auto".to_string(),
        tensor_parallel_size = 1,
    ))]
    fn new(
        model: String,
        max_model_len: usize,
        gpu_memory_utilization: f32,
        dtype: String,
        tensor_parallel_size: usize,
    ) -> Self {
        Self {
            model,
            max_model_len,
            gpu_memory_utilization,
            dtype,
            tensor_parallel_size,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        let cfg = self.to_rust();
        serde_json::to_string_pretty(&cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "EngineConfig(model='{}', max_model_len={}, gpu_mem={:.2})",
            self.model, self.max_model_len, self.gpu_memory_utilization
        )
    }
}

impl PyEngineConfig {
    fn to_rust(&self) -> rvllm_config::EngineConfig {
        rvllm_config::EngineConfig::builder()
            .model(
                rvllm_config::ModelConfigImpl::builder()
                    .model_path(&self.model)
                    .dtype(
                        self.dtype
                            .parse::<rvllm_core::types::Dtype>()
                            .unwrap_or_default(),
                    )
                    .max_model_len(self.max_model_len)
                    .build(),
            )
            .cache(
                rvllm_config::CacheConfigImpl::builder()
                    .gpu_memory_utilization(self.gpu_memory_utilization)
                    .build(),
            )
            .parallel(rvllm_config::ParallelConfigImpl {
                tensor_parallel_size: self.tensor_parallel_size,
                ..Default::default()
            })
            .build()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (logits_batch, temperature = 1.0, top_p = 1.0, top_k = 0, min_p = 0.0, seed = None))]
fn sample_batch(
    logits_batch: Vec<Vec<f32>>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    seed: Option<u64>,
) -> PyResult<Vec<PySamplerOutput>> {
    let params = SamplingParams {
        temperature,
        top_p,
        top_k,
        min_p,
        ..Default::default()
    };
    let base_seed = seed.unwrap_or(0);
    let has_seed = seed.is_some();

    let results: Vec<PyResult<PySamplerOutput>> = logits_batch
        .par_iter()
        .enumerate()
        .map(|(i, logits)| {
            let sampler = RustSampler::new();
            let vocab_size = logits.len();
            let mut rng = if has_seed {
                ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64))
            } else {
                ChaCha8Rng::from_entropy()
            };
            sampler
                .sample(logits, vocab_size, &params, &[], &mut rng)
                .map(PySamplerOutput::from)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
        .collect();

    results.into_iter().collect()
}

#[pyfunction]
fn system_info() -> String {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    format!(
        "rvllm v{version}\nos: {os}\narch: {arch}\ncpu_threads: {cpus}\nrayon_threads: {rayon}",
        version = env!("CARGO_PKG_VERSION"),
        os = os,
        arch = arch,
        cpus = cpus,
        rayon = rayon::current_num_threads(),
    )
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn rvllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PySampler>()?;
    m.add_class::<PySamplerOutput>()?;
    m.add_class::<PySamplingParams>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEngineConfig>()?;
    m.add_function(wrap_pyfunction!(sample_batch, m)?)?;
    m.add_function(wrap_pyfunction!(system_info, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_sampling_params_defaults_match_rust_defaults() {
        let params =
            PySamplingParams::new(1.0, 1.0, 0, 0.0, 1.0, 0.0, 0.0, 256, None, None, 1.0, false);
        let rust = params.to_rust();
        assert_eq!(rust.length_penalty, 1.0);
        assert!(!rust.early_stopping);
    }

    #[test]
    fn py_sampling_params_to_rust_copies_beam_controls() {
        let params = PySamplingParams::new(
            0.8,
            0.9,
            10,
            0.1,
            1.1,
            0.2,
            0.3,
            42,
            Some(7),
            Some(4),
            0.5,
            true,
        );
        let rust = params.to_rust();
        assert_eq!(rust.length_penalty, 0.5);
        assert!(rust.early_stopping);
        assert_eq!(rust.max_tokens, 42);
        assert_eq!(rust.logprobs, Some(4));
    }
}
