#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rvrlm::{Rlm, RvllmCudaConfig};

    let prompt = std::env::var("RVRLM_PROMPT")
        .unwrap_or_else(|_| "Write one short sentence about recursive language models.".to_owned());
    let mut config = RvllmCudaConfig::from_env()?;
    if std::env::var("RVRLM_MAX_NEW_TOKENS").is_err() {
        config.max_new_tokens = 32;
    }

    let mut rlm = Rlm::from_rvllm_cuda(config)?;
    let completion = rlm.completion(prompt)?;

    println!("model: {}", completion.root_model);
    println!("seconds: {:.3}", completion.execution_time_secs);
    println!(
        "tokens_in: {} tokens_out: {}",
        completion.usage_summary.total_input_tokens(),
        completion.usage_summary.total_output_tokens()
    );
    println!("response:\n{}", completion.response);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_smoke requires --features cuda");
    std::process::exit(1);
}
