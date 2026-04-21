pub fn find_code_blocks(input: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut inside = false;
    let mut current = Vec::new();

    for line in input.lines() {
        if line.trim_start().starts_with("```") {
            if inside {
                blocks.push(current.join("\n"));
                current.clear();
                inside = false;
            } else {
                inside = true;
            }
            continue;
        }

        if inside {
            current.push(line.to_owned());
        }
    }

    blocks
}

pub fn find_final_answer(input: &str) -> Option<String> {
    input
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("FINAL_ANSWER:").map(str::trim))
        .map(str::to_owned)
}
