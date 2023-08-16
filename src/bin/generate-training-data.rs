use std::fs::File;
use std::io::Write;
use wildbg::evaluator::{Evaluator, Probabilities};
use wildbg::inputs::Inputs;
use wildbg::position::{Position, STARTING};
use wildbg::rollout::*;

fn main() {
    write().unwrap();
}

fn write() -> std::io::Result<()> {
    let path = "training-data/rollouts.csv";
    println!("Write CSV data to {}", path);
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(path);
    let mut file = File::create(path)?;

    let evaluator = RolloutEvaluator::new_random();
    let position = STARTING;
    let probabilities = evaluator.eval(&position);

    file.write_all(csv_header().as_bytes())?;
    file.write_all(csv_line(&position, &probabilities).as_bytes())?;
    Ok(())
}

fn csv_header() -> String {
    Probabilities::csv_header() + ";" + Inputs::csv_header().as_str() + "\n"
}

fn csv_line(position: &Position, probabilities: &Probabilities) -> String {
    probabilities.to_string() + ";" + Inputs::from_position(position).to_string().as_str() + "\n"
}
