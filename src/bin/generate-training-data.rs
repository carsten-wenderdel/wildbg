use std::fs::File;
use std::io::{stdout, Write};
use wildbg::evaluator::{Evaluator, Probabilities};
use wildbg::inputs::Inputs;
use wildbg::position::Position;
use wildbg::position_finder::PositionFinder;
use wildbg::rollout::*;

fn main() -> std::io::Result<()> {
    let path = "training-data/rollouts.csv";
    println!("Roll out and write CSV data to {}", path);
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(path);
    let mut file = File::create(path)?;
    file.write_all(csv_header().as_bytes())?;

    let amount = 100_000;
    let random_positions = PositionFinder::new().find_positions(amount);
    let evaluator = RolloutEvaluator::new_random();

    for (i, position) in random_positions.iter().enumerate() {
        let probabilities = evaluator.eval(position);
        file.write_all(csv_line(position, &probabilities).as_bytes())?;
        print!(
            "\rProgress: {:.2} %",
            (i + 1) as f32 / amount as f32 * 100.0
        );
        stdout().flush().unwrap()
    }
    println!("\nDone!");
    Ok(())
}

fn csv_header() -> String {
    Probabilities::csv_header() + ";" + Inputs::csv_header().as_str() + "\n"
}

fn csv_line(position: &Position, probabilities: &Probabilities) -> String {
    probabilities.to_string() + ";" + Inputs::from_position(position).to_string().as_str() + "\n"
}
