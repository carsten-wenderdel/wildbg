use coach::data::PositionRecord;
use coach::position_finder::PositionFinder;
use coach::rollout::RolloutEvaluator;
use csv::Writer;
use engine::evaluator::{Evaluator, RandomEvaluator};
use engine::onnx::OnnxEvaluator;
use engine::position::{OngoingPhase, Position};
use engine::probabilities::Probabilities;
use std::fs::File;
use std::io::{stdout, Write};
use std::time::Instant;

const AMOUNT: usize = 100;

fn main() -> std::io::Result<()> {
    let phase = OngoingPhase::Race;

    let path = format!("training-data/{:?}.csv", phase).to_lowercase();
    println!("Roll out and write CSV data to {}", path);
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(&path);
    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&path)?);
    csv_writer.write_record(PositionRecord::csv_header())?;

    let evaluator = OnnxEvaluator::contact_default().map(RolloutEvaluator::with_evaluator);
    let finder = OnnxEvaluator::contact_default().map(PositionFinder::with_random_dice);
    let start = Instant::now();

    match (evaluator, finder) {
        (Some(evaluator), Some(mut finder)) => {
            println!("Use onnx evaluator");
            let positions = finder.find_positions(AMOUNT, phase);
            for (i, position) in positions.iter().enumerate() {
                let probabilities = evaluator.eval(position);
                write_csv_line(&mut csv_writer, position, &probabilities, i, start)?;
            }
        }
        (_, _) => {
            println!("Couldn't find onnx file, use random evaluator");
            let evaluator = RolloutEvaluator::with_evaluator(RandomEvaluator {});
            let mut finder = PositionFinder::with_random_dice(RandomEvaluator {});
            let positions = finder.find_positions(AMOUNT, phase);
            for (i, position) in positions.iter().enumerate() {
                let probabilities = evaluator.eval(position);
                write_csv_line(&mut csv_writer, position, &probabilities, i, start)?;
            }
        }
    }
    println!("\nDone!");
    Ok(())
}

fn write_csv_line(
    writer: &mut Writer<File>,
    position: &Position,
    probabilities: &Probabilities,
    i: usize,
    start: Instant,
) -> std::io::Result<()> {
    let record = PositionRecord::new(position, probabilities);
    writer.serialize(record)?;

    let done = (i + 1) as f32 / AMOUNT as f32;
    let todo = 1.0 - done;
    let seconds_done = start.elapsed().as_secs();
    let seconds_todo = (seconds_done as f32 * (todo / done)) as u64;
    print!(
        "\rProgress: {:2.2} %. Time elapsed: {}. Time left: {}.",
        done * 100.0,
        duration(seconds_done),
        duration(seconds_todo),
    );
    stdout().flush()?;
    Ok(())
}

fn duration(seconds: u64) -> String {
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let minutes = minutes % 60;
    let seconds = seconds % 60;
    format!("{:02}:{:02}:{:02} h", hours, minutes, seconds)
}
