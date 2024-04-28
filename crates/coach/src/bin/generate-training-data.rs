use coach::coach_helpers::{positions_file_name, print_progress};
use coach::data::PositionRecord;
use coach::rollout::RolloutEvaluator;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator;
use engine::position::{OngoingPhase, Position};
use mimalloc::MiMalloc;
use std::fs::File;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// This binary is for generating training data in CSV format.
///
/// The data is persisted with position ID and the "classic" 5 values for the probabilities.
/// The resulting file cannot be read by the Python scripts, they have to be converted first with `convert-to-inputs.rs`.
fn main() -> std::io::Result<()> {
    // Change the next couple of lines to configure what, how and how much you want to roll out.
    let phase = OngoingPhase::Race;
    let rollout_evaluator = CompositeEvaluator::try_default()
        .map(RolloutEvaluator::with_evaluator)
        .unwrap_or_exit_with_message();
    find_and_roll_out(rollout_evaluator, phase)?;

    println!("\nDone!");
    Ok(())
}

fn find_and_roll_out<T: Evaluator>(
    rollout_evaluator: T,
    phase: OngoingPhase,
) -> std::io::Result<()> {
    let positions_path = positions_file_name(&phase);
    let training_path = format!("training-data/{:?}.csv", phase).to_lowercase();

    println!(
        "Read positions from {} and write training data to {}",
        positions_path, training_path
    );

    let reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&positions_path)?;
    let positions: Vec<Position> = reader
        .into_records()
        .map(|record| Position::from_id(record.unwrap().as_slice().to_string()))
        .collect();

    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(&training_path);
    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&training_path)?);
    csv_writer.write_record(PositionRecord::csv_header())?;

    println!("Roll out {} '{:?}' positions", positions.len(), phase);

    let rollout_start = Instant::now();
    for (i, position) in positions.iter().enumerate() {
        let probabilities = rollout_evaluator.eval(position);
        let record = PositionRecord::new(position, &probabilities);
        csv_writer.serialize(record)?;
        csv_writer.flush()?;
        print_progress(i, positions.len(), rollout_start)?;
    }
    Ok(())
}
