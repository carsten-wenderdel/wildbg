use clap::Parser;
use coach::coach_helpers::{positions_file_name, print_progress};
use coach::data::PositionRecord;
use coach::rollout::RolloutEvaluator;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator;
use engine::position::{OngoingPhase, Position};
use mimalloc::MiMalloc;
use serde::Serialize;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version)]
#[command(about = "Generate training data by rolling out positions from a CSV file.", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
}

#[derive(clap::ValueEnum, Clone, Serialize)]
enum Phase {
    Contact,
    Race,
}

/// This binary is for generating training data in CSV format.
///
/// The data is persisted with position ID and the "classic" 5 values for the probabilities.
/// The resulting file cannot be read by the Python scripts, they have to be converted first with `convert-to-inputs.rs`.
fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let phase = match args.phase {
        Phase::Contact => OngoingPhase::Contact,
        Phase::Race => OngoingPhase::Race,
    };

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
    let training_path = format!("training-data/{phase:?}.csv").to_lowercase();

    println!("Read positions from {positions_path} and write training data to {training_path}");

    let positions: Vec<Position> = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&positions_path)?
        .into_records()
        .map(|record| Position::from_id(record.unwrap().as_slice()))
        .collect();

    // We want to be able to stop this application and resume later on.
    // So let's see whether we've already rolled out positions to skip them.
    let existing_positions: Vec<Position> = if Path::new(&training_path).exists() {
        csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&training_path)?
            .into_records()
            .map(|record| Position::from_id(record.unwrap().get(0).unwrap()))
            .collect()
    } else {
        Vec::new()
    };

    let positions = positions
        .strip_prefix(existing_positions.as_slice())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "The target file already contains other rollout data, delete it first.",
            )
        })?;

    let mut csv_writer = csv::WriterBuilder::new().has_headers(false).from_writer(
        File::options()
            .create(true)
            .append(true)
            .open(&training_path)?,
    );

    if !existing_positions.is_empty() {
        println!(
            "{} positions have already been rolled out, let's deal with the remaining:",
            existing_positions.len(),
        );
    } else {
        csv_writer.write_record(PositionRecord::csv_header())?;
    }

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
