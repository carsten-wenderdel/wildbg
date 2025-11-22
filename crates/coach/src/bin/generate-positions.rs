use clap::Parser;
use coach::coach_helpers::{duration, positions_file_name};
use coach::position_finder::position_finder_with_evaluator;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::position::OngoingPhase;
use mimalloc::MiMalloc;
use serde::Serialize;
use std::fs::File;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version)]
#[command(about = "Let standard neural nets duel various neural nets in the folder `training-data`", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
    /// Number of positions to find.
    #[arg(long, default_value_t = 200_000)]
    number: usize,
}

#[derive(clap::ValueEnum, Clone, Serialize)]
enum Phase {
    Contact,
    Race,
}

/// This binary is for generating positions.
///
/// The data is persisted with position ID only in a CSV file.
/// Later `generate-training-data.rs` can be used to roll out these positions.
fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let phase = match args.phase {
        Phase::Contact => OngoingPhase::Contact,
        Phase::Race => OngoingPhase::Race,
    };

    let finder_evaluator = CompositeEvaluator::try_default().unwrap_or_exit_with_message();

    let path = positions_file_name(&phase);
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(&path);

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&path)?);
    csv_writer.write_record(["position_id"])?;

    println!(
        "Find {:?} '{phase:?}' positions and write them to {path}.",
        args.number
    );

    let find_start = Instant::now();
    let mut finder = position_finder_with_evaluator(finder_evaluator);
    let positions = finder.find_positions(args.number, phase);
    for position in positions {
        csv_writer.write_record([position.position_id()])?;
    }
    csv_writer.flush()?;

    println!(
        "All positions found in {}.",
        duration(find_start.elapsed().as_secs())
    );

    Ok(())
}
