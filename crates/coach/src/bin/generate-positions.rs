use clap::Parser;
use coach::coach_helpers::{duration, positions_file_name};
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
#[command(about = "Find positions for later rollout and store them in a CSV file. The result can be used for rollout and training.", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
    #[arg(long)]
    strategy: Strategy,
    #[arg(long, required_if_eq("strategy", "discrepancy"))]
    /// Threshold for equity discrepancy between the two best positions according to both evaluators. A good value is 0.4.
    threshold: Option<f32>,
    /// Number of positions to find.
    #[arg(long)]
    number: usize,
}

#[derive(clap::ValueEnum, Clone, Serialize)]
enum Phase {
    Contact,
    Race,
}

#[derive(clap::ValueEnum, Clone, Serialize)]
enum Strategy {
    Diverse,
    Discrepancy,
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

    let mut finder = match args.strategy {
        Strategy::Diverse => coach::position_finder::diverse_with_evaluator(
            CompositeEvaluator::try_default_optimized().unwrap(),
        ),
        Strategy::Discrepancy => coach::position_finder::discrepancy_with_evaluator(
            CompositeEvaluator::try_default_optimized().unwrap(),
            args.threshold.unwrap(),
        ),
    };

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
