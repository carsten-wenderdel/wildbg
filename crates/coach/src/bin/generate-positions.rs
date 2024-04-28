use coach::coach_helpers::{duration, positions_file_name};
use coach::position_finder::PositionFinder;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator;
use engine::position::OngoingPhase;
use mimalloc::MiMalloc;
use std::fs::File;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// This binary is for generating training data in CSV format.
///
/// The data is persisted with position ID only.
/// Later `generate-training-data.rs` can be used to roll out these positions.
fn main() -> std::io::Result<()> {
    // Change the next couple of lines to configure what, how and how much you want to roll out.
    let phase = OngoingPhase::Race;
    let amount = 200_000;
    let finder_evaluator = CompositeEvaluator::try_default().unwrap_or_exit_with_message();
    find_and_roll_out(finder_evaluator, amount, phase)?;
    Ok(())
}

fn find_and_roll_out<T: Evaluator>(
    finder_evaluator: T,
    amount: usize,
    phase: OngoingPhase,
) -> std::io::Result<()> {
    let path = positions_file_name(&phase);
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(&path);

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&path)?);
    csv_writer.write_record(["position_id"])?;

    println!(
        "Find {} '{:?}' positions and write them to {}.",
        amount, phase, path
    );

    let find_start = Instant::now();
    let mut finder = PositionFinder::with_random_dice(finder_evaluator);
    let positions = finder.find_positions(amount, phase);
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
