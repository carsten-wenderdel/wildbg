use coach::data::PositionRecord;
use coach::position_finder::PositionFinder;
use coach::rollout::RolloutEvaluator;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator;
use engine::position::OngoingPhase;
use mimalloc::MiMalloc;
use std::fs::File;
use std::io::{stdout, Write};
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
    let amount = 200_000;
    let rollout_evaluator = CompositeEvaluator::try_default()
        .map(RolloutEvaluator::with_evaluator)
        .unwrap_or_exit_with_message();
    let finder_evaluator = CompositeEvaluator::try_default().unwrap_or_exit_with_message();

    find_and_roll_out(finder_evaluator, rollout_evaluator, amount, phase)?;

    println!("\nDone!");
    Ok(())
}

fn find_and_roll_out<T: Evaluator, U: Evaluator>(
    finder_evaluator: T,
    rollout_evaluator: U,
    amount: usize,
    phase: OngoingPhase,
) -> std::io::Result<()> {
    let path = format!("training-data/{:?}.csv", phase).to_lowercase();
    _ = std::fs::create_dir("training-data");
    _ = std::fs::remove_file(&path);

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&path)?);
    csv_writer.write_record(PositionRecord::csv_header())?;

    println!(
        "Find {} '{:?}' positions, roll them out and write data to {}.",
        amount, phase, path
    );
    let find_start = Instant::now();
    let mut finder = PositionFinder::with_random_dice(finder_evaluator);
    let positions = finder.find_positions(amount, phase);

    println!(
        "All positions found in {}. Now performing rollouts on {} threads:",
        duration(find_start.elapsed().as_secs()),
        rayon::current_num_threads()
    );

    let rollout_start = Instant::now();
    for (i, position) in positions.iter().enumerate() {
        let probabilities = rollout_evaluator.eval(position);
        let record = PositionRecord::new(position, &probabilities);
        csv_writer.serialize(record)?;
        csv_writer.flush()?;
        print_progress(i, amount, rollout_start)?;
    }
    Ok(())
}

fn print_progress(i: usize, amount: usize, start: Instant) -> std::io::Result<()> {
    let done = (i + 1) as f32 / amount as f32;
    let todo = 1.0 - done;
    let seconds_done = start.elapsed().as_secs();
    let seconds_todo = (seconds_done as f32 * (todo / done)) as u64;
    print!(
        "\rProgress: {:2.2} %. Time elapsed: {}. Time left: {}.  ",
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
