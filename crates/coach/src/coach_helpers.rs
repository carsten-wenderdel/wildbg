use engine::position::OngoingPhase;
use std::io::{stdout, Write};
use std::time::Instant;

pub fn positions_file_name(phase: &OngoingPhase) -> String {
    format!("training-data/{:?}-positions.csv", phase).to_lowercase()
}

pub fn print_progress(done: usize, total: usize, start: Instant) -> std::io::Result<()> {
    let progress = (done + 1) as f32 / total as f32;
    let left = 1.0 - progress;
    let seconds_done = start.elapsed().as_secs();
    let seconds_todo = (seconds_done as f32 * (left / progress)) as u64;
    print!(
        "\rProgress: {:2.2} %. Time elapsed: {}. Time left: {}.  ",
        progress * 100.0,
        duration(seconds_done),
        duration(seconds_todo),
    );
    stdout().flush()?;
    Ok(())
}

pub fn duration(seconds: u64) -> String {
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let minutes = minutes % 60;
    let seconds = seconds % 60;
    format!("{:02}:{:02}:{:02} h", hours, minutes, seconds)
}
