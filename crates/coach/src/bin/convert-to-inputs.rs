use coach::data::{InputsRecord, PositionRecord};
use engine::inputs;
use engine::position::OngoingPhase;
use std::fs::File;

fn main() -> std::io::Result<()> {
    // Change the next two lines in case you want to create inputs for another game phase.
    let phase = OngoingPhase::Race;
    let inputs_gen = inputs::RaceInputsGen {};

    let training_path = format!("training-data/{:?}.csv", phase).to_lowercase();
    let inputs_path = format!("training-data/{:?}-inputs.csv", phase).to_lowercase();
    println!(
        "Read training data from {} and write inputs to {}",
        training_path, inputs_path
    );

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&training_path)?;

    let mut inputs_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&inputs_path)?);

    for result in csv_reader.deserialize() {
        let position_record: PositionRecord = result?;
        let inputs_record = InputsRecord::new(&position_record, &inputs_gen);
        inputs_writer.serialize(inputs_record)?;
        inputs_writer.flush()?;
    }

    println!("\nDone!");
    Ok(())
}
