use clap::Parser;
use coach::data::{InputsRecord, PositionRecord};
use engine::inputs::{ContactInputsGen, RaceInputsGen};
use serde::Serialize;
use std::fs::File;

#[derive(Parser)]
#[command(version)]
#[command(about = "Convert position IDs and probabilities to neural net inputs.", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
}

#[derive(clap::ValueEnum, Clone, Serialize, Debug)]
enum Phase {
    Contact,
    Race,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let training_path = format!("training-data/{:?}.csv", args.phase).to_lowercase();
    let inputs_path = format!("training-data/{:?}-inputs.csv", args.phase).to_lowercase();
    println!("Read training data from {training_path} and write inputs to {inputs_path}");

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&training_path)?;

    let mut inputs_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create(&inputs_path)?);

    for result in csv_reader.deserialize() {
        let position_record: PositionRecord = result?;
        let inputs_record = match args.phase {
            Phase::Contact => InputsRecord::new(&position_record, &ContactInputsGen {}),
            Phase::Race => InputsRecord::new(&position_record, &RaceInputsGen {}),
        };
        inputs_writer.serialize(inputs_record)?;
    }

    println!("\nDone!");
    Ok(())
}
