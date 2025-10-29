use clap::{Parser, ValueEnum};
use coach::duel::Duel;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::dice_gen::FastrandDice;
use engine::probabilities::{Probabilities, ResultCounter};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::io::{Write, stdout};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version)]
#[command(about = "Let standard neural nets duel various neural nets in the folder `training-data`", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
    /// Number of games played. Half the number of duels is executed.
    #[arg(long, default_value_t = 100_000)]
    number: u32,
}

#[derive(clap::ValueEnum, Clone, Serialize)]
enum Phase {
    Contact,
    Race,
}

/// Compare one evaluator with neural nets in the folder `training-data`.
/// This is meant to be used together with `train_contact.py` or `train_race.py`.
fn main() {
    let args = Args::parse();

    let folder_name = "training-data";
    println!("Start benchmarking, read contents of {folder_name}");

    let mut already_compared_nets: HashSet<String> = HashSet::new();

    loop {
        // Let's find all neural nets in the folder that are not yet compared.
        let mut paths_of_nets_to_compare: Vec<String> = fs::read_dir(folder_name)
            .unwrap()
            .map(|x| x.unwrap().file_name().into_string().unwrap())
            .filter(|x| x.starts_with(args.phase.to_possible_value().unwrap().get_name()))
            .filter(|x| x.ends_with(".onnx"))
            .filter(|x| !already_compared_nets.contains(x))
            .collect();

        if paths_of_nets_to_compare.is_empty() {
            print!("\rWaiting for new neural nets to be created. Kill the process to stop.");
            stdout().flush().unwrap();
            std::thread::sleep(std::time::Duration::from_secs(1));
        } else {
            paths_of_nets_to_compare.sort();
            compare_nets_with_default_nets(&paths_of_nets_to_compare, folder_name, &args);
            already_compared_nets.extend(paths_of_nets_to_compare);
        }
    }
}

fn compare_nets_with_default_nets(paths: &Vec<String>, folder_name: &str, args: &Args) {
    for file_name in paths {
        stdout().flush().unwrap();
        let current = CompositeEvaluator::from_file_paths_optimized(
            "neural-nets/contact.onnx",
            "neural-nets/race.onnx",
        )
        .unwrap_or_exit_with_message();

        let path_string = folder_name.to_string() + "/" + file_name.as_str();
        // We print lots of whitespace to overwrite the previous line.
        print!("\rTry {path_string}                                   ");
        stdout().flush().unwrap();
        let contender = match args.phase {
            Phase::Contact => {
                CompositeEvaluator::from_file_paths_optimized(&path_string, "neural-nets/race.onnx")
            }
            Phase::Race => CompositeEvaluator::from_file_paths_optimized(
                "neural-nets/contact.onnx",
                &path_string,
            ),
        }
        .unwrap_or_exit_with_message();

        let duel = Duel::new(contender, current);

        let mut dice_gen = FastrandDice::new();

        // If we create n seeds, then n duels are played in parallel, which gives us 2*n GameResults.
        let seeds: Vec<u64> = (0..args.number / 2).map(|_| dice_gen.seed()).collect();
        let counter = seeds
            .into_par_iter()
            .map(|seed| duel.duel(&mut FastrandDice::with_seed(seed)))
            .reduce(ResultCounter::default, |a, b| a.combine(&b));

        let probabilities = Probabilities::from(&counter);
        let winning_or_losing = if probabilities.equity() > 0.0 {
            "winning"
        } else {
            " losing"
        };
        println!(
            "\r{} is {}. After {} games the equity is {:7.4}. {:?}",
            file_name.strip_suffix(".onnx").unwrap(),
            winning_or_losing,
            counter.sum(),
            probabilities.equity(),
            probabilities,
        );
    }
}
