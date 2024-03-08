use coach::duel::Duel;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::dice_gen::FastrandDice;
use engine::probabilities::{Probabilities, ResultCounter};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use std::fs;
use std::io::{stdout, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Compare one evaluator with neural nets in the folder `training-data`.
fn main() {
    let folder_name = "training-data";
    println!("Start benchmarking, read contents of {}", folder_name);
    let mut paths = fs::read_dir(folder_name)
        .unwrap()
        .map(|x| x.unwrap().file_name().into_string().unwrap())
        .filter(|x| x.ends_with(".onnx"))
        .collect::<Vec<_>>();
    paths.sort();

    for file_name in paths {
        print!("Load current neural nets");
        stdout().flush().unwrap();
        let current = CompositeEvaluator::from_file_paths_optimized(
            "neural-nets/contact.onnx",
            "neural-nets/race.onnx",
        )
        .unwrap_or_exit_with_message();

        let path_string = folder_name.to_string() + "/" + file_name.as_str();
        print!("\rTry {}", path_string);
        stdout().flush().unwrap();
        let contender =
            CompositeEvaluator::from_file_paths_optimized(&path_string, "neural-nets/race.onnx")
                .unwrap_or_exit_with_message();

        let duel = Duel::new(contender, current);

        let mut dice_gen = FastrandDice::new();

        let number_of_games = 100_000;

        // If we create n seeds, than n duels are played in parallel which gives us 2*n GameResults.
        let seeds: Vec<u64> = (0..number_of_games / 2).map(|_| dice_gen.seed()).collect();
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
    println!("Finished benchmarking");
}
