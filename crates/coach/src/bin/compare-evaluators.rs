use coach::duel::Duel;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::dice_gen::FastrandDice;
use engine::probabilities::{Probabilities, ResultCounter};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use std::io::{stdout, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let evaluator_1 = CompositeEvaluator::from_file_paths_optimized(
        "neural-nets/contact.onnx",
        "neural-nets/race.onnx",
    )
    .unwrap_or_exit_with_message();

    let evaluator_2 = CompositeEvaluator::from_file_paths_optimized(
        "neural-nets/contact.onnx",
        "neural-nets/race.onnx",
    )
    .unwrap_or_exit_with_message();
    // let evaluator_2 = engine::multiply::MultiPlyEvaluator {
    //     evaluator: evaluator_2,
    // };

    let duel = Duel::new(evaluator_1, evaluator_2);

    let mut dice_gen = FastrandDice::new();
    let mut global_counter = ResultCounter::default();

    loop {
        // If we create n seeds, than n duels are played in parallel which gives us 2*n GameResults.
        // When the duels have finished, the 2*n results are reduced to a single one and then
        // added to the `global_counter`.
        // Those global results are printed out and the endless loop starts again.
        let seeds: Vec<u64> = (0..50).map(|_| dice_gen.seed()).collect();
        let counter = seeds
            .into_par_iter()
            .map(|seed| duel.duel(&mut FastrandDice::with_seed(seed)))
            .reduce(ResultCounter::default, |a, b| a.combine(&b));

        global_counter = global_counter.combine(&counter);
        let probabilities = Probabilities::from(&global_counter);
        let better_evaluator = if probabilities.equity() > 0.0 { 1 } else { 2 };
        print!(
            "\rEvaluator {} is leading. After {:.1} thousand games the equity is {:.3}. {:?}",
            better_evaluator,
            global_counter.sum() as f32 / 1000.0,
            probabilities.equity(),
            probabilities,
        );
        stdout().flush().unwrap();
    }
}
