use coach::duel::Duel;
use engine::dice::FastrandDice;
use engine::inputs::ContactInputsGen;
use engine::onnx::OnnxEvaluator;
use engine::probabilities::{Probabilities, ResultCounter};
use rayon::prelude::*;
use std::io::{stdout, Write};

fn main() {
    let evaluator1 = OnnxEvaluator::from_file_path("neural-nets/wildbg.onnx", ContactInputsGen {})
        .expect("Neural net for evaluator1 could not be found");
    let evaluator2 =
        OnnxEvaluator::from_file_path("neural-nets/wildbg01.onnx", ContactInputsGen {})
            .expect("Neural net for evaluator2 could not be found");
    let duel = Duel::new(evaluator1, evaluator2);

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
