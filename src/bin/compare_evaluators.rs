use std::io::{stdout, Write};
use wildbg::duel::Duel;
use wildbg::onnx::OnnxEvaluator;

fn main() {
    let evaluator1 = OnnxEvaluator::from_file_path("neural-nets/wildbg.onnx").unwrap();
    let evaluator2 = OnnxEvaluator::from_file_path("neural-nets/wildbg01.onnx").unwrap();
    let mut duel = Duel::new(evaluator1, evaluator2);

    println!("Let two Evaluators duel each other:");
    for _ in 0..100_000 {
        duel.duel_once();
        let probabilities = duel.probabilities();
        print!(
            "\rAfter {} games is the equity {:.3}. {:?}",
            duel.number_of_games(),
            probabilities.equity(),
            probabilities,
        );
        stdout().flush().unwrap()
    }
    println!("\nDone");
}
