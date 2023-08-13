use wildbg::evaluator::Evaluator;
use wildbg::inputs::Inputs;
use wildbg::position::STARTING;
use wildbg::rollout::*;

fn main() {
    // Try out the RolloutEvaluator, will evolve into generating training data.
    let evaluator = RolloutEvaluator::new_random();
    let position = STARTING;
    let probabilities = evaluator.eval(&position);
    let inputs = Inputs::from_position(&position);
    println!("{:?}", probabilities);
    println!("{}", Inputs::csv_header());
    println!("{}", inputs);
}
