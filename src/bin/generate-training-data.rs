use wildbg::evaluator::Evaluator;
use wildbg::position::STARTING;
use wildbg::rollout::*;

fn main() {
    // Try out the RolloutEvaluator, will evolve into generating training data.
    let evaluator = RolloutEvaluator::new_random();
    let position = STARTING;
    let probabilities = evaluator.eval(&position);
    println!("{:?}", probabilities);
}
