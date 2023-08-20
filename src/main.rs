use std::collections::HashMap;
use wildbg::dice_gen::Dice;
use wildbg::evaluator::Evaluator;
use wildbg::onnx::OnnxEvaluator;
use wildbg::pos;
use wildbg::position::Position;
use wildbg::position::STARTING;

fn main() {
    let onnx = OnnxEvaluator::with_default_model().unwrap();

    let position = STARTING;
    let best = onnx.best_position(&position, &Dice::new(3, 1));
    println!("best after rolling 31: {:?}", best.switch_sides());

    let best = onnx.best_position(&position, &Dice::new(6, 1));
    println!("best after rolling 61: {:?}", best.switch_sides());

    let position = pos!(x 5:1, 3:4; o 24:3);
    let best = onnx.best_position(&position, &Dice::new(4, 3));
    println!("best in bearoff after 43 {:?}", best.switch_sides());
}
