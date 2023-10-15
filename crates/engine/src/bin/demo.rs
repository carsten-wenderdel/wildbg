use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::onnx::OnnxEvaluator;
use engine::pos;
use engine::position::Position;
use engine::position::STARTING;
use std::collections::HashMap;

fn main() {
    let onnx = OnnxEvaluator::contact_default().unwrap();

    let position = STARTING;
    let best = onnx.best_position_by_equity(&position, &Dice::new(3, 1));
    println!("best after rolling 31: {:?}", best.switch_sides());

    let best = onnx.best_position_by_equity(&position, &Dice::new(6, 1));
    println!("best after rolling 61: {:?}", best.switch_sides());

    let position = pos!(x 5:1, 3:4; o 24:3);
    let best = onnx.best_position_by_equity(&position, &Dice::new(4, 3));
    println!("best in bearoff after 43 {:?}", best.switch_sides());
}
