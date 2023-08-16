use wildbg::evaluator::Evaluator;
use wildbg::onnx::OnnxEvaluator;
use wildbg::position::STARTING;

fn main() {
    let position = STARTING;
    let onnx = OnnxEvaluator::from_file_path("neural-nets/wildbg.onnx").unwrap();
    let best = onnx.best_position(&position, 3, 1);
    println!("best: {:?}", best);
}
