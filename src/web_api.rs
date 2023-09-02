use crate::bg_move::{BgMove, MoveDetail};
use crate::dice::Dice;
use crate::evaluator::Evaluator;
use crate::onnx::OnnxEvaluator;
use crate::position::STARTING;

pub struct WebApi {
    evaluator: OnnxEvaluator,
}

impl WebApi {
    pub fn try_default() -> Option<Self> {
        OnnxEvaluator::with_default_model().map(|evaluator| Self { evaluator })
    }

    /// Currently this returns a static move. Work in progress
    pub fn get_move(&self) -> Vec<MoveDetail> {
        let position = STARTING;
        let dice = Dice::new(3, 1);
        let new = self.evaluator.best_position(&position, &dice);
        let bg_move = BgMove::new(&position, &new.switch_sides(), &dice);
        bg_move.into_details()
    }
}
