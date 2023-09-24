use crate::evaluator::Evaluator;
use crate::inputs::Inputs;
use crate::position::Position;
use crate::probabilities::Probabilities;
use tract_onnx::prelude::*;

pub struct OnnxEvaluator {
    #[allow(clippy::type_complexity)]
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl Evaluator for OnnxEvaluator {
    fn eval(&self, position: &Position) -> Probabilities {
        let inputs = Inputs::from_position(position).to_vec();
        let tract_inputs = tract_ndarray::Array1::from_vec(inputs)
            .into_shape([1, crate::inputs::NUM_INPUTS])
            .unwrap();
        let tensor = tract_inputs.into_tensor();

        // run the model on the input
        let result = self.model.run(tvec!(tensor.into())).unwrap();
        let array_view = result[0].to_array_view::<f32>().unwrap();
        let result_vec: Vec<&f32> = array_view.iter().collect();
        Probabilities {
            win_normal: *result_vec[0],
            win_gammon: *result_vec[1],
            win_bg: *result_vec[2],
            lose_normal: *result_vec[3],
            lose_gammon: *result_vec[4],
            lose_bg: *result_vec[5],
        }
    }
}

impl OnnxEvaluator {
    pub fn with_default_model() -> Option<Self> {
        OnnxEvaluator::from_file_path("neural-nets/wildbg.onnx")
    }

    pub fn from_file_path(file_path: &str) -> Option<OnnxEvaluator> {
        match OnnxEvaluator::model(file_path) {
            Ok(model) => Some(OnnxEvaluator { model }),
            Err(_) => None,
        }
    }

    #[allow(clippy::type_complexity)]
    fn model(
        file_path: &str,
    ) -> TractResult<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>
    {
        let model = onnx()
            .model_for_path(file_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(model)
    }
}

/// The following tests mainly test the quality of the neural nets
#[cfg(test)]
mod tests {
    use crate::evaluator::Evaluator;
    use crate::onnx::OnnxEvaluator;
    use crate::pos;
    use crate::position::Position;
    use std::collections::HashMap;

    #[test]
    fn eval_certain_win_normal() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 1:1; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_normal > 0.85);
        assert!(probabilities.win_normal < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_gammon() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 1:1; o 18:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_gammon > 0.85);
        assert!(probabilities.win_gammon < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_bg() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 1:1; o 6:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_bg > 0.27);
        assert!(probabilities.win_bg < 0.32); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_normal() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 1:6; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_normal > 0.77);
        assert!(probabilities.lose_normal < 0.82); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_gammon() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 7:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_gammon > 0.92);
        assert!(probabilities.lose_gammon < 0.98); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_bg() {
        let onnx = OnnxEvaluator::with_default_model().unwrap();
        let position = pos![x 19:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_bg > 0.02);
        assert!(probabilities.lose_bg < 0.05); // This should be wrong, let's improve the nets.
    }
}
