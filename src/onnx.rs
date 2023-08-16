use crate::evaluator::{Evaluator, Probabilities};
use crate::inputs::Inputs;
use crate::position::Position;
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
