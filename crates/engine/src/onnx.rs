use crate::evaluator::BatchEvaluator;
use crate::inputs::{ContactInputsGen, InputsGen, RaceInputsGen};
use crate::position::Position;
use crate::probabilities::Probabilities;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::shapefactoid;

pub struct OnnxEvaluator<T: InputsGen> {
    #[allow(clippy::type_complexity)]
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    inputs_gen: T,
}

impl<T: InputsGen> BatchEvaluator for OnnxEvaluator<T> {
    fn eval_positions(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        if positions.is_empty() {
            return Vec::new();
        }

        // Turn all inputs into one big vector:
        let inputs: Vec<f32> = positions
            .iter()
            .flat_map(|p| self.inputs_gen.input_vec(p))
            .collect();

        let tract_inputs = tract_ndarray::Array1::from_vec(inputs)
            .into_shape((positions.len(), T::NUM_INPUTS))
            .unwrap();
        let tensor = tract_inputs.into_tensor();

        // run the model on the input
        let result = self.model.run(tvec!(tensor.into())).unwrap();

        // Extract all the probabilities from the result:
        let array_view = result[0].to_array_view::<f32>().unwrap();
        let probabilities_in_shape = array_view.into_shape((positions.len(), 6)).unwrap();
        let probabilities_iter = probabilities_in_shape.outer_iter().map(|x| Probabilities {
            win_normal: x[0],
            win_gammon: x[1],
            win_bg: x[2],
            lose_normal: x[3],
            lose_gammon: x[4],
            lose_bg: x[5],
        });
        let positions_and_probabilities: Vec<(Position, Probabilities)> =
            positions.into_iter().zip(probabilities_iter).collect();
        positions_and_probabilities
    }
}

const CONTACT_FILE_PATH: &str = "neural-nets/contact.onnx";
const RACE_FILE_PATH: &str = "neural-nets/race.onnx";

impl OnnxEvaluator<RaceInputsGen> {
    pub fn race_default() -> Option<Self> {
        OnnxEvaluator::from_file_path(RACE_FILE_PATH, RaceInputsGen {})
    }

    pub fn race_default_tests() -> Self {
        // Tests are executed from a different path than binary crates - so we need to slightly change the folder for them.
        OnnxEvaluator::from_file_path(&("../../".to_owned() + RACE_FILE_PATH), RaceInputsGen {})
            .expect("onnx file should exist at that path.")
    }
}

impl OnnxEvaluator<ContactInputsGen> {
    pub fn contact_default() -> Option<Self> {
        OnnxEvaluator::from_file_path(CONTACT_FILE_PATH, ContactInputsGen {})
    }

    pub fn contact_default_tests() -> Self {
        // Tests are executed from a different path than binary crates - so we need to slightly change the folder for them.
        OnnxEvaluator::from_file_path(
            &("../../".to_owned() + CONTACT_FILE_PATH),
            ContactInputsGen {},
        )
        .expect("onnx file should exist at that path.")
    }
}

impl<T: InputsGen> OnnxEvaluator<T> {
    pub fn from_file_path(file_path: &str, inputs_gen: T) -> Option<OnnxEvaluator<T>> {
        match Self::model(file_path) {
            Ok(model) => Some(OnnxEvaluator { model, inputs_gen }),
            Err(_) => None,
        }
    }

    #[allow(clippy::type_complexity)]
    fn model(
        file_path: &str,
    ) -> TractResult<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>
    {
        // The input tensor for the model could be for example [1, 202] - one position with 202 inputs.
        // We override with [N, 202] to allow batch evaluation.
        let batch = SymbolTable::default().sym("N");
        let model = onnx()
            .model_for_path(file_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid![batch, (T::NUM_INPUTS)]),
            )?
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
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 1:1; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_normal > 0.85);
        assert!(probabilities.win_normal < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_gammon() {
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 1:1; o 18:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_gammon > 0.85);
        assert!(probabilities.win_gammon < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_bg() {
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 1:1; o 6:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_bg > 0.27);
        assert!(probabilities.win_bg < 0.32); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_normal() {
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 1:6; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_normal > 0.77);
        assert!(probabilities.lose_normal < 0.82); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_gammon() {
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 7:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_gammon > 0.92);
        assert!(probabilities.lose_gammon < 0.98); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_bg() {
        let onnx = OnnxEvaluator::contact_default_tests();
        let position = pos![x 19:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_bg > 0.02);
        assert!(probabilities.lose_bg < 0.05); // This should be wrong, let's improve the nets.
    }
}
