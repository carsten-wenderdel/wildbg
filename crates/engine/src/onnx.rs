use std::fs::File;

use tract_onnx::prelude::*;
use tract_onnx::tract_hir::shapefactoid;

use crate::evaluator::BatchEvaluator;
use crate::inputs::{ContactInputsGen, InputsGen, RaceInputsGen};
use crate::position::Position;
use crate::probabilities::Probabilities;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
type Error = String;

mod number_of_models {
    pub(super) const SINGLE: usize = 1;
    pub(super) const OPTIMIZED: usize = 50;
}

pub struct OnnxEvaluator<T: InputsGen> {
    /// Onnx models optimized for different batch sizes.
    ///
    /// On index 0 is the model optimized for any batch size. It is used for batch sizes bigger than
    /// the length of this vector.
    /// For a batch size `i`, smaller than the length of this vector, the model at `models[i]` is used.
    ///
    /// This is an optimization, everything would also work with just the model at index 0 which has
    /// a dynamic input shape. But models optimized for a specific batch size give a performance
    /// boost of about globally 2% for rollouts.
    ///
    /// See also https://github.com/sonos/tract/discussions/716#discussioncomment-2769616
    models: Vec<TractModel>,

    /// Inputs generator specific to a certain game phase (like contact or race). The neural nets
    /// have different inputs for different game phases.
    inputs_gen: T,
}

impl<T: InputsGen> BatchEvaluator for OnnxEvaluator<T> {
    fn eval_positions(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        if positions.is_empty() {
            return Vec::new();
        }

        let inputs = self.inputs_gen.inputs_for_all(&positions);
        let tract_inputs = tract_ndarray::Array1::from_vec(inputs)
            .into_shape((positions.len(), T::NUM_INPUTS))
            .unwrap();
        let tensor = tract_inputs.into_tensor();

        // run the model on the input
        let index = if positions.len() < self.models.len() {
            positions.len()
        } else {
            0
        };
        let result = self.models[index].run(tvec!(tensor.into())).unwrap();

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

impl OnnxEvaluator<RaceInputsGen> {
    pub fn race_default() -> Result<Self, Error> {
        Self::race_default_number(number_of_models::SINGLE)
    }

    /// Compared to `race_default`, this function takes much longer to execute and the
    /// resulting struct is about 50 times bigger. But rollouts are about 2% faster.   
    pub fn race_default_optimized() -> Result<Self, Error> {
        Self::race_default_number(number_of_models::OPTIMIZED)
    }

    fn race_default_number(number_of_optimized_models: usize) -> Result<Self, Error> {
        let mut bytes = include_bytes!("../../../neural-nets/race.onnx").as_slice();
        OnnxEvaluator::from_reader_with_variable_number_of_models(
            &mut bytes,
            RaceInputsGen {},
            number_of_optimized_models,
        )
    }
}

impl OnnxEvaluator<ContactInputsGen> {
    pub fn contact_default() -> Result<Self, Error> {
        Self::contact_default_number(number_of_models::SINGLE)
    }

    /// Compared to `contact_default`, this function takes much longer to execute and the
    /// resulting struct is about 50 times bigger. But rollouts are about 2% faster.   
    pub fn contact_default_optimized() -> Result<Self, Error> {
        Self::contact_default_number(number_of_models::OPTIMIZED)
    }

    fn contact_default_number(number_of_optimized_models: usize) -> Result<Self, Error> {
        let mut bytes = include_bytes!("../../../neural-nets/contact.onnx").as_slice();
        OnnxEvaluator::from_reader_with_variable_number_of_models(
            &mut bytes,
            ContactInputsGen {},
            number_of_optimized_models,
        )
    }
}

impl<T: InputsGen> OnnxEvaluator<T> {
    /// Load the onnx model from the file path and optimize it for any batch size.
    ///
    /// Use it when you are low on memory or if this initializer is called very often.
    pub fn from_file_path(file_path: &str, inputs_gen: T) -> Result<OnnxEvaluator<T>, Error> {
        Self::from_file_path_with_variable_number_of_models(
            file_path,
            inputs_gen,
            number_of_models::SINGLE,
        )
    }

    /// Load the onnx model from the file path and optimize it several times for various batch sizes.
    ///
    /// Compared to `from_file_path`, this function takes much longer to execute and the
    /// resulting struct is about 50 times bigger. But rollouts are about 2% faster.
    pub fn from_file_path_optimized(
        file_path: &str,
        inputs_gen: T,
    ) -> Result<OnnxEvaluator<T>, Error> {
        Self::from_file_path_with_variable_number_of_models(
            file_path,
            inputs_gen,
            number_of_models::OPTIMIZED,
        )
    }

    fn from_file_path_with_variable_number_of_models(
        file_path: &str,
        inputs_gen: T,
        number_of_optimized_models: usize,
    ) -> Result<OnnxEvaluator<T>, Error> {
        match File::open(file_path) {
            Ok(mut file) => {
                match Self::from_reader_with_variable_number_of_models(
                    &mut file,
                    inputs_gen,
                    number_of_optimized_models,
                ) {
                    Ok(evaluator) => Ok(evaluator),
                    Err(_) => Err(format!("Could not process onnx file {file_path}")),
                }
            }
            Err(_) => Err(format!("Could not find onnx file {file_path}")),
        }
    }

    fn from_reader_with_variable_number_of_models(
        reader: &mut dyn std::io::Read,
        inputs_gen: T,
        number_of_optimized_models: usize,
    ) -> Result<OnnxEvaluator<T>, Error> {
        match Self::models(reader, number_of_optimized_models) {
            Ok(models) => Ok(OnnxEvaluator { models, inputs_gen }),
            Err(_) => Err("Could not process onnx file".to_string()),
        }
    }

    /// Load the onnx model from the `reader` and optimize it several times for different batch sizes.
    ///
    /// `number_of_optimized_models` is the number of models that will be optimized for a specific batch size.
    /// Use `1` for a single model that is optimized for any batch size.
    /// When using for example `50`, one model is optimized for any batch size (at index `0` in
    /// the returning array), the other 49 are optimized for batch sizes from `1` to `49`.
    fn models(
        reader: &mut dyn std::io::Read,
        number_of_optimized_models: usize,
    ) -> TractResult<Vec<TractModel>> {
        let model = onnx().model_for_read(reader)?;
        let mut models: Vec<TractModel> = Vec::new();
        for i in 0..number_of_optimized_models {
            let fact: InferenceFact = if i == 0 {
                // The input tensor for the model could be for example [1, 202] - one position with 202 inputs.
                // We override with [N, 202] to allow batch evaluation.
                let batch = SymbolTable::default().sym("N");
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid![batch, (T::NUM_INPUTS)])
            } else {
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid![i, (T::NUM_INPUTS)])
            };
            let tract_model = model
                .clone()
                .with_input_fact(0, fact)?
                .into_optimized()?
                .into_runnable()?;
            models.push(tract_model);
        }
        Ok(models)
    }
}

/// The following tests mainly test the quality of the neural nets
#[cfg(test)]
mod tests {
    use crate::evaluator::Evaluator;
    use crate::onnx::OnnxEvaluator;
    use crate::pos;

    #[test]
    fn eval_certain_win_normal() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 1:1; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_normal > 0.85);
        assert!(probabilities.win_normal < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_gammon() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 1:1; o 18:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_gammon > 0.85);
        assert!(probabilities.win_gammon < 0.9); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_win_bg() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 1:1; o 6:15];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.win_bg > 0.27);
        assert!(probabilities.win_bg < 0.32); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_normal() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 1:6; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_normal > 0.77);
        assert!(probabilities.lose_normal < 0.82); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_gammon() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 7:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_gammon > 0.92);
        assert!(probabilities.lose_gammon < 0.98); // This should be wrong, let's improve the nets.
    }

    #[test]
    fn eval_certain_lose_bg() {
        let onnx = OnnxEvaluator::contact_default().unwrap();
        let position = pos![x 19:15; o 24:1];

        let probabilities = onnx.eval(&position);
        assert!(probabilities.lose_bg > 0.02);
        assert!(probabilities.lose_bg < 0.05); // This should be wrong, let's improve the nets.
    }
}
