use crate::bg_move::{BgMove, MoveDetail};
use crate::dice::Dice;
use crate::evaluator::Evaluator;
use crate::onnx::OnnxEvaluator;
use crate::position::Position;
use hyper::StatusCode;
use serde::Deserialize;

pub struct WebApi {
    evaluator: OnnxEvaluator,
}

impl WebApi {
    pub fn try_default() -> Option<Self> {
        OnnxEvaluator::with_default_model().map(|evaluator| Self { evaluator })
    }

    /// Currently this returns a static move. Work in progress
    pub fn get_move(
        &self,
        pip_params: PipParams,
        dice_params: DiceParams,
    ) -> Result<Vec<MoveDetail>, (StatusCode, String)> {
        let position = Position::try_from(pip_params);
        let dice = Dice::try_from((dice_params.die1, dice_params.die2));
        match position {
            Err(error) => Err((StatusCode::BAD_REQUEST, error.to_string())),
            Ok(position) => match dice {
                Err(error) => Err((StatusCode::BAD_REQUEST, error.to_string())),
                Ok(dice) => {
                    let new = self.evaluator.best_position(&position, &dice);
                    let bg_move = BgMove::new(&position, &new.switch_sides(), &dice);
                    Ok(bg_move.into_details())
                }
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct DiceParams {
    die1: usize,
    die2: usize,
}

#[derive(Debug, Deserialize)]
pub struct PipParams {
    /// Bar for player `o`.
    p0: Option<i8>,
    p1: Option<i8>,
    p2: Option<i8>,
    p3: Option<i8>,
    p4: Option<i8>,
    p5: Option<i8>,
    p6: Option<i8>,
    p7: Option<i8>,
    p8: Option<i8>,
    p9: Option<i8>,
    p10: Option<i8>,
    p11: Option<i8>,
    p12: Option<i8>,
    p13: Option<i8>,
    p14: Option<i8>,
    p15: Option<i8>,
    p16: Option<i8>,
    p17: Option<i8>,
    p18: Option<i8>,
    p19: Option<i8>,
    p20: Option<i8>,
    p21: Option<i8>,
    p22: Option<i8>,
    p23: Option<i8>,
    p24: Option<i8>,
    /// Bar for player `x`.
    p25: Option<i8>,
}

impl TryFrom<PipParams> for Position {
    type Error = &'static str;

    fn try_from(params: PipParams) -> Result<Self, Self::Error> {
        // let params = params.pips;
        let pips: [i8; 26] = [
            params.p0.unwrap_or_default(),
            params.p1.unwrap_or_default(),
            params.p2.unwrap_or_default(),
            params.p3.unwrap_or_default(),
            params.p4.unwrap_or_default(),
            params.p5.unwrap_or_default(),
            params.p6.unwrap_or_default(),
            params.p7.unwrap_or_default(),
            params.p8.unwrap_or_default(),
            params.p9.unwrap_or_default(),
            params.p10.unwrap_or_default(),
            params.p11.unwrap_or_default(),
            params.p12.unwrap_or_default(),
            params.p13.unwrap_or_default(),
            params.p14.unwrap_or_default(),
            params.p15.unwrap_or_default(),
            params.p16.unwrap_or_default(),
            params.p17.unwrap_or_default(),
            params.p18.unwrap_or_default(),
            params.p19.unwrap_or_default(),
            params.p20.unwrap_or_default(),
            params.p21.unwrap_or_default(),
            params.p22.unwrap_or_default(),
            params.p23.unwrap_or_default(),
            params.p24.unwrap_or_default(),
            params.p25.unwrap_or_default(),
        ];
        Position::try_from(pips)
    }
}
