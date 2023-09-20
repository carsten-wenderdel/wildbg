use crate::bg_move::{BgMove, MoveDetail};
use crate::dice::Dice;
use crate::evaluator::Evaluator;
use crate::onnx::OnnxEvaluator;
use crate::position::Position;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};

pub struct WebApi<T: Evaluator> {
    evaluator: T,
}

impl WebApi<OnnxEvaluator> {
    pub fn try_default() -> Option<Self> {
        OnnxEvaluator::with_default_model().map(|evaluator| Self { evaluator })
    }
}

impl<T: Evaluator> WebApi<T> {
    pub fn new(evaluator: T) -> Self {
        Self { evaluator }
    }

    pub fn get_move(
        &self,
        pip_params: PipParams,
        dice_params: DiceParams,
    ) -> Result<MoveResponse, (StatusCode, String)> {
        let position = Position::try_from(pip_params);
        let dice = Dice::try_from((dice_params.die1, dice_params.die2));
        match position {
            Err(error) => Err((StatusCode::BAD_REQUEST, error.to_string())),
            Ok(position) => match dice {
                Err(error) => Err((StatusCode::BAD_REQUEST, error.to_string())),
                Ok(dice) => {
                    let pos_and_probs = self
                        .evaluator
                        .positions_and_probabilities_by_equity(&position, &dice);
                    let moves: Vec<MoveInfo> = pos_and_probs
                        .into_iter()
                        .map(|(new_pos, probabilities)| {
                            let bg_move = BgMove::new(&position, &new_pos, &dice);
                            let play = bg_move.into_details();
                            let probabilities = probabilities.into(); // convert model into view model
                            MoveInfo {
                                play,
                                probabilities,
                            }
                        })
                        .collect();
                    Ok(MoveResponse { moves })
                }
            },
        }
    }
}

#[derive(Serialize)]
pub struct MoveResponse {
    moves: Vec<MoveInfo>,
}

#[derive(Serialize)]
pub struct MoveInfo {
    play: Vec<MoveDetail>,
    probabilities: Probabilities,
}

// This is similar to evaluator::Probabilities. But while the former serves
// as a model for calculations, this is more like a view model for the web API.
// While in evaluator::Probabilities all 6 numbers add up to 1.0, this is different.
// `win` includes the chances to win gammon or BG.
// `lose` is not given, you can calculate it by through `1 - win`.
// `winG` includes the chances to win BG and `loseG` includes the chance to lose BG.
// This way we use the same format as earlier engines like GnuBG have done.
#[derive(Serialize)]
#[allow(non_snake_case)]
pub struct Probabilities {
    pub(crate) win: f32,
    pub(crate) winG: f32,
    pub(crate) winBg: f32,
    pub(crate) loseG: f32,
    pub(crate) loseBg: f32,
}

impl From<crate::evaluator::Probabilities> for Probabilities {
    fn from(value: crate::evaluator::Probabilities) -> Self {
        Self {
            win: value.win_normal + value.win_gammon + value.win_bg,
            winG: value.win_gammon + value.win_bg,
            winBg: value.win_bg,
            loseG: value.lose_gammon + value.lose_bg,
            loseBg: value.lose_bg,
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

#[cfg(test)]
mod probabilities_tests {
    #[test]
    fn from() {
        let model_probs = crate::evaluator::Probabilities {
            win_normal: 0.32,
            win_gammon: 0.26,
            win_bg: 0.12,
            lose_normal: 0.15,
            lose_gammon: 0.1,
            lose_bg: 0.05,
        };

        let view_probs: crate::web_api::Probabilities = model_probs.into();
        assert_eq!(view_probs.win, 0.7);
        assert_eq!(view_probs.winG, 0.38);
        assert_eq!(view_probs.winBg, 0.12);
        assert_eq!(view_probs.loseG, 0.15);
        assert_eq!(view_probs.loseBg, 0.05);
    }
}
