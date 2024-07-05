use engine::composite::CompositeEvaluator;
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::position::Position;
use hyper::StatusCode;
use logic::bg_move::{BgMove, MoveDetail};
use logic::cube::CubeInfo;
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

pub struct WebApi<T: Evaluator> {
    evaluator: T,
}

impl WebApi<CompositeEvaluator> {
    pub fn try_default() -> Option<Self> {
        match CompositeEvaluator::try_default() {
            Ok(evaluator) => Some(Self { evaluator }),
            Err(_) => None,
        }
    }
}

impl<T: Evaluator> WebApi<T> {
    pub fn new(evaluator: T) -> Self {
        Self { evaluator }
    }

    pub fn get_eval(&self, pip_params: PipParams) -> Result<EvalResponse, (StatusCode, String)> {
        let position = Position::try_from(pip_params);
        match position {
            Err(error) => Err((StatusCode::BAD_REQUEST, error.to_string())),
            Ok(position) => {
                let evaluation = self.evaluator.eval(&position);
                let cube = CubeInfo::from(&evaluation);
                let probabilities = ProbabilitiesView::from(evaluation);
                Ok(EvalResponse {
                    cube,
                    probabilities,
                })
            }
        }
    }

    pub fn get_move(
        &self,
        pip_params: PipParams,
        dice_params: DiceParams,
    ) -> Result<MoveResponse, &'static str> {
        let position = Position::try_from(pip_params)?;
        let dice = Dice::try_from((dice_params.die1, dice_params.die2))?;
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
}

#[derive(Serialize, ToSchema)]
/// The whole body of the HTTP response.
/// Contains the probabilities for this position and cube decisions.
pub struct EvalResponse {
    cube: CubeInfo,
    probabilities: ProbabilitiesView,
}

#[derive(Serialize, ToSchema)]
/// The whole body of the HTTP response. Contains the list of all legal moves.
pub struct MoveResponse {
    /// The array is ordered by match equity. First move is the best one.
    /// If moving checkers is not possible, the array contains exactly one move
    /// and the `play` array is empty.
    #[schema(minimum = 0)]
    moves: Vec<MoveInfo>,
}

#[derive(Serialize, ToSchema)]
/// This represents one complete move.
pub struct MoveInfo {
    /// Contains 0 to 4 elements for moving a single checker.
    /// If no move is possible because everything is blocked, the array is empty.
    /// If the dice are different, the array contains up to 2 elements.
    /// If the dice are identical (double roll), the array contains up to 4 elements.
    #[schema(minimum = 0, maximum = 4)]
    play: Vec<MoveDetail>,
    probabilities: ProbabilitiesView,
}

// This is similar to evaluator::Probabilities. But while the former serves
// as a model for calculations, this is more like a view model for the web API.
// While in evaluator::Probabilities all 6 numbers add up to 1.0, this is different.
/// `win` includes the chances to win gammon or BG.
/// `winG` includes the chances to win BG and `loseG` includes the chance to lose BG.
/// This way we use the same format as earlier engines like GnuBG have done.
/// `lose` is not given, you can calculate it through `1 - win`.
#[derive(Serialize, ToSchema)]
#[allow(non_snake_case)]
#[schema(title = "Probabilities")]
pub struct ProbabilitiesView {
    /// Probability to win normal, gammon or backgammon
    #[schema(minimum = 0, maximum = 1)]
    pub(crate) win: f32,
    /// Probability to win gammon or backgammon
    #[schema(minimum = 0, maximum = 1)]
    pub(crate) winG: f32,
    /// Probability to win backgammon
    #[schema(minimum = 0, maximum = 1)]
    pub(crate) winBg: f32,
    /// Probability to lose gammon or backgammon
    #[schema(minimum = 0, maximum = 1)]
    pub(crate) loseG: f32,
    /// Probability to lose backgammon
    #[schema(minimum = 0, maximum = 1)]
    pub(crate) loseBg: f32,
}

impl From<engine::probabilities::Probabilities> for ProbabilitiesView {
    fn from(value: engine::probabilities::Probabilities) -> Self {
        Self {
            win: value.win_normal + value.win_gammon + value.win_bg,
            winG: value.win_gammon + value.win_bg,
            winBg: value.win_bg,
            loseG: value.lose_gammon + value.lose_bg,
            loseBg: value.lose_bg,
        }
    }
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct DiceParams {
    #[param(minimum = 1, maximum = 6, example = 3)]
    die1: usize,
    #[param(minimum = 1, maximum = 6, example = 1)]
    die2: usize,
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct PipParams {
    /// Bar for the opponent `o`.
    #[param(minimum = -15, maximum = 0)]
    p0: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = -2)]
    p1: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p2: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p3: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p4: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p5: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = 5)]
    p6: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p7: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = 3)]
    p8: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p9: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p10: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p11: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = -5)]
    p12: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = 5)]
    p13: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p14: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p15: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p16: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = -3)]
    p17: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p18: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = -5)]
    p19: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p20: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p21: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p22: Option<i8>,
    #[param(minimum = -15, maximum = 15)]
    p23: Option<i8>,
    #[param(minimum = -15, maximum = 15, example = 2)]
    p24: Option<i8>,
    /// Bar for your player `x`.
    #[param(minimum = 0, maximum = 15)]
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
        let model_probs = engine::probabilities::Probabilities {
            win_normal: 0.32,
            win_gammon: 0.26,
            win_bg: 0.12,
            lose_normal: 0.15,
            lose_gammon: 0.1,
            lose_bg: 0.05,
        };

        let view_probs: crate::web_api::ProbabilitiesView = model_probs.into();
        assert_eq!(view_probs.win, 0.7);
        assert_eq!(view_probs.winG, 0.38);
        assert_eq!(view_probs.winBg, 0.12);
        assert_eq!(view_probs.loseG, 0.15);
        assert_eq!(view_probs.loseBg, 0.05);
    }
}
