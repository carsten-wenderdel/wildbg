use engine::composite::CompositeEvaluator;
use engine::dice::Dice;
use engine::multiply::MultiPlyEvaluator;
use engine::position::{GamePhase, GameResult, Position, STARTING};
use engine::probabilities::Probabilities;
use logic::bg_move::BgMove;
use logic::wildbg_api::{ScoreConfig, WildbgApi};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, PartialEq, Serialize)]
pub struct MoveStep {
    pub from: usize,
    pub to: usize,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct ProbabilityView {
    pub win: f32,
    pub win_gammon: f32,
    pub win_backgammon: f32,
    pub lose_gammon: f32,
    pub lose_backgammon: f32,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct AnalyzedMove {
    pub play: Vec<MoveStep>,
    pub position: Vec<i8>,
    pub equity: f32,
    pub score: f32,
    pub probabilities: ProbabilityView,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Analysis {
    pub moves: Vec<AnalyzedMove>,
    pub phase: &'static str,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct PositionEvaluation {
    pub equity: f32,
    pub phase: &'static str,
    pub probabilities: ProbabilityView,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct CubeDecision {
    pub should_double: bool,
    pub should_take: bool,
    pub equity_cubeless: f32,
    pub equity_no_double: f32,
    pub equity_double_take: f32,
}

fn probability_view(value: &Probabilities) -> ProbabilityView {
    ProbabilityView {
        win: value.win(),
        win_gammon: value.win_gammon + value.win_bg,
        win_backgammon: value.win_bg,
        lose_gammon: value.lose_gammon + value.lose_bg,
        lose_backgammon: value.lose_bg,
    }
}

fn position_from_vec(pips: Vec<i8>) -> Result<Position, String> {
    let pips: [i8; 26] = pips
        .try_into()
        .map_err(|_| "A WildBG position must contain exactly 26 points.".to_string())?;
    Position::try_from(pips).map_err(|e| e.to_string())
}

fn phase_name(position: &Position) -> &'static str {
    match position.game_phase() {
        GamePhase::Ongoing(phase) => match phase {
            engine::position::OngoingPhase::Contact => "contact",
            engine::position::OngoingPhase::Race => "race",
        },
        GamePhase::GameOver(_) => "game-over",
    }
}

#[wasm_bindgen]
pub struct Wildbg {
    api: WildbgApi<MultiPlyEvaluator<CompositeEvaluator>>,
}

impl Wildbg {
    pub fn new_internal() -> Result<Self, String> {
        let evaluator = CompositeEvaluator::try_default()?;
        let api = WildbgApi::with_evaluator(MultiPlyEvaluator { evaluator });
        Ok(Self { api })
    }

    pub fn analyze_internal(
        &self,
        pips: Vec<i8>,
        die_one: usize,
        die_two: usize,
        one_pointer: bool,
    ) -> Result<Analysis, String> {
        let position = position_from_vec(pips)?;
        let dice = Dice::try_from((die_one, die_two)).map_err(|e| e.to_string())?;
        let config = if one_pointer {
            ScoreConfig::OnePointer
        } else {
            ScoreConfig::MoneyGame
        };

        let moves = self
            .api
            .all_moves(&position, &dice, &config)
            .into_iter()
            .map(|(new_position, probabilities)| {
                let bg_move = BgMove::new(&position, &new_position, &dice);
                let score = if one_pointer {
                    probabilities.win()
                } else {
                    probabilities.equity()
                };
                let play = bg_move
                    .into_details()
                    .into_iter()
                    .map(|step| MoveStep {
                        from: step.from(),
                        to: step.to(),
                    })
                    .collect();
                let position_array: [i8; 26] = new_position.into();

                AnalyzedMove {
                    play,
                    position: position_array.to_vec(),
                    equity: probabilities.equity(),
                    score,
                    probabilities: probability_view(&probabilities),
                }
            })
            .collect();

        Ok(Analysis {
            moves,
            phase: phase_name(&position),
        })
    }

    pub fn evaluate_internal(&self, pips: Vec<i8>) -> Result<PositionEvaluation, String> {
        let position = position_from_vec(pips)?;
        let probabilities = self.api.probabilities(&position);
        Ok(PositionEvaluation {
            equity: probabilities.equity(),
            phase: phase_name(&position),
            probabilities: probability_view(&probabilities),
        })
    }

    pub fn cube_info_internal(&self, pips: Vec<i8>) -> Result<CubeDecision, String> {
        let position = position_from_vec(pips)?;
        let cube = self.api.cube_info(&position);
        Ok(CubeDecision {
            should_double: cube.double(),
            should_take: cube.accept(),
            equity_cubeless: cube.equity_cubeless(),
            equity_no_double: cube.equity_no_double(),
            equity_double_take: cube.equity_double_take(),
        })
    }

    pub fn result_internal(&self, pips: Vec<i8>) -> Result<String, String> {
        let position = position_from_vec(pips)?;
        let result = match position.game_phase() {
            GamePhase::Ongoing(_) => "ongoing",
            GamePhase::GameOver(result) => match result {
                GameResult::WinNormal => "win",
                GameResult::WinGammon => "win-gammon",
                GameResult::WinBg => "win-backgammon",
                GameResult::LoseNormal => "loss",
                GameResult::LoseGammon => "loss-gammon",
                GameResult::LoseBg => "loss-backgammon",
            },
        };
        Ok(result.to_string())
    }
}

#[wasm_bindgen]
impl Wildbg {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Wildbg, JsValue> {
        console_error_panic_hook::set_once();
        Self::new_internal().map_err(|e| JsValue::from_str(&e))
    }

    pub fn analyze(
        &self,
        pips: Vec<i8>,
        die_one: usize,
        die_two: usize,
        one_pointer: bool,
    ) -> Result<JsValue, JsValue> {
        let analysis = self
            .analyze_internal(pips, die_one, die_two, one_pointer)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&analysis).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn evaluate(&self, pips: Vec<i8>) -> Result<JsValue, JsValue> {
        let evaluation = self
            .evaluate_internal(pips)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&evaluation).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn cube_info(&self, pips: Vec<i8>) -> Result<JsValue, JsValue> {
        let decision = self
            .cube_info_internal(pips)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&decision).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn result(&self, pips: Vec<i8>) -> Result<String, JsValue> {
        self.result_internal(pips)
            .map_err(|e| JsValue::from_str(&e))
    }
}

#[wasm_bindgen]
pub fn starting_position() -> Vec<i8> {
    let pips: [i8; 26] = STARTING.into();
    pips.to_vec()
}

#[wasm_bindgen]
pub fn wildbg_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn wildbg_revision() -> String {
    env!("WILDBG_GIT_HASH").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::pos;

    /// x has almost finished bearing off (a single checker deep in its own
    /// home board, 14 already off), while o has not moved a single checker
    /// and is stuck far away from its own home board. This should be an
    /// overwhelming win for x: high `win`, a real chance of `win_gammon`,
    /// and `lose`-side fields near zero.
    fn heavily_winning_position() -> Vec<i8> {
        let position = pos!(x 2:1; o 12:15);
        let pips: [i8; 26] = position.into();
        pips.to_vec()
    }

    /// Mirror image of `heavily_winning_position`: x has not moved a single
    /// checker and is stuck far from its home board, while o has almost
    /// finished bearing off. This should be an overwhelming loss for x.
    fn heavily_losing_position() -> Vec<i8> {
        let position = pos!(x 12:15; o 2:1);
        let pips: [i8; 26] = position.into();
        pips.to_vec()
    }

    #[test]
    fn test_probability_view_field_mapping_winning_position() {
        let engine = Wildbg::new_internal().expect("engine creation should succeed");
        let eval = engine
            .evaluate_internal(heavily_winning_position())
            .expect("evaluation should succeed");
        let p = &eval.probabilities;

        // Win-side fields should dominate.
        assert!(p.win > 0.9, "expected win > 0.9, got {}", p.win);
        assert!(
            p.win_gammon > 0.5,
            "expected win_gammon > 0.5, got {}",
            p.win_gammon
        );

        // Lose-side fields should be negligible.
        assert!(
            p.lose_gammon < 0.05,
            "expected lose_gammon < 0.05, got {}",
            p.lose_gammon
        );
        assert!(
            p.lose_backgammon < 0.05,
            "expected lose_backgammon < 0.05, got {}",
            p.lose_backgammon
        );

        // Cumulative gammon fields must be at least as large as their
        // backgammon-only counterparts, since gammon includes backgammon.
        assert!(p.win_gammon >= p.win_backgammon);
        assert!(p.lose_gammon >= p.lose_backgammon);
    }

    #[test]
    fn test_probability_view_field_mapping_losing_position() {
        let engine = Wildbg::new_internal().expect("engine creation should succeed");
        let eval = engine
            .evaluate_internal(heavily_losing_position())
            .expect("evaluation should succeed");
        let p = &eval.probabilities;

        // Lose-side fields should dominate.
        assert!(p.win < 0.1, "expected win < 0.1, got {}", p.win);
        assert!(
            p.lose_gammon > 0.5,
            "expected lose_gammon > 0.5, got {}",
            p.lose_gammon
        );

        // Win-side fields should be negligible.
        assert!(
            p.win_gammon < 0.05,
            "expected win_gammon < 0.05, got {}",
            p.win_gammon
        );
        assert!(
            p.win_backgammon < 0.05,
            "expected win_backgammon < 0.05, got {}",
            p.win_backgammon
        );

        // Cumulative gammon fields must be at least as large as their
        // backgammon-only counterparts, since gammon includes backgammon.
        assert!(p.win_gammon >= p.win_backgammon);
        assert!(p.lose_gammon >= p.lose_backgammon);
    }

    #[test]
    fn test_starting_position() {
        let pos = starting_position();
        assert_eq!(pos.len(), 26);
        assert_eq!(pos[1], -2);
        assert_eq!(pos[6], 5);
        assert_eq!(pos[8], 3);
        assert_eq!(pos[12], -5);
        assert_eq!(pos[13], 5);
        assert_eq!(pos[17], -3);
        assert_eq!(pos[19], -5);
        assert_eq!(pos[24], 2);
    }

    #[test]
    fn test_wildbg_version_and_revision() {
        assert!(!wildbg_version().is_empty());
        assert!(!wildbg_revision().is_empty());
    }

    #[test]
    fn test_engine_analyze_evaluate_cube_and_result() {
        let engine = Wildbg::new_internal().expect("engine creation should succeed");
        let start = starting_position();

        // 1. Result on starting position
        let res = engine
            .result_internal(start.clone())
            .expect("result should succeed");
        assert_eq!(res, "ongoing");

        // 2. Evaluate starting position
        let eval = engine
            .evaluate_internal(start.clone())
            .expect("evaluation should succeed");
        assert_eq!(eval.phase, "contact");
        assert!(eval.probabilities.win > 0.45 && eval.probabilities.win < 0.55);

        // 3. Cube info on starting position
        let cube = engine
            .cube_info_internal(start.clone())
            .expect("cube info should succeed");
        assert!(!cube.should_double);
        assert!(cube.should_take);

        // 4. Analyze opening roll (3-1)
        let analysis = engine
            .analyze_internal(start, 3, 1, false)
            .expect("analysis should succeed");
        assert_eq!(analysis.phase, "contact");
        assert_eq!(analysis.moves.len(), 16);
        // Best opening 3-1 move makes the 5 point: 8/5, 6/5
        let best_move = &analysis.moves[0];
        assert_eq!(
            best_move.play,
            vec![MoveStep { from: 8, to: 5 }, MoveStep { from: 6, to: 5 }]
        );

        // 5. Evaluate game-over position (does not panic or debug_assert fail)
        let mut won_pips = vec![0i8; 26];
        won_pips[1] = -2; // opponent still has checkers, we have none on board (15 off)
        let won_res = engine.result_internal(won_pips.clone()).unwrap();
        assert_eq!(won_res, "win");
        let won_eval = engine.evaluate_internal(won_pips.clone()).unwrap();
        assert_eq!(won_eval.phase, "game-over");
        assert_eq!(won_eval.equity, 1.0);
    }
}
