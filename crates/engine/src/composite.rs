use crate::evaluator::{BatchEvaluator, Evaluator, PartialEvaluator};
use crate::inputs::{ContactInputsGen, RaceInputsGen};
use crate::onnx::OnnxEvaluator;
use crate::position::{GamePhase, GameResult, GameState, OngoingPhase, Position};
use crate::probabilities::Probabilities;

type Error = String;

/// Evaluates each position with the matching of three evaluators: contact, race, game over.
///
/// This is pretty much the same as the "Composite" GoF design pattern.
pub struct CompositeEvaluator {
    contact_evaluator: OnnxEvaluator<ContactInputsGen>,
    race_evaluator: OnnxEvaluator<RaceInputsGen>,
    game_over_evaluator: GameOverEvaluator,
}

impl BatchEvaluator for CompositeEvaluator {
    fn eval_positions(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        let length = positions.len();
        let mut game_over: Vec<(Position, Probabilities)> = Vec::with_capacity(length);
        let mut contact: Vec<Position> = Vec::new();
        let mut race: Vec<Position> = Vec::new();

        for position in positions.into_iter() {
            match position.game_phase() {
                GamePhase::Ongoing(ongoing) => match ongoing {
                    OngoingPhase::Contact => {
                        if contact.is_empty() {
                            contact.reserve_exact(length)
                        }
                        contact.push(position)
                    }
                    OngoingPhase::Race => {
                        if race.is_empty() {
                            race.reserve_exact(length);
                        }
                        race.push(position);
                    }
                },
                GamePhase::GameOver(_) => {
                    // The `try_eval` runs `match pos.game_state()` a second time.
                    // This is not a performance problem for rollouts: When the array of positions/moves contains a
                    // position which is game over, the `CompositeEvaluator` is not used anyway, so we never get here.
                    let probabilities = self
                        .game_over_evaluator
                        .try_eval(&position)
                        .expect("GameOver must be handled by this.");
                    game_over.push((position, probabilities));
                }
            }
        }
        let mut contact = self.contact_evaluator.eval_batch(contact);
        let mut race = self.race_evaluator.eval_batch(race);
        game_over.append(&mut contact);
        game_over.append(&mut race);

        game_over
    }
}

impl CompositeEvaluator {
    pub fn try_default() -> Result<Self, Error> {
        let contact_evaluator = OnnxEvaluator::contact_default()?;
        let race_evaluator = OnnxEvaluator::race_default()?;
        Ok(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }

    /// Compared to `try_default`, this function takes much longer to execute and the
    /// resulting struct is about 50 times bigger. But rollouts are about 2% faster.
    pub fn try_default_optimized() -> Result<Self, Error> {
        let contact_evaluator = OnnxEvaluator::contact_default_optimized()?;
        let race_evaluator = OnnxEvaluator::race_default_optimized()?;
        Ok(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }

    pub fn default_tests() -> Self {
        let contact_evaluator = OnnxEvaluator::contact_default().unwrap();
        let race_evaluator = OnnxEvaluator::race_default().unwrap();
        Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        }
    }

    pub fn from_file_paths_optimized(contact_path: &str, race_path: &str) -> Result<Self, Error> {
        let contact_evaluator =
            OnnxEvaluator::from_file_path_optimized(contact_path, ContactInputsGen {})?;
        let race_evaluator = OnnxEvaluator::from_file_path_optimized(race_path, RaceInputsGen {})?;
        Ok(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }
}

struct GameOverEvaluator {}

impl PartialEvaluator for GameOverEvaluator {
    fn try_eval(&self, pos: &Position) -> Option<Probabilities> {
        match pos.game_state() {
            GameState::Ongoing => None,
            GameState::GameOver(result) => match result {
                GameResult::WinNormal => Some(Probabilities {
                    win_normal: 1.,
                    ..Default::default()
                }),
                GameResult::WinGammon => Some(Probabilities {
                    win_gammon: 1.,
                    ..Default::default()
                }),
                GameResult::WinBg => Some(Probabilities {
                    win_bg: 1.,
                    ..Default::default()
                }),
                GameResult::LoseNormal => Some(Probabilities {
                    lose_normal: 1.,
                    ..Default::default()
                }),
                GameResult::LoseGammon => Some(Probabilities {
                    lose_gammon: 1.,
                    ..Default::default()
                }),
                GameResult::LoseBg => Some(Probabilities {
                    lose_bg: 1.,
                    ..Default::default()
                }),
            },
        }
    }
}

#[cfg(test)]
mod composite_tests {
    use crate::evaluator::Evaluator;
    use crate::onnx::OnnxEvaluator;
    use crate::pos;

    #[test]
    fn game_over() {
        // Given
        let evaluator = super::CompositeEvaluator::default_tests();
        let pos_1 = pos![x 1:1; o];
        let pos_2 = pos![x 1: 1; o 24:1];

        // When
        let positions_and_probabilities = evaluator.eval_batch(vec![pos_1, pos_2]);

        // Then
        assert_eq!(positions_and_probabilities.len(), 2);
        let game_over_index = positions_and_probabilities
            .iter()
            .position(|(position, _)| position.has_lost())
            .unwrap();
        assert_eq!(
            positions_and_probabilities[game_over_index].1.equity(),
            -1.0
        );
    }

    #[test]
    fn uses_correct_evaluator_for_race_and_contact() {
        // Given
        let evaluator = super::CompositeEvaluator::default_tests();
        let contact = pos![x 1:1, 24:1; o 10:1];
        let race = pos![x 1: 1; o 24:1];

        // When
        let positions_and_probabilities = evaluator.eval_batch(vec![contact, race]);

        // Then
        assert_eq!(positions_and_probabilities.len(), 2);

        let race_index = positions_and_probabilities
            .iter()
            .position(|(position, _)| position == &race)
            .unwrap();
        assert_eq!(
            positions_and_probabilities[race_index].1,
            OnnxEvaluator::race_default().unwrap().eval(&race)
        );

        let contact_index = positions_and_probabilities
            .iter()
            .position(|(position, _)| position == &contact)
            .unwrap();
        assert_eq!(
            positions_and_probabilities[contact_index].1,
            OnnxEvaluator::contact_default().unwrap().eval(&contact)
        );
    }
}

#[cfg(test)]
mod game_over_tests {
    use crate::evaluator::Evaluator;
    use crate::pos;

    #[test]
    fn game_over_lose_normal() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 12:1; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_normal, 1.);
        assert_eq!(probabilities.equity(), -1.);
    }

    #[test]
    fn game_over_lose_gammon() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 12:15; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_gammon, 1.);
        assert_eq!(probabilities.equity(), -2.);
    }

    #[test]
    fn game_over_lose_bg() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 20:15; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_bg, 1.);
        assert_eq!(probabilities.equity(), -3.);
    }

    #[test]
    fn game_over_win_normal() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 12:1; o).sides_switched();
        let probabilities = evaluator.eval(&position);
        // The following numbers should be random
        assert_eq!(probabilities.win_normal, 1.);
        assert_eq!(probabilities.equity(), 1.);
    }

    #[test]
    fn game_over_win_gammon() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 15:15; o).sides_switched();
        let probabilities = evaluator.eval(&position);
        // The following numbers should be random
        assert_eq!(probabilities.win_gammon, 1.);
        assert_eq!(probabilities.equity(), 2.);
    }

    #[test]
    fn game_over_win_backgammon() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 22:15; o).sides_switched();
        let probabilities = evaluator.eval(&position);
        // The following numbers should be random
        assert_eq!(probabilities.win_bg, 1.);
        assert_eq!(probabilities.equity(), 3.);
    }

    #[test]
    fn game_over_ongoing() {
        let evaluator = super::CompositeEvaluator::default_tests();
        let position = pos!(x 1:1; o 2:2).sides_switched();
        let probabilities = evaluator.eval(&position);
        // The probabilities now come from the onnx evaluator
        assert!(probabilities.equity() < 0.);
        assert!(probabilities.equity() > -1.);
    }
}
