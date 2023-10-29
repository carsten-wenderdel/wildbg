use crate::composite::GameOverEvaluator;
use crate::evaluator::{BatchEvaluator, Evaluator, PartialEvaluator};
use crate::inputs::{ContactInputsGen, RaceInputsGen};
use crate::onnx::OnnxEvaluator;
use crate::position::{GamePhase, OngoingPhase, Position};
use crate::probabilities::Probabilities;

pub struct ComplexEvaluator {
    contact_evaluator: OnnxEvaluator<ContactInputsGen>,
    race_evaluator: OnnxEvaluator<RaceInputsGen>,
    game_over_evaluator: GameOverEvaluator,
}

impl BatchEvaluator for ComplexEvaluator {
    fn eval_positions(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        let mut game_over: Vec<(Position, Probabilities)> = Vec::with_capacity(positions.len());
        let mut contact: Vec<Position> = Vec::new();
        let mut race: Vec<Position> = Vec::new();

        for position in positions.into_iter() {
            match position.game_phase() {
                GamePhase::Ongoing(ongoing) => match ongoing {
                    OngoingPhase::Contact => contact.push(position),
                    OngoingPhase::Race => race.push(position),
                },
                GamePhase::GameOver(_) => {
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

impl ComplexEvaluator {
    pub fn try_default() -> Option<Self> {
        let contact_evaluator = OnnxEvaluator::contact_default()?;
        let race_evaluator = OnnxEvaluator::race_default()?;
        Some(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }

    /// Compared to `try_default`, this function takes much longer to execute and the
    /// resulting struct is about 50 times bigger. But rollouts are about 2% faster.
    pub fn try_default_optimized() -> Option<Self> {
        let contact_evaluator = OnnxEvaluator::contact_default_optimized()?;
        let race_evaluator = OnnxEvaluator::race_default_optimized()?;
        Some(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }

    pub fn default_tests() -> Self {
        let contact_evaluator = OnnxEvaluator::contact_default_tests();
        let race_evaluator = OnnxEvaluator::race_default_tests();
        Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        }
    }

    pub fn from_file_paths_optimized(contact_path: &str, race_path: &str) -> Option<Self> {
        let contact_evaluator =
            OnnxEvaluator::from_file_path_optimized(contact_path, ContactInputsGen {})?;
        let race_evaluator = OnnxEvaluator::from_file_path_optimized(race_path, RaceInputsGen {})?;
        Some(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluator::Evaluator;
    use crate::onnx::OnnxEvaluator;
    use crate::pos;
    use crate::position::Position;
    use std::collections::HashMap;

    #[test]
    fn game_over() {
        // Given
        let evaluator = super::ComplexEvaluator::default_tests();
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
        let evaluator = super::ComplexEvaluator::default_tests();
        let contact = pos![x 1:1, 24:1; o 10:1];
        let race = pos![x 1: 1; o 24:1];

        // When
        let positions_and_probabilities = evaluator.eval_batch(vec![contact.clone(), race.clone()]);

        // Then
        assert_eq!(positions_and_probabilities.len(), 2);

        let race_index = positions_and_probabilities
            .iter()
            .position(|(position, _)| position == &race)
            .unwrap();
        assert_eq!(
            positions_and_probabilities[race_index].1,
            OnnxEvaluator::race_default_tests().eval(&race)
        );

        let contact_index = positions_and_probabilities
            .iter()
            .position(|(position, _)| position == &contact)
            .unwrap();
        assert_eq!(
            positions_and_probabilities[contact_index].1,
            OnnxEvaluator::contact_default_tests().eval(&contact)
        );
    }
}
