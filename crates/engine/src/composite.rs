use crate::evaluator::{Evaluator, PartialEvaluator};
use crate::position::{GameResult, GameState, Position};
use crate::probabilities::Probabilities;
use std::default::Default;

pub struct CompositeEvaluator<T: PartialEvaluator, U: Evaluator> {
    partial_evaluator: T,
    evaluator: U,
}

impl<T: PartialEvaluator, U: Evaluator> CompositeEvaluator<T, U> {
    pub fn new(partial: T, evaluator: U) -> CompositeEvaluator<T, U> {
        CompositeEvaluator {
            partial_evaluator: partial,
            evaluator,
        }
    }
}

impl<T: PartialEvaluator, U: Evaluator> Evaluator for CompositeEvaluator<T, U> {
    fn eval(&self, pos: &Position) -> Probabilities {
        self.partial_evaluator
            .try_eval(pos)
            .unwrap_or_else(|| self.evaluator.eval(pos))
    }
}

pub struct GameOverEvaluator {}

impl PartialEvaluator for GameOverEvaluator {
    fn try_eval(&self, pos: &Position) -> Option<Probabilities> {
        if pos.has_lost() {
            match pos.game_state() {
                GameState::GameOver(result) => match result {
                    GameResult::LoseNormal => Some(Probabilities {
                        lose_normal: 1f32,
                        ..Default::default()
                    }),
                    GameResult::LoseGammon => Some(Probabilities {
                        lose_gammon: 1f32,
                        ..Default::default()
                    }),
                    GameResult::LoseBg => Some(Probabilities {
                        lose_bg: 1f32,
                        ..Default::default()
                    }),
                    _ => {
                        unreachable!()
                    }
                },
                _ => unreachable!(),
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::composite::{CompositeEvaluator, GameOverEvaluator};
    use crate::evaluator::{Evaluator, RandomEvaluator};
    use crate::pos;

    #[test]
    fn game_over_lose_normal() {
        let evaluator = CompositeEvaluator::new(GameOverEvaluator {}, RandomEvaluator {});
        let position = pos!(x 12:1; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_normal, 1.0);
        assert_eq!(probabilities.equity(), -1.0);
    }

    #[test]
    fn game_over_lose_gammon() {
        let evaluator = CompositeEvaluator::new(GameOverEvaluator {}, RandomEvaluator {});
        let position = pos!(x 12:15; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_gammon, 1.0);
        assert_eq!(probabilities.equity(), -2.0);
    }

    #[test]
    fn game_over_lose_bg() {
        let evaluator = CompositeEvaluator::new(GameOverEvaluator {}, RandomEvaluator {});
        let position = pos!(x 20:15; o);
        let probabilities = evaluator.eval(&position);
        assert_eq!(probabilities.lose_bg, 1.0);
        assert_eq!(probabilities.equity(), -3.0);
    }

    #[test]
    fn game_over_win_normal() {
        let evaluator = CompositeEvaluator::new(GameOverEvaluator {}, RandomEvaluator {});
        let position = pos!(x 12:1; o).switch_sides();
        let probabilities = evaluator.eval(&position);
        // The following numbers should be random
        assert!(probabilities.lose_normal > 0.0);
        assert!(probabilities.lose_normal < 1.0);
        assert_ne!(probabilities.equity(), 1.0);
    }

    #[test]
    fn game_over_ongoing() {
        let evaluator = CompositeEvaluator::new(GameOverEvaluator {}, RandomEvaluator {});
        let position = pos!(x 1:1; o 2:2).switch_sides();
        let probabilities = evaluator.eval(&position);
        // The following numbers should be random
        assert!(probabilities.lose_normal > 0.0);
        assert!(probabilities.lose_normal < 1.0);
        assert_ne!(probabilities.equity(), 1.0);
    }
}
