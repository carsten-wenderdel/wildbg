use crate::dice::Dice;
use crate::position::Position;
use crate::probabilities::Probabilities;

pub trait Evaluator {
    /// Returns a cubeless evaluation of a position.
    /// Implementing types will calculate the probabilities with different strategies.
    /// Examples of such strategies are a rollout or 1-ply inference of a neural net.
    fn eval(&self, pos: &Position) -> Probabilities;

    /// Returns the position after applying the *best* move to `pos`.
    /// The returned `Position` has already switches sides.
    /// This means the returned position will have the *lowest* equity of possible positions.
    fn best_position(&self, pos: &Position, dice: &Dice) -> Position {
        self.worst_position(&pos.all_positions_after_moving(dice))
            .clone()
    }

    /// Worst position might be interesting, because when you switch sides, it's suddenly the best.
    fn worst_position<'a>(&'a self, positions: &'a [Position]) -> &Position {
        positions
            .iter()
            .map(|pos| (pos, self.eval(pos).equity()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    /// All legal positions after moving with the given dice.
    /// Sorted, the best move/position is first in the vector.
    /// The positions are again from the perspective of player `x`.
    /// The probabilities have switched sides, so they are from the perspective of player `x` who has to move.
    fn positions_and_probabilities_by_equity(
        &self,
        position: &Position,
        dice: &Dice,
    ) -> Vec<(Position, Probabilities)> {
        let after_moving = position.all_positions_after_moving(dice);
        let mut pos_and_probs: Vec<(Position, Probabilities)> = after_moving
            .into_iter()
            .map(|pos| {
                let probabilities = self.eval(&pos).switch_sides();
                let pos = pos.switch_sides();
                (pos, probabilities)
            })
            .collect();
        pos_and_probs.sort_unstable_by(|a, b| b.1.equity().partial_cmp(&a.1.equity()).unwrap());
        pos_and_probs
    }
}

pub struct RandomEvaluator {}

impl Evaluator for RandomEvaluator {
    #[allow(dead_code)]
    /// Returns random probabilities. Each call will return different values.
    fn eval(&self, _pos: &Position) -> Probabilities {
        let win_normal = fastrand::f32();
        let win_gammon = fastrand::f32();
        let win_bg = fastrand::f32();
        let lose_normal = fastrand::f32();
        let lose_gammon = fastrand::f32();
        let lose_bg = fastrand::f32();

        // Now we like to make sure that the different probabilities add up to 1
        let sum = win_normal + win_gammon + win_bg + lose_normal + lose_gammon + lose_bg;
        Probabilities {
            win_normal: win_normal / sum,
            win_gammon: win_gammon / sum,
            win_bg: win_bg / sum,
            lose_normal: lose_normal / sum,
            lose_gammon: lose_gammon / sum,
            lose_bg: lose_bg / sum,
        }
    }
}

#[cfg(test)]
mod evaluator_trait_tests {
    use crate::dice::Dice;
    use crate::evaluator::{Evaluator, Probabilities};
    use crate::pos;
    use crate::position::Position;
    use std::collections::HashMap;

    fn expected_pos() -> Position {
        pos!(x 5:1, 3:1; o 20:2).switch_sides()
    }

    /// Test double. Returns not so good probabilities for `expected_pos`, better for everything else.
    struct EvaluatorFake {}
    impl Evaluator for EvaluatorFake {
        fn eval(&self, pos: &Position) -> Probabilities {
            if pos == &expected_pos() {
                Probabilities {
                    win_normal: 0.5,
                    win_gammon: 0.1,
                    win_bg: 0.1,
                    lose_normal: 0.1,
                    lose_gammon: 0.1,
                    lose_bg: 0.1,
                }
            } else {
                Probabilities {
                    win_normal: 0.4,
                    win_gammon: 0.2,
                    win_bg: 0.1,
                    lose_normal: 0.1,
                    lose_gammon: 0.1,
                    lose_bg: 0.1,
                }
            }
        }
    }

    #[test]
    fn best_position() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = EvaluatorFake {};
        // When
        let best_pos = evaluator.best_position(&given_pos, &Dice::new(4, 2));
        // Then
        assert_eq!(best_pos, expected_pos());
    }

    #[test]
    fn positions_and_probabilities_by_equity() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = EvaluatorFake {};
        // When
        let values = evaluator.positions_and_probabilities_by_equity(&given_pos, &Dice::new(4, 2));
        // Then
        let (best_pos, best_probability) = values.first().unwrap();
        let best_pos = best_pos.switch_sides();
        assert_eq!(
            &best_pos,
            &evaluator.best_position(&given_pos, &Dice::new(4, 2))
        );
        assert_eq!(best_probability.switch_sides(), evaluator.eval(&best_pos));
    }
}

#[cfg(test)]
mod random_evaluator_tests {
    use crate::evaluator::{Evaluator, RandomEvaluator};
    use crate::position;

    #[test]
    fn sum_is_1() {
        let evaluator = RandomEvaluator {};
        let p = evaluator.eval(&position::STARTING);
        let sum =
            p.win_normal + p.win_gammon + p.win_bg + p.lose_normal + p.lose_gammon + p.lose_bg;
        assert!((sum - 1.0).abs() < 0.0001);
    }
}
