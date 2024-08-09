use crate::dice::Dice;
use crate::position::Position;
use crate::probabilities::Probabilities;

/// A `PartialEvaluator` can only evaluate certain positions, for example only backgames or only bearoffs.
pub trait PartialEvaluator {
    /// Return `None` if the position can not be evaluated.
    ///
    /// Return `Some(Probabilities)` if the the position can be evaluated.
    fn try_eval(&self, pos: &Position) -> Option<Probabilities>;
}

impl<T: Evaluator> PartialEvaluator for T {
    /// Will always return `Some(Probabilities`) as all positions can be evaluated.
    fn try_eval(&self, pos: &Position) -> Option<Probabilities> {
        Some(self.eval(pos))
    }
}

/// [Evaluator] is one of the central parts of the engine. Implementing structs only have to
/// implement the function [Evaluator::eval], Examples are [crate::composite::GameOverEvaluator]
/// and `RolloutEvaluator`.
///
/// The function [Evaluator::eval_batch] is implemented by default, building on `eval`.
/// If there is way to optimize evaluations by looking at all legal moves at once, then don't
/// implement `eval_batch` yourself, instead implement [BatchEvaluator].
pub trait Evaluator {
    /// Returns a cubeless evaluation of a position.
    /// Implementing types will calculate the probabilities with different strategies.
    /// Examples of such strategies are a rollout or 1-ply inference of a neural net.
    fn eval(&self, pos: &Position) -> Probabilities;

    /// Evaluates several positions at once. For optimization it's assumed that those positions
    /// are the result of *all* legal moves following a certain position/dice combination.
    /// The positions have to be switched, so they are from the point of view of the opponent, not
    /// the player who would move.
    fn eval_batch(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        positions
            .into_iter()
            .map(|pos| {
                let probabilities = self.eval(&pos);
                (pos, probabilities)
            })
            .collect()
    }

    /// Returns the position after applying the *best* move to `pos`.
    /// The returned `Position` has already switches sides.
    /// This means the returned position will have the *lowest* equity of possible positions.
    #[inline]
    fn best_position_by_equity(&self, pos: &Position, dice: &Dice) -> Position {
        self.best_position(pos, dice, |probabilities| probabilities.equity())
    }

    /// Returns the position after applying the *best* move to `pos`.
    /// The returned `Position` has already switches sides.
    /// This means the returned position will have the *lowest* value of possible positions.
    fn best_position<F>(&self, pos: &Position, dice: &Dice, value: F) -> Position
    where
        F: Fn(&Probabilities) -> f32,
    {
        let mut positions = pos.all_positions_after_moving(dice);

        // Two optimizations so that we don't have to call eval_batch that often.
        // The function would also work without the next 6 lines.
        if positions.len() == 1 {
            return positions.pop().unwrap();
        }
        if let Some(end_of_game) = positions.iter().position(|p| p.has_lost()) {
            return positions.swap_remove(end_of_game);
        }
        self.eval_batch(positions)
            .into_iter()
            .map(|(position, probabilities)| (position, value(&probabilities)))
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
        let mut pos_and_probs: Vec<(Position, Probabilities)> = self
            .eval_batch(after_moving)
            .into_iter()
            .map(|(pos, probabilities)| (pos.sides_switched(), probabilities.switch_sides()))
            .collect();
        pos_and_probs.sort_unstable_by(|a, b| b.1.equity().partial_cmp(&a.1.equity()).unwrap());
        pos_and_probs
    }
}

/// [BatchEvaluator] is a subtrait of [Evaluator]. The function [Evaluator::eval_batch] is
/// implemented by default. This trait is meant for evaluating all legal moves at once.
/// An example is [crate::onnx::OnnxEvaluator], where feeding several positions at once to the
/// neural net is more performant than evaluating them one by one.
///
/// In the future, also multiply evaluators could implement this trait, deciding which positions
/// are worth to look more into at deeper plies, while others don't have to be considered further.
pub trait BatchEvaluator: Evaluator {
    /// Evaluate all legal moves following a certain position/dice combination.
    fn eval_positions(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)>;
}

impl<T: BatchEvaluator> Evaluator for T {
    #[inline]
    fn eval(&'_ self, pos: &Position) -> Probabilities {
        BatchEvaluator::eval_positions(self, vec![*pos])
            .pop()
            .unwrap()
            .1
    }

    #[inline]
    fn eval_batch(&self, positions: Vec<Position>) -> Vec<(Position, Probabilities)> {
        BatchEvaluator::eval_positions(self, positions)
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

    fn position_with_lowest_equity() -> Position {
        pos!(x 5:1, 3:1; o 20:2).sides_switched()
    }

    /// Test double. Returns not so good probabilities for `expected_pos`, better for everything else.
    struct EvaluatorFake {}
    impl Evaluator for EvaluatorFake {
        fn eval(&self, pos: &Position) -> Probabilities {
            if pos == &position_with_lowest_equity() {
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
                    win_normal: 0.38,
                    win_gammon: 0.2,
                    win_bg: 0.1,
                    lose_normal: 0.12,
                    lose_gammon: 0.1,
                    lose_bg: 0.1,
                }
            }
        }
    }

    #[test]
    fn best_position_by_equity() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = EvaluatorFake {};
        // When
        let best_pos = evaluator.best_position_by_equity(&given_pos, &Dice::new(4, 2));
        // Then
        assert_eq!(best_pos, position_with_lowest_equity());
    }

    #[test]
    /// This is basically the same test as the one above (best_position_by_equity), but with different outcome for 1 ptrs.
    fn best_position_for_1ptr() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = EvaluatorFake {};
        // When
        let best_pos = evaluator.best_position(&given_pos, &Dice::new(4, 2), |p| p.win());
        // Then
        let expected = pos!(x 7:1, 1:1; o 20: 2);
        assert_eq!(best_pos, expected.sides_switched());
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
        let best_pos = best_pos.sides_switched();
        assert_eq!(
            &best_pos,
            &evaluator.best_position_by_equity(&given_pos, &Dice::new(4, 2))
        );
        assert_eq!(best_probability.switch_sides(), evaluator.eval(&best_pos));
    }

    #[test]
    fn eval_batch_empty() {
        // Given
        let evaluator = EvaluatorFake {};
        // When
        let values = evaluator.eval_batch(vec![]);
        // Then
        assert!(values.is_empty());
    }

    #[test]
    fn eval_batch() {
        // Given
        let pos_1 = pos!(x 7:2; o 20:2);
        let pos_2 = position_with_lowest_equity();
        let evaluator = EvaluatorFake {};
        // When
        let values = evaluator.eval_batch(vec![pos_1, pos_2]);
        // Then
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].0, pos_1);
        assert_eq!(values[1].0, pos_2);
        assert_eq!(values[0].1, evaluator.eval(&pos_1));
        assert_eq!(values[1].1, evaluator.eval(&pos_2));
        assert_ne!(evaluator.eval(&pos_1), evaluator.eval(&pos_2));
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
