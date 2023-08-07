use crate::position::Position;

#[allow(dead_code)]
/// Sum of all six fields will always be 1.0
pub(crate) struct Probabilities {
    win_normal: f32,
    win_gammon: f32,
    win_bg: f32,
    lose_normal: f32,
    lose_gammon: f32,
    lose_bg: f32,
}

impl Probabilities {
    /// Cubeless equity
    #[allow(dead_code)]
    fn equity(&self) -> f32 {
        self.win_gammon - self.lose_normal
            + 2.0 * (self.win_gammon - self.lose_gammon)
            + 3.0 * (self.win_bg - self.lose_bg)
    }
}

pub(crate) trait Evaluator {
    /// Returns a cubeless evaluation of a position.
    /// Implementing types will calculate the probabilities with different strategies.
    /// Examples of such strategies are a rollout or 1-ply inference of a neural net.
    fn eval(&self, pos: &Position) -> Probabilities;

    /// Returns the position after applying the *best* move to `pos`.
    /// The returned `Position` has already switches sides.
    /// This means the returned position will have the *lowest* equity of possible positions.
    fn best_position(&self, pos: &Position, die1: usize, die2: usize) -> Position {
        pos.all_positions_after_moving(die1, die2)
            .iter()
            .map(|pos| (pos, self.eval(pos).equity()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
            .clone()
    }
}

pub(crate) struct RandomEvaluator {}

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
        let best_pos = evaluator.best_position(&given_pos, 4, 2);
        // Then
        assert_eq!(best_pos, expected_pos());
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
