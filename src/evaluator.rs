use crate::position::GameResult::{LoseBg, LoseGammon, LoseNormal, WinBg, WinGammon, WinNormal};
use crate::position::Position;
use std::fmt;

/// Sum of all six fields will always be 1.0
#[derive(Debug)]
pub struct Probabilities {
    pub(crate) win_normal: f32,
    pub(crate) win_gammon: f32,
    pub(crate) win_bg: f32,
    pub(crate) lose_normal: f32,
    pub(crate) lose_gammon: f32,
    pub(crate) lose_bg: f32,
}

/// Used when writing CSV data to a file
impl fmt::Display for Probabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{};{};{};{};{};{}",
            self.win_normal,
            self.win_gammon,
            self.win_bg,
            self.lose_normal,
            self.lose_gammon,
            self.lose_bg
        )
    }
}

impl Probabilities {
    pub fn csv_header() -> String {
        "win_normal;win_gammon;win_bg;lose_normal;lose_gammon;lose_bg".to_string()
    }

    /// Typically used from rollouts.
    /// The index within the array has to correspond to the discriminant of the `Probabilities` enum.
    /// Input integer values will be normalized so that the sum in the return value is 1.0
    pub(crate) fn new(results: &[u32; 6]) -> Self {
        let sum = results.iter().sum::<u32>() as f32;
        Probabilities {
            win_normal: results[WinNormal as usize] as f32 / sum,
            win_gammon: results[WinGammon as usize] as f32 / sum,
            win_bg: results[WinBg as usize] as f32 / sum,
            lose_normal: results[LoseNormal as usize] as f32 / sum,
            lose_gammon: results[LoseGammon as usize] as f32 / sum,
            lose_bg: results[LoseBg as usize] as f32 / sum,
        }
    }

    /// Cubeless equity
    fn equity(&self) -> f32 {
        self.win_normal - self.lose_normal
            + 2.0 * (self.win_gammon - self.lose_gammon)
            + 3.0 * (self.win_bg - self.lose_bg)
    }
}

pub trait Evaluator {
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
mod probabilities_tests {
    use crate::evaluator::Probabilities;

    #[test]
    fn new() {
        // sum of `results is 32, a power of 2. Makes fractions easier to handle.
        let results = [0_u32, 1, 3, 4, 8, 16];
        let probabilities = Probabilities::new(&results);
        assert_eq!(probabilities.win_normal, 0.0);
        assert_eq!(probabilities.win_gammon, 0.03125);
        assert_eq!(probabilities.win_bg, 0.09375);
        assert_eq!(probabilities.lose_normal, 0.125);
        assert_eq!(probabilities.lose_gammon, 0.25);
        assert_eq!(probabilities.lose_bg, 0.5);
    }

    #[test]
    fn to_string() {
        let probabilities = Probabilities {
            win_normal: 1.0 / 21.0,
            win_gammon: 2.0 / 21.0,
            win_bg: 3.0 / 21.0,
            lose_normal: 4.0 / 21.0,
            lose_gammon: 5.0 / 21.0,
            lose_bg: 6.0 / 21.0,
        };
        assert_eq!(
            probabilities.to_string(),
            "0.04761905;0.0952381;0.14285715;0.1904762;0.23809524;0.2857143"
        );
    }

    #[test]
    fn equity_win_normal() {
        let probabilities = Probabilities {
            win_normal: 1.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 1.0);
    }

    #[test]
    fn equity_win_gammon() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 1.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 2.0);
    }

    #[test]
    fn equity_win_bg() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 1.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 3.0);
    }

    #[test]
    fn equity_lose_normal() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 1.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), -1.0);
    }

    #[test]
    fn equity_lose_gammon() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 1.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), -2.0);
    }

    #[test]
    fn equity_lose_bg() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 1.0,
        };
        assert_eq!(probabilities.equity(), -3.0);
    }

    #[test]
    fn equity_balanced() {
        let probabilities = Probabilities {
            win_normal: 0.3,
            win_gammon: 0.1,
            win_bg: 0.1,
            lose_normal: 0.3,
            lose_gammon: 0.1,
            lose_bg: 0.1,
        };
        assert_eq!(probabilities.equity(), 0.0);
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
