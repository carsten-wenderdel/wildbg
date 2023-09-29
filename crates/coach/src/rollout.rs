use engine::dice::{Dice, DiceGen, FastrandDice, ALL_1296};
use engine::evaluator::{Evaluator, RandomEvaluator};
use engine::position::GameState::{GameOver, Ongoing};
use engine::position::{GameResult, Position};
use engine::probabilities::Probabilities;
use rayon::prelude::*;

pub struct RolloutEvaluator<T: Evaluator> {
    evaluator: T,
}

impl<T: Evaluator + Sync> Evaluator for RolloutEvaluator<T> {
    /// Rolls out 1296 times, first two half moves are given, rest is random
    fn eval(&self, pos: &Position) -> Probabilities {
        debug_assert!(pos.game_state() == Ongoing);
        let dice = ALL_1296;

        let game_results: Vec<GameResult> = dice
            .par_iter()
            .map(|dice_pair| {
                let mut dice_gen = FastrandDice::new();
                self.single_rollout(pos, &[dice_pair.0, dice_pair.1], &mut dice_gen)
            })
            .collect();

        let mut results = [0; 6];
        for gr in game_results {
            results[gr as usize] += 1;
        }
        debug_assert_eq!(
            results.iter().sum::<u32>(),
            6 * 6 * 6 * 6,
            "Rollout should look at 1296 games"
        );
        Probabilities::new(&results)
    }
}

impl RolloutEvaluator<RandomEvaluator> {
    pub fn new_random() -> Self {
        RolloutEvaluator {
            evaluator: RandomEvaluator {},
        }
    }
}

impl<T: Evaluator> RolloutEvaluator<T> {
    pub fn with_evaluator(evaluator: T) -> Self {
        Self { evaluator }
    }

    /// `first_dice` contains the dice for first moves, starting at index 0. It may be empty.
    /// Once all of those given dice have been used, subsequent dice are generated from `dice_gen`.
    #[allow(dead_code)]
    fn single_rollout<U: DiceGen>(
        &self,
        from: &Position,
        first_dice: &[Dice],
        dice_gen: &mut U,
    ) -> GameResult {
        let mut iteration = 0;
        let mut pos = from.clone();
        loop {
            let dice = if first_dice.len() > iteration {
                first_dice[iteration]
            } else {
                dice_gen.roll()
            };
            pos = self.evaluator.best_position(&pos, &dice);
            match pos.game_state() {
                Ongoing => {
                    iteration += 1;
                    continue;
                }
                GameOver(result) => {
                    return if iteration % 2 == 0 {
                        result.reverse()
                    } else {
                        result
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rollout::RolloutEvaluator;
    use engine::evaluator::{Evaluator, RandomEvaluator};
    use engine::pos;
    use engine::position::Position;
    use std::collections::HashMap;

    #[test]
    fn correct_results_after_first_or_second_half_move() {
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 6:1; o 19:1);

        // From this position both players are only 6 pips (2 moves) away from finishing.
        // During a rollout each first move of each player is predetermined. If this first move
        // doesn't lead to finishing, any random second move will end the game.
        // Because of this we can calculate the results of the 1296 games during a rollout.

        // Player `x` will have the first move. Out of the 36 dice possibilities, everything will
        // end the game, with the exception of 12, 21, 13, 31, 15, 51, 23, 32 and 11.
        // So in 9 of 36 cases the game continues, in 27 of 36 cases `x` wins immediately.
        // 27 of 36 means 972 of the total of all 1296 games will `x` win immediately.

        // In the remaining 324 of 129 games player `o` wins 27 of 36 games immediately.
        // This means `o` will win 243 games. The remaining 81 games will then win `x` with the
        // second move.

        // Son in total we expect `x` to win 972 + 81 = 1053 games and `o` to win 243 games.
        // In percentage: `x` will win normal 81.25% and lose normal 18.75%.

        let results = rollout_eval.eval(&pos);
        assert_eq!(results.win_normal, 0.8125);
        assert_eq!(results.lose_normal, 0.1875);
    }

    #[test]
    fn rollout_always_lose_gammon() {
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 17:15; o 24:8);

        let results = rollout_eval.eval(&pos);
        assert_eq!(results.lose_gammon, 1.0);
    }
    #[test]
    fn rollout_always_win_bg() {
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 1:8; o 2:15);

        let results = rollout_eval.eval(&pos);
        assert_eq!(results.win_bg, 1.0);
    }
}

#[cfg(test)]
mod private_tests {
    use crate::rollout::RolloutEvaluator;
    use engine::dice::{Dice, DiceGenMock, FastrandDice};
    use engine::evaluator::RandomEvaluator;
    use engine::pos;
    use engine::position::GameResult::{
        LoseBg, LoseGammon, LoseNormal, WinBg, WinGammon, WinNormal,
    };
    use engine::position::Position;
    use std::collections::HashMap;

    #[test]
    fn single_rollout_win_normal() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 12:1; o 13:1);
        // When
        let mut dice_gen = DiceGenMock::new(&[Dice::new(2, 1), Dice::new(2, 1)]);
        let result = rollout_eval.single_rollout(&pos, &[Dice::new(4, 5)], &mut dice_gen);
        //Then
        dice_gen.assert_all_dice_were_used();
        assert_eq!(result, WinNormal);
    }

    #[test]
    fn single_rollout_lose_normal() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 12:1; o 13:1);
        // When
        let mut dice_gen = DiceGenMock::new(&[Dice::new(2, 1), Dice::new(2, 1)]);
        let result =
            rollout_eval.single_rollout(&pos, &[Dice::new(1, 2), Dice::new(4, 5)], &mut dice_gen);
        // Then
        dice_gen.assert_all_dice_were_used();
        assert_eq!(result, LoseNormal);
    }

    #[test]
    fn single_rollout_win_gammon() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 1:4; o 12:15);
        // When
        let result =
            rollout_eval.single_rollout(&pos, &[Dice::new(2, 2)], &mut FastrandDice::new());
        //Then
        assert_eq!(result, WinGammon);
    }

    #[test]
    fn single_rollout_lose_gammon() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 12:15; o 24:1);
        // When
        let result = rollout_eval.single_rollout(
            &pos,
            &[Dice::new(2, 1), Dice::new(3, 3)],
            &mut FastrandDice::new(),
        );
        //Then
        assert_eq!(result, LoseGammon);
    }

    #[test]
    fn single_rollout_win_bg() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 24:1; o 1:15);
        // When
        let result =
            rollout_eval.single_rollout(&pos, &[Dice::new(6, 6)], &mut FastrandDice::new());
        //Then
        assert_eq!(result, WinBg);
    }

    #[test]
    fn single_rollout_lose_bg() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 24:15; o 1:1);
        // When
        let result = rollout_eval.single_rollout(
            &pos,
            &[Dice::new(1, 2), Dice::new(6, 6)],
            &mut FastrandDice::new(),
        );
        //Then
        assert_eq!(result, LoseBg);
    }
}
