use crate::dice_gen::DiceGen;
use crate::evaluator::{Evaluator, Probabilities};
use crate::position::GameState::{GameOver, Ongoing};
use crate::position::{GameResult, Position};

struct RolloutEvaluator<T: Evaluator> {
    evaluator: T,
}

impl<T: Evaluator> Evaluator for RolloutEvaluator<T> {
    fn eval(&self, _pos: &Position) -> Probabilities {
        todo!()
    }
}

impl<T: Evaluator> RolloutEvaluator<T> {
    /// `first_dice` contains the dice for first moves, starting at index 0. It may be empty.
    /// Once all of those given dice have been used, subsequent dice are generated from `dice_gen`.
    #[allow(dead_code)]
    fn single_rollout<U: DiceGen>(
        &self,
        from: &Position,
        first_dice: &[(usize, usize)],
        dice_gen: &mut U,
    ) -> GameResult {
        let mut iteration = 0;
        let mut pos = from.clone();
        loop {
            let (die1, die2) = if first_dice.len() > iteration {
                first_dice[iteration]
            } else {
                dice_gen.roll()
            };
            pos = self.evaluator.best_position(&pos, die1, die2);
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
mod private_tests {
    use crate::dice_gen::{DiceGenMock, FastrandDice};
    use crate::evaluator::RandomEvaluator;
    use crate::pos;
    use crate::position::GameResult::{
        LoseBg, LoseGammon, LoseNormal, WinBg, WinGammon, WinNormal,
    };
    use crate::rollout::RolloutEvaluator;
    use crate::Position;
    use std::collections::HashMap;

    #[test]
    fn single_rollout_win_normal() {
        // Given
        let rollout_eval = RolloutEvaluator {
            evaluator: RandomEvaluator {},
        };
        let pos = pos!(x 12:1; o 13:1);
        // When
        let mut dice_gen = DiceGenMock::new(&[(2, 1), (2, 1)]);
        let result = rollout_eval.single_rollout(&pos, &[(4, 5)], &mut dice_gen);
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
        let mut dice_gen = DiceGenMock::new(&[(2, 1), (2, 1)]);
        let result = rollout_eval.single_rollout(&pos, &[(1, 2), (4, 5)], &mut dice_gen);
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
        let result = rollout_eval.single_rollout(&pos, &[(2, 2)], &mut FastrandDice::new());
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
        let result = rollout_eval.single_rollout(&pos, &[(2, 1), (3, 3)], &mut FastrandDice::new());
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
        let result = rollout_eval.single_rollout(&pos, &[(6, 6)], &mut FastrandDice::new());
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
        let result = rollout_eval.single_rollout(&pos, &[(1, 2), (6, 6)], &mut FastrandDice::new());
        //Then
        assert_eq!(result, LoseBg);
    }
}
