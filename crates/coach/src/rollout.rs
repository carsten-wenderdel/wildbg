use engine::dice::{Dice, ALL_441};
use engine::dice_gen::{DiceGen, FastrandDice};
use engine::evaluator::{Evaluator, RandomEvaluator};
use engine::position::GameState::{GameOver, Ongoing};
use engine::position::{GameResult, Position};
use engine::probabilities::{Probabilities, ResultCounter};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Two `RolloutEvaluator`s which are initialized with the same `seed` and the same evaluators,
/// will always return the identical value when `eval` is called for the same position.
pub struct RolloutEvaluator<T: Evaluator> {
    evaluator: T,
    seed: u64,
}

/// We will do 1296 single rollouts and we need different dice for them.
/// Each of those 1296 rollouts will have a separate dice generator, here are the seeds to initialize them.
fn dice_seeds(dice_gen: &mut FastrandDice, amount: usize) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(amount);
    for _ in 0..amount {
        seeds.push(dice_gen.seed());
    }
    seeds
}

impl<T: Evaluator + Sync> Evaluator for RolloutEvaluator<T> {
    /// Rolls out 1296 times, the dice for the first two half moves are given, rest is random
    fn eval(&self, pos: &Position) -> Probabilities {
        debug_assert!(pos.game_state() == Ongoing);

        // We don't want to have identical dice for rollouts of *all* positions.
        // On the other hand, for a certain position, we always want the same dice, this helps in tests.
        // So we initialize `FastrandDice` with a seed depending on the hash of the position combined
        // with the seed of this RolloutEvaluator.
        let mut hasher = DefaultHasher::new();
        pos.hash(&mut hasher);
        self.seed.hash(&mut hasher);
        let seed = hasher.finish();
        let mut dice_gen = FastrandDice::with_seed(seed);

        let dice_and_seeds =
            ALL_441.map(|(dice, amount)| (dice, dice_seeds(&mut dice_gen, amount)));
        let game_results: ResultCounter = dice_and_seeds
            .par_iter()
            .map(|(dice, seeds)| self.results_from_single_rollouts(pos, dice, seeds))
            .reduce(ResultCounter::default, |a, b| a.combine(&b));

        debug_assert_eq!(
            game_results.sum(),
            6 * 6 * 6 * 6,
            "Rollout should look at 1296 games"
        );
        Probabilities::from(&game_results)
    }
}

impl RolloutEvaluator<RandomEvaluator> {
    pub fn with_random_evaluator() -> Self {
        Self::with_evaluator(RandomEvaluator {})
    }
}

impl<T: Evaluator> RolloutEvaluator<T> {
    pub fn with_evaluator(evaluator: T) -> Self {
        let seed = FastrandDice::random_seed();
        Self::with_evaluator_and_seed(evaluator, seed)
    }

    pub fn with_evaluator_and_seed(evaluator: T, seed: u64) -> Self {
        Self { evaluator, seed }
    }

    /// Will do *n* rollouts from the given position, with *n* being the length of `seeds`.
    ///
    /// It will initially use `first_dice` for all these rollouts. If the game hasn't ended then,
    /// this will be followed by random dice. The dice generators for that are initialized with `seeds`.
    fn results_from_single_rollouts(
        &self,
        from: &Position,
        first_dice: &[Dice; 2],
        seeds: &[u64],
    ) -> ResultCounter {
        let mut counter = ResultCounter::default();
        match self.single_rollout_with_dice(from, first_dice) {
            Ok(result) => {
                counter.add_results(result, seeds.len() as u32);
            }
            Err(pos) => seeds.iter().for_each(|seed| {
                let mut dice_gen = FastrandDice::with_seed(*seed);
                let result = self.single_rollout_with_generator(&pos, &mut dice_gen);
                counter.add(result);
            }),
        }
        counter
    }

    /// Will try to do a rollout with the given `first_dice`.
    ///
    /// If the game ends after using `first_dice` it will return the game result as `success`.
    /// If the game has not ended yet, it will return the then reached position as `failure`.
    fn single_rollout_with_dice(
        &self,
        from: &Position,
        first_dice: &[Dice; 2],
    ) -> Result<GameResult, Position> {
        let mut player_on_turn = true;
        let mut pos = *from;
        for dice in first_dice {
            pos = self.evaluator.best_position_by_equity(&pos, dice);
            if let GameOver(result) = pos.game_state() {
                return if player_on_turn {
                    Ok(result.reverse())
                } else {
                    Ok(result)
                };
            }
            player_on_turn = !player_on_turn;
        }
        Err(pos)
    }

    fn single_rollout_with_generator<U: DiceGen>(
        &self,
        from: &Position,
        dice_gen: &mut U,
    ) -> GameResult {
        let mut player_on_turn = true;
        let mut pos = *from;
        loop {
            let dice = dice_gen.roll();
            pos = self.evaluator.best_position_by_equity(&pos, &dice);
            if let GameOver(result) = pos.game_state() {
                return if player_on_turn {
                    result.reverse()
                } else {
                    result
                };
            }
            player_on_turn = !player_on_turn;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rollout::RolloutEvaluator;
    use engine::evaluator::Evaluator;
    use engine::pos;

    #[test]
    fn correct_results_after_first_or_second_half_move() {
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
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
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 17:15; o 24:8);

        let results = rollout_eval.eval(&pos);
        assert_eq!(results.lose_gammon, 1.0);
    }
    #[test]
    fn rollout_always_win_bg() {
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 1:8; o 2:15);

        let results = rollout_eval.eval(&pos);
        assert_eq!(results.win_bg, 1.0);
    }
}

#[cfg(test)]
mod private_tests {
    use crate::rollout::RolloutEvaluator;
    use engine::dice::Dice;
    use engine::dice_gen::DiceGenMock;
    use engine::pos;
    use engine::position::GameResult::{
        LoseBg, LoseGammon, LoseNormal, WinBg, WinGammon, WinNormal,
    };

    #[test]
    fn single_rollout_with_generator_win_normal() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 12:1; o 13:1);
        // When
        let mut dice_gen = DiceGenMock::new(&[Dice::new(2, 1), Dice::new(2, 1), Dice::new(4, 5)]);
        let result = rollout_eval.single_rollout_with_generator(&pos, &mut dice_gen);
        //Then
        dice_gen.assert_all_dice_were_used();
        assert_eq!(result, WinNormal);
    }

    #[test]
    fn single_rollout_with_generator_lose_normal() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 12:1; o 13:1);
        // When
        let mut dice_gen = DiceGenMock::new(&[
            Dice::new(2, 1),
            Dice::new(2, 1),
            Dice::new(1, 2),
            Dice::new(4, 5),
        ]);
        let result = rollout_eval.single_rollout_with_generator(&pos, &mut dice_gen);
        // Then
        dice_gen.assert_all_dice_were_used();
        assert_eq!(result, LoseNormal);
    }

    #[test]
    fn single_rollout_with_dice_win_gammon() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 1:4; o 18:15);
        // When
        let result =
            rollout_eval.single_rollout_with_dice(&pos, &[Dice::new(2, 2), Dice::new(6, 6)]);
        //Then
        assert_eq!(result, Ok(WinGammon));
    }

    #[test]
    fn single_rollout_with_dice_lose_gammon() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 12:15; o 24:1);
        // When
        let result =
            rollout_eval.single_rollout_with_dice(&pos, &[Dice::new(2, 1), Dice::new(3, 3)]);
        //Then
        assert_eq!(result, Ok(LoseGammon));
    }

    #[test]
    fn single_rollout_with_dice_win_bg() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 24:1; o 1:15);
        // When
        let result =
            rollout_eval.single_rollout_with_dice(&pos, &[Dice::new(6, 6), Dice::new(1, 2)]);
        //Then
        assert_eq!(result, Ok(WinBg));
    }

    #[test]
    fn single_rollout_with_dice_lose_bg() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 24:15; o 1:1);
        // When
        let result =
            rollout_eval.single_rollout_with_dice(&pos, &[Dice::new(1, 2), Dice::new(6, 6)]);
        //Then
        assert_eq!(result, Ok(LoseBg));
    }

    #[test]
    fn single_rollout_with_dice_return_failure() {
        // Given
        let rollout_eval = RolloutEvaluator::with_random_evaluator();
        let pos = pos!(x 1:15; o 24:5);
        // When
        let result =
            rollout_eval.single_rollout_with_dice(&pos, &[Dice::new(1, 2), Dice::new(6, 1)]);
        //Then
        let expected_position = pos!(x 1:13; o 24:3);
        assert_eq!(result, Err(expected_position));
    }
}
