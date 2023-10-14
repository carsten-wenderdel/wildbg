use engine::dice::{DiceGen, FastrandDice};
use engine::evaluator::Evaluator;
use engine::position::GameState::Ongoing;
use engine::position::OngoingState::Race;
use engine::position::{GamePhase, Position, STARTING};
use engine::probabilities::Probabilities;
use std::collections::HashSet;

/// Finds random positions for later rollout.
pub struct PositionFinder<T: Evaluator, U: DiceGen> {
    evaluator: T,
    dice_gen: U,
}

impl<T: Evaluator> PositionFinder<T, FastrandDice> {
    /// Contains different random number generator every time it's called.
    pub fn with_random_dice(evaluator: T) -> Self {
        PositionFinder {
            evaluator,
            dice_gen: FastrandDice::new(),
        }
    }
}

impl<T: Evaluator, U: DiceGen> PositionFinder<T, U> {
    pub fn find_positions(&mut self, amount: usize) -> HashSet<Position> {
        let mut found: HashSet<Position> = HashSet::new();
        while found.len() < amount {
            let mut more = self.positions_in_one_random_game();
            while found.len() < amount {
                match more.pop() {
                    Some(pos) => found.insert(pos),
                    None => break,
                };
            }
        }
        found
    }

    fn positions_in_one_random_game(&mut self) -> Vec<Position> {
        let mut positions: Vec<Position> = Vec::new();
        let mut pos = STARTING;
        let mut dice = self.dice_gen.roll_regular();
        loop {
            let positions_and_probabilities = self
                .evaluator
                .positions_and_probabilities_by_equity(&pos, &dice);
            pos = self
                .next_position(&positions_and_probabilities)
                .switch_sides();
            if pos.game_state() == Ongoing {
                positions.push(pos.clone());
                dice = self.dice_gen.roll();
            } else {
                return positions;
            }
        }
    }

    fn next_position(
        &mut self,
        positions_and_probabilities: &[(Position, Probabilities)],
    ) -> Position {
        let best = positions_and_probabilities
            .first()
            .expect("move generator must always return a move");
        // In some cases we definitely want to return the best move
        match best.0.game_phase() {
            GamePhase::GameOver(_) => return best.0.clone(),
            GamePhase::Ongoing(ongoing_state) => {
                if ongoing_state == Race {
                    // For races we always want to take the best move - it's not worth it to explore
                    // different moves here; for example we can't reach a backgame from a race.
                    // Instead we would only roll out strange positions later on.
                    return best.0.clone();
                }
            }
        }
        // Ok, now we are in contact game; sometimes we want return another move than the best one.
        let best_equity = best.1.equity();
        let chances: Vec<f32> = positions_and_probabilities
            .iter()
            .filter_map(|(_, probability)| {
                let equity_loss = best_equity - probability.equity();
                // Let's ignore really bad moves, but if the equity loss is small we sometimes
                // want to use them.
                if equity_loss < 0.05 {
                    // So for the best move the chance is 1.0.
                    // For a move with an equity loss of 0.03 the chance is 0.55
                    Some(1.0 - 15.0 * equity_loss)
                } else {
                    // `positions_and_probabilities` is sorted by equity.
                    // So we give a chance for the first entries in there and filter everything
                    // after that away.
                    None
                }
            })
            .collect();
        let choice = self.dice_gen.choose_index(&chances);
        positions_and_probabilities
            .get(choice)
            .expect("choose_index must return index smaller than the number of moves")
            .0
            .clone()
    }
}

#[cfg(test)]
mod private_tests {
    use crate::position_finder::PositionFinder;
    use engine::dice::{Dice, DiceGen};
    use engine::evaluator::RandomEvaluator;
    use engine::pos;
    use engine::position::Position;
    use engine::probabilities::{Probabilities, ResultCounter};
    use std::collections::HashMap;

    struct DiceGenChooseMock {}

    /// Mock to make sure that the `dice_gen` is called with the right values in the tests.
    impl DiceGen for DiceGenChooseMock {
        fn roll(&mut self) -> Dice {
            unreachable!()
        }

        /// A bit quick and dirty - the values and asserts are hard coded for two tests at once: "contact()" and "race()"
        fn choose_index(&mut self, chances: &[f32]) -> usize {
            // This method should only be called in the test "contact". In race, this method should
            // not be called as the best position should be returned.
            assert_eq!(chances.len(), 2);
            assert_eq!(chances[0], 1.0);
            assert!((chances[1] - 0.85).abs() < 0.000001);
            1
        }
    }

    #[test]
    fn contact() {
        let pos_1 = pos!(x 20:1; o 1:1);
        let pos_2 = pos!(x 20:1; o 2:1);
        let pos_3 = pos!(x 20:1; o 3:1);

        let prob_1 = Probabilities::from(&ResultCounter::new(80, 20, 0, 0, 0, 0));
        let prob_2 = Probabilities::from(&ResultCounter::new(81, 19, 0, 0, 0, 0));
        let prob_3 = Probabilities::from(&ResultCounter::new(86, 14, 0, 0, 0, 0));

        // Not part of the actual test, just to make sure that we test later is done properly
        assert!((prob_1.equity() - prob_2.equity() - 0.01).abs() < 0.0000001);
        assert!((prob_1.equity() - prob_3.equity() - 0.06).abs() < 0.0000001);

        let mut finder = PositionFinder {
            evaluator: RandomEvaluator {},
            dice_gen: DiceGenChooseMock {},
        };

        // Given
        let input = vec![(pos_1, prob_1), (pos_2.clone(), prob_2), (pos_3, prob_3)];
        // When
        let found = finder.next_position(&input);
        // Then
        assert_eq!(
            found, pos_2,
            "Second best move should be returned as specified in DiceGenChooseMock."
        );
    }

    #[test]
    fn race() {
        let pos_1 = pos!(x 1:1; o 10:1);
        let pos_2 = pos!(x 2:1; o 10:1);
        let pos_3 = pos!(x 3:1; o 10:1);

        let prob_1 = Probabilities::from(&ResultCounter::new(80, 20, 0, 0, 0, 0));
        let prob_2 = Probabilities::from(&ResultCounter::new(81, 19, 0, 0, 0, 0));
        let prob_3 = Probabilities::from(&ResultCounter::new(86, 14, 0, 0, 0, 0));

        // Not part of the actual test, just to make sure that we test later is done properly
        assert!((prob_1.equity() - prob_2.equity() - 0.01).abs() < 0.0000001);
        assert!((prob_1.equity() - prob_3.equity() - 0.06).abs() < 0.0000001);

        let mut finder = PositionFinder {
            evaluator: RandomEvaluator {},
            dice_gen: DiceGenChooseMock {},
        };

        // Given
        let input = vec![(pos_1.clone(), prob_1), (pos_2, prob_2), (pos_3, prob_3)];
        // When
        let found = finder.next_position(&input);
        // Then
        assert_eq!(found, pos_1, "Best move should be returned");
    }
}
