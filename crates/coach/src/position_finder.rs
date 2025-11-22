mod finder_rand;

use crate::position_finder::finder_rand::{FinderRand, FinderRandomizer};
use engine::dice::Dice;
use engine::dice_gen::{DiceGen, FastrandDice};
use engine::evaluator::Evaluator;
use engine::position::GameState::Ongoing;
use engine::position::OngoingPhase::Race;
use engine::position::{GamePhase, OngoingPhase, Position, STARTING};
use engine::probabilities::Probabilities;
use indexmap::IndexSet;

/// Finds random positions for later rollout.
pub trait PositionFinder {
    fn find_positions(&mut self, number: usize, phase: OngoingPhase) -> IndexSet<Position>;
}

pub fn position_finder_with_evaluator<'a, T: Evaluator + 'a>(
    evaluator: T,
) -> Box<dyn PositionFinder + 'a> {
    Box::new(ConcreteFinder {
        evaluator,
        dice_gen: FastrandDice::with_seed(0),
        rand: FinderRand::with_seed(0),
    })
}

struct ConcreteFinder<T: Evaluator, U: DiceGen, V: FinderRandomizer> {
    evaluator: T,
    dice_gen: U,
    rand: V,
}

impl<T: Evaluator, U: DiceGen, V: FinderRandomizer> PositionFinder for ConcreteFinder<T, U, V> {
    fn find_positions(&mut self, number: usize, phase: OngoingPhase) -> IndexSet<Position> {
        let phase = GamePhase::Ongoing(phase);
        let mut found: IndexSet<Position> = IndexSet::with_capacity(number);
        while found.len() < number {
            for position in self.positions_in_one_random_game() {
                if found.len() < number && position.game_phase() == phase {
                    found.insert(position);
                }
            }
        }
        found
    }
}

impl<T: Evaluator, U: DiceGen, V: FinderRandomizer> ConcreteFinder<T, U, V> {
    fn positions_in_one_random_game(&mut self) -> Vec<Position> {
        let mut positions: Vec<Position> = Vec::new();
        let mut pos = STARTING;
        let mut dice = self.dice_gen.roll_mixed();
        loop {
            let (next, rollout_positions) = self.next_and_found(pos, dice);
            if next.game_state() == Ongoing {
                positions.extend(rollout_positions);
                pos = next;
                dice = self.dice_gen.roll();
            } else {
                return positions;
            }
        }
    }

    /// Returns the next position and a vector with up to four positions for a rollout.
    ///
    /// The vector potentially contains:
    /// 1. The position at the top of the array
    /// 2. `next`, the position to which the `PositionFinder` is about to move.
    /// 3. Added is then a position from the middle of the input array, so that we also rollout
    ///    positions that are not so good.
    /// 4. If `all` contains both contact and race positions,
    ///    we make sure that at least one position from either phase is returned.
    ///
    /// Some of those positions could appear more than once in the array, but that's ok, we enter
    /// all of them into a set later on.
    ///
    /// The input values need to be from the perspective of the player who is about to move.
    /// The return values have switched sides, so they are in the proper format for a rollout.
    fn next_and_found(&mut self, position: Position, dice: Dice) -> (Position, Vec<Position>) {
        let pos_and_probs = self
            .evaluator
            .positions_and_probabilities_by_equity(&position, &dice);
        let next = self.next_position(&pos_and_probs).sides_switched();

        let positions: Vec<Position> = if next.game_state() != Ongoing {
            vec![]
        } else {
            let mut positions = Vec::with_capacity(4);

            // Best position:
            positions.push(pos_and_probs[0].0.sides_switched());
            // Next position:
            positions.push(next);
            // Mediocre position:
            if pos_and_probs.len() > 1 {
                let middle = positions.len() / 2;
                positions.push(pos_and_probs[middle].0.sides_switched());
            }
            // Best position with different game phase:
            if let Some(different_phase) = pos_and_probs.iter().position(|(pos, _)| {
                pos.game_state() == Ongoing && pos.game_phase() != next.game_phase()
            }) {
                positions.push(pos_and_probs[different_phase].0.sides_switched());
            }

            positions
        };

        (next, positions)
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
            GamePhase::GameOver(_) => return best.0,
            GamePhase::Ongoing(ongoing_state) => {
                if ongoing_state == Race {
                    // For races we always want to take the best move - it's not worth it to explore
                    // different moves here; for example we can't reach a backgame from a race.
                    // Instead we would only roll out strange positions later on.
                    return best.0;
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
        let choice = self.rand.sample(&chances);
        positions_and_probabilities
            .get(choice)
            .expect("choose_index must return index smaller than the number of moves")
            .0
    }
}

#[cfg(test)]
mod private_tests {
    use crate::position_finder::finder_rand::FinderRandomizer;
    use crate::position_finder::{ConcreteFinder, position_finder_with_evaluator};
    use engine::composite::CompositeEvaluator;
    use engine::dice_gen::FastrandDice;
    use engine::evaluator::RandomEvaluator;
    use engine::pos;
    use engine::position::OngoingPhase;
    use engine::probabilities::{Probabilities, ResultCounter};

    struct RandMock {}

    impl FinderRandomizer for RandMock {
        /// A bit quick and dirty - the values and asserts are hard coded for two tests at once: "contact()" and "race()"
        fn sample(&mut self, chances: &[f32]) -> usize {
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

        let mut finder = ConcreteFinder {
            evaluator: RandomEvaluator {},
            dice_gen: FastrandDice::with_seed(0),
            rand: RandMock {},
        };

        // Given
        let input = vec![(pos_1, prob_1), (pos_2, prob_2), (pos_3, prob_3)];
        // When
        let found = finder.next_position(&input);
        // Then
        assert_eq!(
            found, pos_2,
            "Second best move should be returned as specified in RandMock."
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

        let mut finder = ConcreteFinder {
            evaluator: RandomEvaluator {},
            dice_gen: FastrandDice::with_seed(0),
            rand: RandMock {},
        };

        // Given
        let input = vec![(pos_1, prob_1), (pos_2, prob_2), (pos_3, prob_3)];
        // When
        let found = finder.next_position(&input);
        // Then
        assert_eq!(found, pos_1, "Best move should be returned");
    }

    #[test]
    // We could look at each position from two sides. Make sure it's the correct one.
    fn direction_of_positions_is_correct() {
        // Given
        let mut finder = position_finder_with_evaluator(CompositeEvaluator::default_tests());
        // When
        let found_position = finder
            .find_positions(1, OngoingPhase::Contact)
            .first()
            .unwrap()
            .to_owned();
        // Then
        // x is still in the starting position, o has some moved pieces.
        let expected = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:4, 12:4, 1:2);
        assert_eq!(found_position, expected);
    }
}
