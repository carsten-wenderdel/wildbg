mod finder_rand;

use crate::position_finder::finder_rand::{FinderRand, FinderRandomizer};
use engine::dice::Dice;
use engine::dice_gen::{DiceGen, FastrandDice};
use engine::evaluator::Evaluator;
use engine::position::GameState::Ongoing;
use engine::position::{GamePhase, OngoingPhase, Position, STARTING};
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
        let next = self.rand.next_position(&pos_and_probs).sides_switched();

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
}

#[cfg(test)]
mod tests {
    use crate::position_finder::position_finder_with_evaluator;
    use engine::composite::CompositeEvaluator;
    use engine::pos;
    use engine::position::OngoingPhase;

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
