mod discrepancy_selector;
mod diverse_selector;
mod finder_rand;

use crate::position_finder::discrepancy_selector::{DiscrepancySelector, MultiPlyDiscrepancy};
use crate::position_finder::diverse_selector::DiverseSelector;
use crate::position_finder::finder_rand::{FinderRand, FinderRandomizer};
use engine::dice::Dice;
use engine::dice_gen::{DiceGen, FastrandDice};
use engine::evaluator::Evaluator;
use engine::multiply::MultiPlyEvaluator;
use engine::position::GameState::Ongoing;
use engine::position::{GamePhase, OngoingPhase, Position, STARTING};
use indexmap::IndexSet;

/// Finds random positions for later rollout.
pub trait PositionFinder {
    fn find_positions(&mut self, number: usize, phase: OngoingPhase) -> IndexSet<Position>;
}

pub fn diverse_with_evaluator<'a, T: Evaluator + 'a>(evaluator: T) -> Box<dyn PositionFinder + 'a> {
    let move_selector = DiverseSelector {
        evaluator,
        rand: FinderRand::with_seed(0),
    };
    let dice_gen = FastrandDice::with_seed(0);

    Box::new(ConcreteFinder {
        move_selector,
        dice_gen,
    })
}

pub fn discrepancy_with_evaluator<'a, T: Evaluator + 'a>(
    evaluator: T,
    threshold: f32,
) -> Box<dyn PositionFinder + 'a> {
    let multiply = MultiPlyEvaluator { evaluator };
    let discrepancy = MultiPlyDiscrepancy { multiply };
    let move_selector = DiscrepancySelector {
        evaluators: discrepancy,
        threshold,
    };
    let dice_gen = FastrandDice::with_seed(0);

    Box::new(ConcreteFinder {
        move_selector,
        dice_gen,
    })
}

trait MoveSelector {
    fn next_and_found(
        &mut self,
        position: Position,
        dice: Dice,
        phase: OngoingPhase,
    ) -> (Position, Vec<Position>);
}

struct ConcreteFinder<T: MoveSelector, U: DiceGen> {
    move_selector: T,
    dice_gen: U,
}

impl<T: MoveSelector, U: DiceGen> PositionFinder for ConcreteFinder<T, U> {
    fn find_positions(&mut self, number: usize, phase: OngoingPhase) -> IndexSet<Position> {
        let mut found: IndexSet<Position> = IndexSet::with_capacity(number);
        while found.len() < number {
            for position in self.positions_in_one_random_game(phase) {
                if found.len() < number {
                    assert_eq!(position.game_phase(), GamePhase::Ongoing(phase));
                    found.insert(position);
                }
            }
        }
        found
    }
}

impl<T: MoveSelector, U: DiceGen> ConcreteFinder<T, U> {
    fn positions_in_one_random_game(&mut self, phase: OngoingPhase) -> Vec<Position> {
        let mut positions: Vec<Position> = Vec::new();
        let mut pos = STARTING;
        let mut dice = self.dice_gen.roll_mixed();
        loop {
            let (next, rollout_positions) = self.move_selector.next_and_found(pos, dice, phase);
            if next.game_state() == Ongoing {
                positions.extend(rollout_positions);
                pos = next;
                dice = self.dice_gen.roll();
            } else {
                return positions;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::position_finder::diverse_with_evaluator;
    use engine::composite::CompositeEvaluator;
    use engine::pos;
    use engine::position::OngoingPhase;

    #[test]
    // We could look at each position from two sides. Make sure it's the correct one.
    fn direction_of_positions_is_correct() {
        // Given
        let mut finder = diverse_with_evaluator(CompositeEvaluator::default_tests());
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
