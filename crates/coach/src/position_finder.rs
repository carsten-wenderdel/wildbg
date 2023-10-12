use engine::dice::{DiceGen, FastrandDice};
use engine::evaluator::Evaluator;
use engine::position::GameState::Ongoing;
use engine::position::{Position, STARTING};
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
        while pos.game_state() == Ongoing {
            let new_positions = pos.all_positions_after_moving(&dice);
            // Todo: remove cloning by implementing the Copy trait -> maybe better performance
            pos = self
                .evaluator
                .worst_position(&new_positions, |probabilities| probabilities.equity())
                .clone();
            let mut ongoing_games: Vec<Position> = new_positions
                .into_iter()
                .filter(|p| p.game_state() == Ongoing)
                .collect();
            positions.append(&mut ongoing_games);
            dice = self.dice_gen.roll();
        }
        positions
    }
}
