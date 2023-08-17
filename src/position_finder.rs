use crate::dice_gen::{DiceGen, FastrandDice};
use crate::evaluator::Evaluator;
use crate::position::GameState::Ongoing;
use crate::position::{Position, STARTING};
use std::collections::HashSet;

/// Finds random positions for later rollout.
pub struct PositionFinder<T: Evaluator> {
    evaluator: T,
    dice_gen: FastrandDice,
}

impl<T: Evaluator> PositionFinder<T> {
    /// Contains different random number generators every time it's called.
    #[allow(clippy::new_without_default)]
    pub fn new(evaluator: T) -> Self {
        PositionFinder {
            evaluator,
            dice_gen: FastrandDice::new(),
        }
    }

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
        while pos.game_state() == Ongoing {
            // Todo: Don't allow doubles in first move
            let (die1, die2) = self.dice_gen.roll();
            let new_positions = pos.all_positions_after_moving(die1, die2);
            // Todo: remove cloning by implementing the Copy trait -> maybe better performance
            pos = self.evaluator.worst_position(&new_positions).clone();
            let mut ongoing_games: Vec<Position> = new_positions
                .into_iter()
                .filter(|p| p.game_state() == Ongoing)
                .collect();
            positions.append(&mut ongoing_games);
        }
        positions
    }
}
