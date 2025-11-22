use crate::position_finder::{FinderRandomizer, MoveSelector};
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::position::GameState::Ongoing;
use engine::position::{GamePhase, OngoingPhase, Position};

pub(super) struct DiverseSelector<T: Evaluator, U: FinderRandomizer> {
    pub(super) evaluator: T,
    pub(super) rand: U,
}

impl<T: Evaluator, U: FinderRandomizer> MoveSelector for DiverseSelector<T, U> {
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
    fn next_and_found(
        &mut self,
        position: Position,
        dice: Dice,
        phase: OngoingPhase,
    ) -> (Position, Vec<Position>) {
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

            positions.retain(|p| p.game_phase() == GamePhase::Ongoing(phase));
            positions
        };

        (next, positions)
    }
}
