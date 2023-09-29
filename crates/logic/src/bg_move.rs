use engine::dice::Dice;
use engine::position::Position;
use serde::Serialize;
use std::cmp::max;
use utoipa::ToSchema;

mod double;
mod regular;

/// `BgMove` is not used during rollouts or evaluation but only when returning moves via an API
/// This is why a new `BgMove` is always calculated based on a `old` and a resulting `new` position.
pub struct BgMove {
    details: Vec<MoveDetail>,
}

#[derive(Debug, PartialEq, Serialize, ToSchema)]
/// Single movement of one checker. We always move from bigger pips to smaller pips.
/// If the same checker is moved more than once, multiple `MoveDetail`s are given.
/// Therefore: `from > to` and `from - to <= 6`.
pub struct MoveDetail {
    /// The bar is represented by `25`.
    #[schema(minimum = 1, maximum = 25)]
    from: usize,
    /// bear off is represented by `0`.
    #[schema(minimum = 0, maximum = 24)]
    to: usize,
}

impl BgMove {
    #[inline(always)]
    pub fn into_details(self) -> Vec<MoveDetail> {
        self.details
    }

    pub fn new(old: &Position, new: &Position, dice: &Dice) -> BgMove {
        match dice {
            Dice::Regular(dice) => Self::new_regular(old, new, dice),
            Dice::Double(die) => Self::new_double(old, new, *die),
        }
    }

    /// Finds up to two pips where `more[pip] > less[pip]`.
    /// Only looks at positive pips, so it ignores the opponent.
    /// If `less[pip]` is negative (the opponent had a checker there), it will treat at 0, not as -1.
    fn more_checkers(more: &Position, less: &Position) -> [Option<usize>; 2] {
        let mut from: [Option<usize>; 2] = [None; 2];
        let mut from_index = 0_usize;
        for i in (1..26).rev() {
            match more.pip(i) - max(0, less.pip(i)) {
                2 => return [Some(i), Some(i)],
                1 => {
                    from[from_index] = Some(i);
                    from_index += 1;
                }
                _ => {}
            }
        }
        from
    }
}

#[cfg(test)]
mod tests {
    use crate::bg_move::{BgMove, MoveDetail};
    use engine::dice::RegularDice;
    use engine::pos;
    use engine::position::Position;
    use std::collections::HashMap;

    #[test]
    fn double_could_move_only_one_pieces() {
        // Given
        let old = pos!(x 20:1; o 12:2);
        let new = pos!(x 16:2; o 12:2);
        // When
        let bg_move = BgMove::new_double(&old, &new, 4);
        // Then
        assert_eq!(bg_move.details, vec![MoveDetail { from: 20, to: 16 },]);
    }

    #[test]
    fn regular() {
        // Given
        let old = pos!(x 20:2; o 12:2);
        let new = pos!(x 18:1, 15:1; o 12:2);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(5, 2));
        // Then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 20, to: 15 },
                MoveDetail { from: 20, to: 18 },
            ]
        );
    }
}
