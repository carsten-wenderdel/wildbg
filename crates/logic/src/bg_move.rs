use engine::dice::Dice;
use engine::position::Position;
#[cfg(feature = "web")]
use serde::Serialize;
use std::cmp::max;
#[cfg(feature = "web")]
use utoipa::ToSchema;

mod double;
mod mixed;

/// `BgMove` is not used during rollouts or evaluation but only when returning moves via an API
/// This is why a new `BgMove` is always calculated based on a `old` and a resulting `new` position.
#[derive(Debug, PartialEq)]
pub struct BgMove {
    pub(crate) details: Vec<MoveDetail>,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "web", derive(Serialize, ToSchema))]
/// Single movement of one checker. We always move from bigger pips to smaller pips.
/// If the same checker is moved more than once, multiple `MoveDetail`s are given.
/// Therefore: `from > to` and `from - to <= 6`.
pub struct MoveDetail {
    /// The bar is represented by `25`.
    #[cfg_attr(feature = "web", schema(minimum = 1, maximum = 25))]
    pub(crate) from: usize,
    /// bear off is represented by `0`.
    #[cfg_attr(feature = "web", schema(minimum = 0, maximum = 24))]
    pub(crate) to: usize,
}

impl MoveDetail {
    pub fn from(&self) -> usize {
        self.from
    }

    pub fn to(&self) -> usize {
        self.to
    }
}

impl BgMove {
    #[inline]
    pub fn into_details(self) -> Vec<MoveDetail> {
        self.details
    }

    pub fn new(old: &Position, new: &Position, dice: &Dice) -> BgMove {
        match dice {
            Dice::Mixed(dice) => Self::new_mixed(old, new, dice),
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
    use engine::dice::MixedDice;
    use engine::pos;

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
    fn mixed() {
        // Given
        let old = pos!(x 20:2; o 12:2);
        let new = pos!(x 18:1, 15:1; o 12:2);
        // When
        let bg_move = BgMove::new_mixed(&old, &new, &MixedDice::new(5, 2));
        // Then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 20, to: 15 },
                MoveDetail { from: 20, to: 18 },
            ]
        );
    }

    #[test]
    fn bear_off_one_die_used() {
        // Given
        let old = pos!(x 1:1; o 23:4, 24:3);
        let new = pos!(x 1:3, 2:4; o).sides_switched(); // The macro only works when `x` has checkers

        // When
        let bg_move = BgMove::new_mixed(&old, &new, &MixedDice::new(5, 2));

        // Then
        assert_eq!(bg_move.details, vec![MoveDetail { from: 1, to: 0 },]);
    }

    #[test]
    fn bear_off_use_two_dice_instead_of_one() {
        // Given
        let old = pos!(x 4:1; o 23:4, 24:3);
        let new = pos!(x 1:3, 2:4; o).sides_switched(); // The macro only works when `x` has checkers

        // When
        let bg_move = BgMove::new_mixed(&old, &new, &MixedDice::new(5, 2));

        // Then
        assert_eq!(
            bg_move.details,
            vec![MoveDetail { from: 4, to: 2 }, MoveDetail { from: 2, to: 0 },]
        );
    }
}
