use crate::bg_move::{BgMove, MoveDetail};
use engine::position::Position;

impl BgMove {
    pub(super) fn new_double(old: &Position, new: &Position, die: usize) -> BgMove {
        let mut details: Vec<MoveDetail> = Vec::with_capacity(4);
        let mut index = 25_usize;
        let mut position = *old;
        while index > 0 {
            if position.pip(index) > new.pip(index) {
                // We could speed this up by replacing the next line with the private method `move_single_checker`
                position = position.try_move_single_checker(index, die).unwrap();
                let to = index.saturating_sub(die);
                details.push(MoveDetail { from: index, to })
            } else {
                index -= 1
            }
        }
        BgMove { details }
    }
}

#[cfg(test)]
mod tests {
    use crate::bg_move::{BgMove, MoveDetail};
    use engine::pos;

    #[test]
    fn could_not_move() {
        // Given
        let old = pos!(x 20:4; o 16:2);
        let new = old;
        // When
        let bg_move = BgMove::new_double(&old, &new, 4);
        // Then
        assert_eq!(bg_move.details, Vec::new());
    }

    #[test]
    fn could_move_only_two_pieces() {
        // Given
        let old = pos!(x 20:2; o 12:2);
        let new = pos!(x 16:2; o 12:2);
        // When
        let bg_move = BgMove::new_double(&old, &new, 4);
        // Then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 20, to: 16 },
                MoveDetail { from: 20, to: 16 },
            ]
        );
    }

    #[test]
    fn bear_off() {
        // Given
        let old = pos!(x 5:1, 4:1, 3:2; o 12:2);
        let new = pos!(x 1:1; o 12:2);
        // When
        let bg_move = BgMove::new_double(&old, &new, 4);
        // Then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 5, to: 1 },
                MoveDetail { from: 4, to: 0 },
                MoveDetail { from: 3, to: 0 },
                MoveDetail { from: 3, to: 0 },
            ]
        );
    }
}
