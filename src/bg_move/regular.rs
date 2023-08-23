use crate::bg_move::{BgMove, MoveDetail};
use crate::dice_gen::RegularDice;
use crate::position::Position;

impl BgMove {
    #[allow(dead_code)]
    pub(super) fn new_regular(old: &Position, new: &Position, dice: &RegularDice) -> BgMove {
        let from = Self::more_checkers(old, new);
        let details: Vec<MoveDetail> = match from {
            [None, None] => Vec::new(),
            [None, Some(_)] => panic!("BgMove: If index 0 is None, index 1 must also be None."),
            [Some(from_pip), None] => {
                // Let's see whether one checker as only moved once:
                for die in [dice.small, dice.big] {
                    if let Some(position) = old.try_move_single_checker(from_pip, die) {
                        if position == *new {
                            let to = from_pip.saturating_sub(die);
                            return BgMove {
                                details: vec![MoveDetail { from: from_pip, to }],
                            };
                        }
                    }
                }
                // One checker was moved twice. Where was the stopover?
                for (die1, die2) in [(dice.small, dice.big), (dice.big, dice.small)] {
                    if let Some(position) = old.can_move_both(from_pip, die1, from_pip - die1, die2)
                    {
                        if position == *new {
                            // In case of bear off, we want 0, not negative numbers
                            let to = from_pip.saturating_sub(die1 + die2);
                            return BgMove {
                                details: vec![
                                    MoveDetail {
                                        from: from_pip,
                                        to: from_pip - die1,
                                    },
                                    MoveDetail {
                                        from: from_pip - die1,
                                        to,
                                    },
                                ],
                            };
                        }
                    }
                }
                panic!("One of the previous if/else branches should have returned something.");
            }
            [Some(from_pip_1), Some(from_pip_2)] => {
                // Two different checkers were moved. But how exactly?
                // There are four different combinations, let's try them out:
                let combinations: [(usize, usize, usize, usize); 4] = [
                    (from_pip_1, dice.big, from_pip_2, dice.small),
                    (from_pip_1, dice.small, from_pip_2, dice.big),
                    (from_pip_2, dice.big, from_pip_1, dice.small),
                    (from_pip_2, dice.small, from_pip_1, dice.big),
                ];
                let (from1, die1, from2, die2) = combinations
                    .into_iter()
                    .find(|(from1, die1, from2, die2)| {
                        old.can_move_both(*from1, *die1, *from2, *die2) == Some(new.clone())
                    })
                    .expect("some move combination should work");
                let to1 = from1.saturating_sub(die1);
                let to2 = from2.saturating_sub(die2);
                vec![
                    MoveDetail {
                        from: from1,
                        to: to1,
                    },
                    MoveDetail {
                        from: from2,
                        to: to2,
                    },
                ]
            }
        };
        BgMove { details }
    }
}

/// Checks whether it's legal to move from `from1` with `die1` and after that from `from2`
/// with `die2`. Returns the resulting position if legal.
impl Position {
    fn can_move_both(
        &self,
        from1: usize,
        die1: usize,
        from2: usize,
        die2: usize,
    ) -> Option<Position> {
        debug_assert!(die1 != die2);
        self.try_move_single_checker(from1, die1)
            .and_then(|p| p.try_move_single_checker(from2, die2))
    }
}

#[cfg(test)]
mod tests {
    use crate::bg_move::{BgMove, MoveDetail};
    use crate::dice_gen::RegularDice;
    use crate::pos;
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

    #[test]
    fn could_not_move() {
        // Given
        let old = pos!(x 20:4; o 16:2, 13:2);
        let new = pos!(x 20:4; o 16:2, 14:1);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // Then
        assert_eq!(bg_move.details, Vec::new());
    }

    #[test]
    fn moved_one_checker_from_bar() {
        // Given
        let old = pos!(x X_BAR:2; o 21:2);
        let new = pos!(x X_BAR:1, 19:1; o 21:2);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // Then
        assert_eq!(
            bg_move.details,
            vec![MoveDetail {
                from: X_BAR,
                to: 19
            },]
        );
    }

    #[test]
    fn hit_piece_midway() {
        // Given
        let old = pos!(x 20:4, 10:1; o 24:1, 14:1);
        let new = pos!(x 20:3, 10:2; o 24:1, O_BAR: 1);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 20, to: 14 },
                MoveDetail { from: 14, to: 10 }
            ]
        );
    }

    #[test]
    fn hit_nothing_midway() {
        // Given
        let old = pos!(x 20:4, 10:1; o 24:1, 14:1);
        let new = pos!(x 20:3, 10:2; o 24:1, 14:1);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // then
        assert_eq!(
            bg_move.details,
            vec![
                MoveDetail { from: 20, to: 16 },
                MoveDetail { from: 16, to: 10 }
            ]
        );
    }

    #[test]
    fn hit_nothing_and_bear_off() {
        // Given
        let old = pos!(x 8:1, 1:10; o 2:2);
        let new = pos!(x 1:10; o 2:2);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // then
        assert_eq!(
            bg_move.details,
            vec![MoveDetail { from: 8, to: 4 }, MoveDetail { from: 4, to: 0 }]
        );
    }

    #[test]
    fn two_bear_offs_from_different_points() {
        // Given
        let old = pos!(x 4:1, 3:1, 1:1; o 2:2);
        let new = pos!(x 1:1; o 2:2);
        // When
        let bg_move = BgMove::new_regular(&old, &new, &RegularDice::new(6, 4));
        // then
        assert_eq!(
            bg_move.details,
            vec![MoveDetail { from: 4, to: 0 }, MoveDetail { from: 3, to: 0 }]
        );
    }
}
