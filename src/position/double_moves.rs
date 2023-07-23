use crate::position::{Position, O_BAR, X_BAR};
use std::cmp::min;

impl Position {
    /// Returns a vector of all possible moves when rolling a double.
    /// The return value both contains the moves and the resulting positions.
    /// The move is encoded in an array of 4 numbers, representing the pip from where to move.
    /// If a checker cannot be moved, the corresponding number in the array is `O_BAR`.
    pub(super) fn all_double_moves(&self, die: usize) -> Vec<([usize; 4], Position)> {
        if self.pips[X_BAR] > 0 && self.pips[X_BAR - die] <= -1 {
            // Has at least one checker on the bar but can't move it
            return Vec::from([([O_BAR, O_BAR, O_BAR, O_BAR], self.clone())]);
        }

        let (position, number_of_entered_checkers) = self.position_after_entering_checkers(die);
        if number_of_entered_checkers == 4 {
            return Vec::from([([X_BAR, X_BAR, X_BAR, X_BAR], position.clone())]);
        }

        let mut moves = position.double_moves_after_entering(die, number_of_entered_checkers);
        if number_of_entered_checkers != 0 {
            for x in moves.iter_mut() {
                x.0.rotate_right(number_of_entered_checkers as usize);
                for i in 0..number_of_entered_checkers as usize {
                    x.0[i] = X_BAR;
                }
            }
        }
        moves
    }

    fn position_after_entering_checkers(&self, die: usize) -> (Position, u32) {
        if self.pips[X_BAR] == 0 {
            return (self.clone(), 0);
        }
        debug_assert!(self.pips[X_BAR - die] > -2);
        let number_of_checkers_to_enter = min(4, self.pips[X_BAR]);
        let mut position = self.clone();
        position.pips[X_BAR] -= number_of_checkers_to_enter;
        position.pips[X_BAR - die] = number_of_checkers_to_enter;
        if self.pips[X_BAR - die] == -1 {
            position.pips[O_BAR] -= 1;
        }
        (position, number_of_checkers_to_enter as u32)
    }

    /// Returns a vector of all possible moves after entering the checkers from the bar.
    /// It takes into account the number of already entered checkers.
    fn double_moves_after_entering(
        &self,
        die: usize,
        number_of_entered_checkers: u32,
    ) -> Vec<([usize; 4], Position)> {
        let nr_movable_checkers = self.number_of_movable_checkers(die, number_of_entered_checkers);
        let mut moves: Vec<([usize; 4], Position)> = Vec::new();
        if nr_movable_checkers == 0 {
            return moves;
        }
        for i1 in (1..X_BAR).rev() {
            if self.can_move(i1, die) {
                let pos = self.clone_and_move_single_checker(i1, die);
                if nr_movable_checkers == 1 {
                    moves.push(([i1, O_BAR, O_BAR, O_BAR], pos));
                    continue;
                }
                for i2 in (1..i1 + 1).rev() {
                    if pos.can_move(i2, die) {
                        let pos = pos.clone_and_move_single_checker(i2, die);
                        if nr_movable_checkers == 2 {
                            moves.push(([i1, i2, O_BAR, O_BAR], pos));
                            continue;
                        }
                        for i3 in (1..i2 + 1).rev() {
                            if pos.can_move(i3, die) {
                                let pos = pos.clone_and_move_single_checker(i3, die);
                                if nr_movable_checkers == 3 {
                                    moves.push(([i1, i2, i3, O_BAR], pos));
                                    continue;
                                }
                                for i4 in (1..i3 + 1).rev() {
                                    if pos.can_move(i4, die) {
                                        let pos = pos.clone_and_move_single_checker(i4, die);
                                        moves.push(([i1, i2, i3, i4], pos));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        moves
    }

    /// Will return 4 if 4 or more checkers can be moved.
    /// The return value is never bigger than `number_of_entered_checkers`.
    /// Will return 0 if no checker can be moved.
    fn number_of_movable_checkers(&self, die: usize, number_of_entered_checkers: u32) -> u32 {
        let mut number_of_checkers = 0;
        let mut pip = 24;
        let mut position = self.clone();
        while number_of_checkers < 4 - number_of_entered_checkers && pip > 0 {
            if position.can_move(pip, die) {
                position.move_single_checker(pip, die);
                number_of_checkers += 1;
            } else {
                pip -= 1;
            }
        }
        number_of_checkers
    }
}

#[cfg(test)]
mod tests {
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

    #[test]
    fn cannot_enter_from_the_bar() {
        // Given
        let position = Position::from(&HashMap::from([(X_BAR, 4)]), &HashMap::from([(22, 2)]));
        // When
        let moves = position.all_double_moves(3);
        // Then
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].0, [O_BAR, O_BAR, O_BAR, O_BAR]);
        assert_eq!(moves[0].1, position);
    }

    #[test]
    fn enter_all_four_from_the_bar() {
        // Given
        let actual = Position::from(
            &HashMap::from([(X_BAR, 4)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        // When
        let moves = actual.all_double_moves(4);
        // Then
        let expected = Position::from(
            &HashMap::from([(21, 4)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        assert_eq!(moves, Vec::from([([X_BAR, X_BAR, X_BAR, X_BAR], expected)]));
    }

    #[test]
    fn enter_one_and_move_one_more_and_no_bearoff() {
        // Given
        let actual = Position::from(
            &HashMap::from([(X_BAR, 1), (15, 1), (10, 1), (4, 1)]),
            &HashMap::from([(22, 2), (20, 2), (17, 3), (11, 2), (6, 1), (2, 2)]),
        );
        // When
        let moves = actual.all_double_moves(4);
        // Then
        let expected = Position::from(
            &HashMap::from([(21, 1), (15, 1), (6, 1), (4, 1)]),
            &HashMap::from([(22, 2), (20, 2), (17, 3), (11, 2), (2, 2), (O_BAR, 1)]),
        );
        assert_eq!(moves, Vec::from([([X_BAR, 10, O_BAR, O_BAR], expected)]));
    }

    #[test]
    fn enter_two_and_move_two_out_of_many() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (4, 1), (3, 1)]),
            &HashMap::from([(24, 2)]),
        );
        // When
        let moves = position.all_double_moves(3);
        // Then
        let expected1 = (
            [X_BAR, X_BAR, 22, 22],
            Position::from(
                &HashMap::from([(19, 2), (4, 1), (3, 1)]),
                &HashMap::from([(24, 2)]),
            ),
        );
        let expected2 = (
            [X_BAR, X_BAR, 22, 19],
            Position::from(
                &HashMap::from([(22, 1), (16, 1), (4, 1), (3, 1)]),
                &HashMap::from([(24, 2)]),
            ),
        );
        let expected3 = (
            [X_BAR, X_BAR, 22, 4],
            Position::from(
                &HashMap::from([(22, 1), (19, 1), (3, 1), (1, 1)]),
                &HashMap::from([(24, 2)]),
            ),
        );
        assert_eq!(moves.len(), 3);
        assert_eq!(moves, Vec::from([expected1, expected2, expected3]));
    }

    #[test]
    fn bearoff_4_or_bearoff_less() {
        // Given
        let position = Position::from(
            &HashMap::from([(4, 1), (3, 1), (2, 4)]),
            &HashMap::from([(22, 2)]),
        );
        // When
        let moves = position.all_double_moves(2);
        // Then
        let expected1 = (
            [4, 3, 2, 2],
            Position::from(&HashMap::from([(2, 3), (1, 1)]), &HashMap::from([(22, 2)])),
        );
        let expected2 = (
            [4, 2, 2, 2],
            Position::from(&HashMap::from([(3, 1), (2, 2)]), &HashMap::from([(22, 2)])),
        );
        let expected3 = (
            [3, 2, 2, 2],
            Position::from(
                &HashMap::from([(4, 1), (2, 1), (1, 1)]),
                &HashMap::from([(22, 2)]),
            ),
        );
        let expected4 = (
            [2, 2, 2, 2],
            Position::from(&HashMap::from([(4, 1), (3, 1)]), &HashMap::from([(22, 2)])),
        );
        assert_eq!(moves.len(), 4);
        assert_eq!(
            moves,
            Vec::from([expected1, expected2, expected3, expected4])
        );
    }
}

#[cfg(test)]
mod private_tests {
    use crate::position::Position;
    use std::collections::HashMap;

    #[test]
    fn number_of_movable_checkers_when_completely_blocked() {
        // Given
        let position = Position::from(&HashMap::from([(20, 2)]), &HashMap::from([(16, 2)]));
        // When
        let actual = position.number_of_movable_checkers(4, 0);
        // Then
        assert_eq!(actual, 0);
    }

    #[test]
    fn number_of_movable_checkers_when_many_moves_would_be_possible() {
        // Given
        let position = Position::from(&HashMap::from([(20, 2)]), &HashMap::from([(16, 1)]));
        // When
        let actual = position.number_of_movable_checkers(4, 0);
        // Then
        assert_eq!(actual, 4);
    }

    #[test]
    fn number_of_movable_checkers_when_one_checker_was_entered_from_bar() {
        // Given
        let position = Position::from_x(&HashMap::from([(20, 2)]));
        // When
        let actual = position.number_of_movable_checkers(4, 1);
        // Then
        assert_eq!(actual, 3);
    }

    #[test]
    fn number_of_movable_checkers_when_blocked_after_one_move() {
        // Given
        let position = Position::from(&HashMap::from([(20, 2)]), &HashMap::from([(12, 2)]));
        // When
        let actual = position.number_of_movable_checkers(4, 0);
        // Then
        assert_eq!(actual, 2);
    }
}
