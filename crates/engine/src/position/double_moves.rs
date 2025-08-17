use crate::position::{MOVES_CAPACITY, O_BAR, Position, X_BAR};
use std::cmp::min;

impl Position {
    /// Returns a vector of all possible moves when rolling a double.
    #[inline]
    pub(super) fn all_positions_after_double_move(&self, die: usize) -> Vec<Position> {
        if self.pips[X_BAR] > 0 && self.pips[X_BAR - die] < -1 {
            // Has at least one checker on the bar but can't move it
            return vec![*self];
        }

        let (position, number_of_entered_checkers) = self.position_after_entering_checkers(die);
        let max_to_move = 4 - number_of_entered_checkers;
        for checkers in (1..max_to_move + 1).rev() {
            // Let's try to use as many dice as possible. If this doesn't work, try with less dice
            let moves = position.double_moves_after_entering(die, checkers);
            if !moves.is_empty() {
                return moves;
            }
        }
        vec![position]
    }

    /// Returns the position after entering all possible checkers and the number of entered checkers (0 to 4)
    fn position_after_entering_checkers(&self, die: usize) -> (Position, u32) {
        if self.pips[X_BAR] == 0 {
            return (*self, 0);
        }
        debug_assert!(self.pips[X_BAR - die] > -2);
        let number_of_checkers_to_enter = min(4, self.pips[X_BAR]);
        let mut position = *self;
        if number_of_checkers_to_enter > 0 {
            position.pips[X_BAR] -= number_of_checkers_to_enter;
            if self.pips[X_BAR - die] == -1 {
                position.pips[O_BAR] -= 1;
                position.pips[X_BAR - die] = number_of_checkers_to_enter;
            } else {
                position.pips[X_BAR - die] += number_of_checkers_to_enter;
            }
        }
        (position, number_of_checkers_to_enter as u32)
    }

    /// Returns a vector of all possible moves after entering the checkers from the bar.
    /// It takes into account the number of checkers that can be moved (1 to 4).
    fn double_moves_after_entering(&self, die: usize, nr_movable_checkers: u32) -> Vec<Position> {
        let mut moves: Vec<Position> = Vec::with_capacity(MOVES_CAPACITY);
        (self.smallest_pip_to_check(die)..X_BAR).for_each(|i1| {
            if self.can_move_when_bearoff_is_legal(i1, die) {
                let pos = self.clone_and_move_single_checker(i1, die);
                if nr_movable_checkers == 1 {
                    moves.push(pos);
                    return;
                }
                (pos.smallest_pip_to_check(die)..i1 + 1).for_each(|i2| {
                    if pos.can_move_when_bearoff_is_legal(i2, die) {
                        let pos = pos.clone_and_move_single_checker(i2, die);
                        if nr_movable_checkers == 2 {
                            moves.push(pos);
                            return;
                        }
                        (pos.smallest_pip_to_check(die)..i2 + 1).for_each(|i3| {
                            if pos.can_move_when_bearoff_is_legal(i3, die) {
                                let pos = pos.clone_and_move_single_checker(i3, die);
                                if nr_movable_checkers == 3 {
                                    moves.push(pos);
                                    return;
                                }
                                (pos.smallest_pip_to_check(die)..i3 + 1).for_each(|i4| {
                                    if pos.can_move_when_bearoff_is_legal(i4, die) {
                                        let pos = pos.clone_and_move_single_checker(i4, die);
                                        moves.push(pos);
                                    }
                                });
                            }
                        });
                    }
                });
            }
        });
        moves
    }
}

#[cfg(test)]
mod tests {
    use crate::pos;
    use crate::position::{O_BAR, X_BAR};

    #[test]
    fn cannot_enter_from_the_bar() {
        // Given
        let position = pos!(x X_BAR:4; o 22:2);
        // When
        let resulting_positions = position.all_positions_after_double_move(3);
        // Then
        assert_eq!(resulting_positions, vec![position]);
    }

    #[test]
    fn enter_all_four_from_the_bar() {
        // Given
        let actual = pos!(x X_BAR:4; o 22:2, 20:2);
        // When
        let resulting_positions = actual.all_positions_after_double_move(4);
        // Then
        let expected = pos!(x 21:4; o 22:2, 20:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn enter_one_and_move_one_more_and_no_bearoff() {
        // Given
        let actual = pos!(x X_BAR:1, 15:1, 10:1, 4:1; o 22:2, 20:2, 17:3, 11:2, 6:1, 2:2);
        // When
        let resulting_positions = actual.all_positions_after_double_move(4);
        // Then
        let expected = pos!(x 21:1, 15:1, 6:1, 4:1; o 22:2, 20:2, 17:3, 11:2, 2:2, O_BAR:1);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn enter_two_and_move_two_out_of_many() {
        // Given
        let position = pos!(x X_BAR:2, 4:1, 3:1; o 24:2);
        // When
        let resulting_positions = position.all_positions_after_double_move(3);
        // Then
        let expected1 = pos!(x 19:2, 4:1, 3:1; o 24:2);
        let expected2 = pos!(x 22:1, 16:1, 4:1, 3:1; o 24:2);
        let expected3 = pos!(x 22:1, 19:1, 3:1, 1:1; o 24:2);
        assert_eq!(resulting_positions, vec![expected3, expected2, expected1]);
    }

    #[test]
    fn bearoff_4_or_bearoff_less() {
        // Given
        let position = pos!(x 4:1, 3:1, 2:4; o 22:2);
        // When
        let resulting_positions = position.all_positions_after_double_move(2);
        // Then
        let expected1 = pos!(x 2:3, 1:1; o 22:2);
        let expected2 = pos!(x 3:1, 2:2; o 22:2);
        let expected3 = pos!(x 4:1, 2:1, 1:1; o 22:2);
        let expected4 = pos!(x 4:1, 3:1; o 22:2);
        assert_eq!(
            resulting_positions,
            vec![expected4, expected3, expected2, expected1],
        );
    }

    #[test]
    fn no_checkers_on_the_bar_but_would_hit_opponent_if_entering() {
        // Given
        let actual = pos!(x 10:4; o 22:1, 4:2);
        // When
        let resulting_positions = actual.all_positions_after_double_move(3);
        // Then
        let expected = pos!(x 7:4; o 22:1, 4:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn hits_opponent_when_entering_and_cannot_move_afterwards() {
        // Given
        let actual = pos!(x X_BAR:2; o 22:1, 19:2);
        // When
        let resulting_positions = actual.all_positions_after_double_move(3);
        // Then
        let expected = pos!(x 22:2; o 19:2, O_BAR:1);
        assert_eq!(resulting_positions, vec![expected]);
    }
}
