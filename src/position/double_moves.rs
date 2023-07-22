use crate::position::{Position, O_BAR, X_BAR};
use std::cmp::max;

impl Position {
    #[allow(dead_code)]
    fn all_double_moves(&self, die: usize) -> Vec<usize> {
        if self.pips[X_BAR] > 0 && self.pips[X_BAR - die] <= -1 {
            // Has at least one checker on the bar but can't move it
            return Vec::new();
        }

        #[allow(unused_variables)]
        let (position, number_of_entered_checkers) = self.position_after_entering_checkers(die);
        if number_of_entered_checkers == 4 {
            return vec![X_BAR; 4];
        }
        // Todo: implement possible double moves
        Vec::new()
    }

    fn position_after_entering_checkers(&self, die: usize) -> (&Position, u32) {
        if self.pips[X_BAR] == 0 {
            return (self, 4);
        }
        debug_assert!(self.pips[X_BAR - die] > -2);
        let number_of_checkers_to_enter = max(4, self.pips[X_BAR]);
        let mut position = self.clone();
        position.pips[X_BAR] -= number_of_checkers_to_enter;
        position.pips[X_BAR - die] = number_of_checkers_to_enter;
        if self.pips[X_BAR - die] == -1 {
            position.pips[O_BAR] -= 1;
        }
        (self, number_of_checkers_to_enter as u32)
    }

    /// Will return 4 if 4 or more checkers can be moved.
    /// The return value is never bigger than `number_of_entered_checkers`.
    /// Will return 0 if no checker can be moved.
    #[allow(dead_code)]
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
    use crate::position::{Position, X_BAR};
    use std::collections::HashMap;

    #[test]
    fn cannot_enter_from_the_bar() {
        // Given
        let actual = Position::from(&HashMap::from([(X_BAR, 4)]), &HashMap::from([(22, 2)]));
        // When
        let v = actual.all_double_moves(3);
        // Then
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn enter_all_four_from_the_bar() {
        // Given
        let actual = Position::from(
            &HashMap::from([(X_BAR, 4)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        // When
        let v = actual.all_double_moves(4);
        // Then
        assert_eq!(v, Vec::from([X_BAR, X_BAR, X_BAR, X_BAR]));
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
