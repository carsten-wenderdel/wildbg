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
