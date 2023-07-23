use crate::position::{Position, O_BAR, X_BAR};

impl Position {
    #[allow(dead_code)]
    fn all_regular_moves(&self, die1: usize, die2: usize) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        match self.pips[X_BAR] {
            0 => Vec::new(),
            1 => Vec::new(),
            _ => self.enter_2_move(die1, die2),
        }
    }

    /// All moves (well, exactly one) when at least two checkers are on the bar.
    fn enter_2_move(&self, die1: usize, die2: usize) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] > 1);

        let mut position = self.clone();
        let mut the_move = [None, None];

        if position.can_enter(die1) {
            position.enter_single_checker(die1);
            the_move[0] = Some(X_BAR);
        }
        if position.can_enter(die2) {
            position.enter_single_checker(die2);
            the_move[1] = Some(X_BAR);
        }
        Vec::from([(the_move, position)])
    }

    fn can_enter(&self, die: usize) -> bool {
        debug_assert!(
            self.pips[X_BAR] > 0,
            "only call this function if x has checkers on the bar"
        );
        self.pips[X_BAR - die] > -2
    }

    fn enter_single_checker(&mut self, die: usize) {
        debug_assert!(
            self.pips[X_BAR] > 0,
            "only call this function if x has checkers on the bar"
        );
        debug_assert!(
            self.pips[X_BAR - die] > -2,
            "only call this function if x can enter"
        );
        self.pips[X_BAR] -= 1;
        if self.pips[X_BAR - die] == -1 {
            // hit opponent
            self.pips[X_BAR - die] = 1;
            self.pips[O_BAR] -= 1;
        } else {
            // no hitting
            self.pips[X_BAR - die] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

    #[test]
    fn cannot_enter_with_two_checkers_on_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([None, None], position)]));
    }

    #[test]
    fn can_enter_bigger_die_with_two_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (20, 1), (10, 2)]),
            &HashMap::from([(22, 2)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([Some(X_BAR), None], expected)]));
    }

    #[test]
    fn can_enter_smaller_die_with_two_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 1), (20, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (22, 1), (10, 2)]),
            &HashMap::from([(20, 2), (O_BAR, 1)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([None, Some(X_BAR)], expected)]));
    }

    #[test]
    fn can_enter_both_with_three_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 3), (10, 2)]),
            &HashMap::from([(20, 1)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (22, 1), (20, 1), (10, 2)]),
            &HashMap::from([(O_BAR, 1)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([Some(X_BAR), Some(X_BAR)], expected)]));
    }
}
