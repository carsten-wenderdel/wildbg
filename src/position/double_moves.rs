use crate::position::{Position, X_BAR};

impl Position {
    #[allow(dead_code)]
    fn all_double_moves(&self, die: usize) -> Vec<u8> {
        if self.pips[X_BAR] > 0 && self.pips[X_BAR - die] <= -1 {
            // Has at least one checker on the bar but can't move it
            return Vec::new();
        }

        // Todo: implement possible double moves
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::position::{Position, X_BAR};
    use std::collections::HashMap;

    #[test]
    fn cannot_move_from_the_bar() {
        // Given
        let actual = Position::from(&HashMap::from([(X_BAR, 4)]), &HashMap::from([(22, 2)]));
        // When
        let v = actual.all_double_moves(3);
        // Then
        assert_eq!(v.len(), 0)
    }
}
