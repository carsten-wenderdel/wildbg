/// A single position in backgammon without match information.
/// We assume two players "x" and "o".
#[derive(Debug, PartialEq)]
struct Position {
    // Array positions 25 and 0 are the bar.
    // The other array positions are the pips from the point of view of x, moving from 24 to 0.
    // A positive number means x has that many checkers on that point. Negative for o.
    pips: [i8; 26],
    x_off: u8,
    o_off: u8,
}

impl Position {
    #[allow(dead_code)]
    fn switch_sides(&self) -> Position {
        let mut pips = self.pips.map(|x| -x);
        pips.reverse();
        Position {
            pips,
            x_off: self.o_off,
            o_off: self.x_off,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::position::Position;

    #[test]
    fn switch_sides() {
        // Given
        let original = Position {
            pips: [
                2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0,
            ],
            x_off: 0,
            o_off: 3,
        };
        // When
        let actual = original.switch_sides();
        // Then
        let expected = Position {
            pips: [
                0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -2, -2,
            ],
            x_off: 3,
            o_off: 0,
        };
        assert_eq!(actual, expected);
    }
}
