/// A single position in backgammon without match information.
/// We assume two players "x" and "o".
#[derive(Debug, PartialEq)]
struct Position {
    pips: [i8; 24],
    x_out: OutCheckers,
    o_out: OutCheckers,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OutCheckers {
    bar: u8,
    off: u8,
}

impl Position {
    fn switch_sides(&self) -> Position {
        let mut pips = self.pips.map(|x| -x);
        pips.reverse();
        Position {
            pips,
            x_out: self.o_out,
            o_out: self.x_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::position::{OutCheckers, Position};

    #[test]
    fn switch_sides() {
        // Given
        let original = Position {
            pips: [
                2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2,
            ],
            x_out: OutCheckers { bar: 2, off: 0 },
            o_out: OutCheckers { bar: 0, off: 3 },
        };
        // When
        let actual = original.switch_sides();
        // Then
        let expected = Position {
            pips: [
                2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,
            ],
            x_out: OutCheckers { bar: 0, off: 3 },
            o_out: OutCheckers { bar: 2, off: 0 },
        };
        assert_eq!(actual, expected);
    }
}
