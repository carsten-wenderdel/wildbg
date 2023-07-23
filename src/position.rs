mod double_moves;
mod regular_moves;

use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Write;

const NO_OF_CHECKERS: u8 = 15;
const X_BAR: usize = 25;
const O_BAR: usize = 0;

/// A single position in backgammon without match information.
/// We assume two players "x" and "o".
#[derive(Clone, PartialEq)]
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

    fn from(x: &HashMap<usize, u8>, o: &HashMap<usize, u8>) -> Position {
        let mut pips = [0; 26];
        for (i, v) in x {
            pips[*i] = *v as i8;
        }
        for (i, v) in o {
            debug_assert!(pips[*i] == 0);
            pips[*i] = -(*v as i8);
        }
        let x_sum = x.values().sum::<u8>();
        let o_sum = o.values().sum::<u8>();
        debug_assert!(x_sum <= NO_OF_CHECKERS && o_sum <= NO_OF_CHECKERS);
        Position {
            pips,
            x_off: NO_OF_CHECKERS - x_sum,
            o_off: NO_OF_CHECKERS - o_sum,
        }
    }

    #[allow(dead_code)]
    fn from_x(x: &HashMap<usize, u8>) -> Position {
        Self::from(x, &HashMap::new())
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Position:").unwrap();

        // Write x:
        let mut s = String::from("x: {");
        if self.pips[X_BAR] > 0 {
            write!(s, "bar:{}, ", self.pips[X_BAR]).unwrap();
        }
        for i in (1..X_BAR).rev() {
            if self.pips[i] > 0 {
                write!(s, "{}:{}, ", i, self.pips[i]).unwrap();
            }
        }
        if self.x_off > 0 {
            write!(s, "off:{}, ", self.x_off).unwrap();
        }
        s.pop(); // remove last ", "
        s.pop();
        writeln!(s, "}}").unwrap();
        write!(f, "{}", s).unwrap();

        // Write o:
        let mut s = String::from("o: {");
        if self.o_off > 0 {
            write!(s, "off:{}, ", self.o_off).unwrap();
        }
        for i in (1..X_BAR).rev() {
            if self.pips[i] < 0 {
                write!(s, "{}:{}, ", i, -self.pips[i]).unwrap();
            }
        }
        if self.pips[O_BAR] < 0 {
            write!(s, "bar:{}, ", -self.pips[O_BAR]).unwrap();
        }
        s.pop(); // remove last ", "
        s.pop();
        write!(s, "}}").unwrap();
        write!(f, "{}", s)
    }
}

/// Private helper methods
impl Position {
    /// Only call if this move is legal.
    fn move_single_checker(&mut self, from: usize, die: usize) {
        self.pips[from] -= 1;
        if from > die {
            if self.pips[from - die] == -1 {
                // hit opponent
                self.pips[from - die] = 1;
                self.pips[O_BAR] -= 1;
            } else {
                // regular move
                self.pips[from - die] += 1;
            }
        } else {
            // bear off
            self.x_off += 1;
        }
    }

    /// Only call if this move is legal.
    fn clone_and_move_single_checker(&self, from: usize, die: usize) -> Position {
        let mut new = self.clone();
        new.move_single_checker(from, die);
        new
    }

    fn can_move(&self, from: usize, die: usize) -> bool {
        debug_assert!(
            self.pips[X_BAR] == 0,
            "Don't call this function if x has checkers on the bar"
        );
        return if self.pips[from] < 1 {
            // no checker to move
            false
        } else if from > die {
            // regular move, no bear off
            let number_of_opposing_checkers = self.pips[from - die];
            number_of_opposing_checkers > -2
        } else if from == die {
            // bear off
            let checker_out_of_homeboard = self.pips[7..X_BAR].iter().any(|x| x > &0);
            !checker_out_of_homeboard
        } else {
            // from < die, bear off
            let checker_on_bigger_pip = self.pips[from + 1..X_BAR].iter().any(|x| x > &0);
            !checker_on_bigger_pip
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

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

    #[test]
    fn from() {
        let actual = Position::from(
            &HashMap::from([(X_BAR, 2), (3, 2), (1, 1)]),
            &HashMap::from([(24, 5), (23, 4), (22, 6)]),
        );
        let expected = Position {
            pips: [
                0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, -4, -5, 2,
            ],
            x_off: 10,
            o_off: 0,
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn debug() {
        let actual = format!(
            "{:?}",
            Position::from(
                &HashMap::from([(X_BAR, 2), (3, 5), (1, 1)]),
                &HashMap::from([(24, 7), (23, 4), (O_BAR, 3)])
            )
        );
        let expected = "Position:\nx: {bar:2, 3:5, 1:1, off:7}\no: {off:1, 24:7, 23:4, bar:3}";
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod private_tests {
    use crate::position::{Position, O_BAR};
    use std::collections::HashMap;

    #[test]
    fn move_single_checker_regular_move() {
        let before = Position::from_x(&HashMap::from([(4, 10)]));
        let actual = before.clone_and_move_single_checker(4, 2);
        let expected = Position::from_x(&HashMap::from([(4, 9), (2, 1)]));
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_hit_opponent() {
        let before = Position::from(&HashMap::from([(4, 10)]), &HashMap::from([(2, 1)]));
        let actual = before.clone_and_move_single_checker(4, 2);
        let expected = Position::from(
            &HashMap::from([(4, 9), (2, 1)]),
            &HashMap::from([(O_BAR, 1)]),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_bearoff_regular() {
        let before = Position::from_x(&HashMap::from([(4, 10)]));
        let actual = before.clone_and_move_single_checker(4, 4);
        let expected = Position::from_x(&HashMap::from([(4, 9)]));
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_bearoff_from_lower_pip() {
        let before = Position::from_x(&HashMap::from([(4, 10)]));
        let actual = before.clone_and_move_single_checker(4, 5);
        let expected = Position::from_x(&HashMap::from([(4, 9)]));
        assert_eq!(actual, expected);
    }

    #[test]
    fn cannot_move_no_checker() {
        let given = Position::from_x(&HashMap::from([(4, 10)]));
        assert!(!given.can_move(5, 2));
    }

    #[test]
    fn cannot_move_opposing_checker() {
        let given = Position::from(&HashMap::new(), &HashMap::from([(4, 10)]));
        assert!(!given.can_move(4, 2));
    }

    #[test]
    fn cannot_move_would_land_on_two_opposing_checkers() {
        let given = Position::from(&HashMap::from([(4, 10)]), &HashMap::from([(2, 2)]));
        assert!(!given.can_move(4, 2));
    }

    #[test]
    fn can_move_will_land_on_one_opposing_checker() {
        let given = Position::from(&HashMap::from([(4, 10)]), &HashMap::from([(2, 1)]));
        assert!(given.can_move(4, 2));
    }

    #[test]
    fn can_move_will_land_on_checkers() {
        let given = Position::from(&HashMap::from([(4, 10)]), &HashMap::from([(2, 1)]));
        assert!(given.can_move(4, 2));
    }

    #[test]
    fn cannot_move_bear_off_illegal_because_other_checkers() {
        let given = Position::from_x(&HashMap::from([(10, 2), (4, 10)]));
        assert!(!given.can_move(4, 4));
    }

    #[test]
    fn can_move_will_bear_off_exactly() {
        let given = Position::from_x(&HashMap::from([(4, 10)]));
        assert!(given.can_move(4, 4));
    }

    #[test]
    fn cannot_move_bear_off_skipping_illegal_because_other_checkers() {
        let given = Position::from_x(&HashMap::from([(10, 2), (4, 10)]));
        assert!(!given.can_move(4, 6));
    }

    #[test]
    fn can_move_will_bear_off_skipping() {
        let given = Position::from_x(&HashMap::from([(4, 10)]));
        assert!(given.can_move(4, 6));
    }
}
