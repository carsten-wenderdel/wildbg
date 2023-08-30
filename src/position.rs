mod double_moves;
mod regular_moves;

use crate::dice::Dice;
use crate::position::GameResult::*;
use crate::position::GameState::*;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Write;

const NO_OF_CHECKERS: u8 = 15;
pub(crate) const X_BAR: usize = 25;
pub(crate) const O_BAR: usize = 0;

pub const STARTING: Position = Position {
    pips: [
        0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
    ],
    x_off: 0,
    o_off: 0,
};

#[derive(Debug, PartialEq)]
pub(crate) enum GameResult {
    WinNormal,
    WinGammon,
    WinBg,
    LoseNormal,
    LoseGammon,
    LoseBg,
}

impl GameResult {
    #[allow(dead_code)]
    pub(crate) fn reverse(&self) -> Self {
        match self {
            WinNormal => LoseNormal,
            WinGammon => LoseGammon,
            WinBg => LoseBg,
            LoseNormal => WinNormal,
            LoseGammon => WinGammon,
            LoseBg => WinBg,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum GameState {
    Ongoing,
    GameOver(GameResult),
}

/// Simple way to create positions for testing
/// The starting position would be:
/// pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2)
/// The order is not important, so this is equivalent:
/// pos!(x 24:2, 13:5, 8:3, 6:5; o 1:2, 12:5, 17:3, 19:5)
#[macro_export]
macro_rules! pos {
    ( x $( $x_pip:tt:$x_checkers:tt ), * ;o $( $o_pip:tt:$o_checkers:tt ), * ) => {
        {
            #[allow(unused_mut)]
            let mut x = HashMap::new();
            $(
                x.insert($x_pip as usize, $x_checkers as u8);
            )*

            #[allow(unused_mut)]
            let mut o = HashMap::new();
            $(
                o.insert($o_pip as usize, $o_checkers as u8);
            )*

            Position::from(&x, &o)
        }
    };
}

/// A single position in backgammon without match information.
/// We assume two players "x" and "o".
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Position {
    // Array positions 25 and 0 are the bar.
    // The other array positions are the pips from the point of view of x, moving from 24 to 0.
    // A positive number means x has that many checkers on that point. Negative for o.
    // Both x_off and o_off are never negative.
    pips: [i8; 26],
    x_off: u8,
    o_off: u8,
}

impl Position {
    #[inline(always)]
    pub(crate) fn x_off(&self) -> u8 {
        self.x_off
    }

    #[inline(always)]
    pub(crate) fn o_off(&self) -> u8 {
        self.o_off
    }

    #[inline(always)]
    /// Number of checkers on the bar for `x`. Non negative number.
    pub(crate) fn x_bar(&self) -> u8 {
        self.pips[X_BAR] as u8
    }

    #[inline(always)]
    /// Number of checkers on the bar for `x`. Non negative number in contrast to internal representation.
    pub(crate) fn o_bar(&self) -> u8 {
        -self.pips[O_BAR] as u8
    }

    #[inline(always)]
    /// Will return positive value for checkers of `x`, negative value for checkers of `o`.
    pub(crate) fn pip(&self, pip: usize) -> i8 {
        debug_assert!((1..=25).contains(&pip));
        self.pips[pip]
    }

    pub(crate) fn game_state(&self) -> GameState {
        debug_assert!(
            self.x_off < 15 || self.o_off < 15,
            "Not both sides can win at the same time"
        );
        if self.x_off == 15 {
            if self.o_off > 0 {
                GameOver(WinNormal)
            } else if self.pips[O_BAR..7].iter().any(|pip| pip < &0) {
                GameOver(WinBg)
            } else {
                GameOver(WinGammon)
            }
        } else if self.o_off == 15 {
            if self.x_off > 0 {
                GameOver(LoseNormal)
            } else if self.pips[19..(X_BAR + 1)].iter().any(|pip| pip > &0) {
                GameOver(LoseBg)
            } else {
                GameOver(LoseGammon)
            }
        } else {
            Ongoing
        }
    }

    /// The return values have switched the sides of the players.
    pub fn all_positions_after_moving(&self, dice: &Dice) -> Vec<Position> {
        debug_assert!(self.o_off < NO_OF_CHECKERS && self.x_off < NO_OF_CHECKERS);
        let mut new_positions = match dice {
            Dice::Double(die) => self.all_positions_after_double_move(*die),
            Dice::Regular(dice) => self.all_positions_after_regular_move(dice),
        };
        for position in new_positions.iter_mut() {
            *position = position.switch_sides();
        }
        new_positions
    }

    pub fn switch_sides(&self) -> Position {
        let mut pips = self.pips.map(|x| -x);
        pips.reverse();
        Position {
            pips,
            x_off: self.o_off,
            o_off: self.x_off,
        }
    }

    pub fn from(x: &HashMap<usize, u8>, o: &HashMap<usize, u8>) -> Position {
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

    /// Only call this if no checkers are on `X_BAR`
    fn can_move_in_board(&self, from: usize, die: usize) -> bool {
        debug_assert!(
            self.pips[X_BAR] == 0,
            "Don't call this function if x has checkers on the bar"
        );
        self.can_move_internally(from, die)
    }

    #[inline(always)]
    fn can_move_internally(&self, from: usize, die: usize) -> bool {
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

    /// Works for all of moves, including those from the bar
    fn can_move(&self, from: usize, die: usize) -> bool {
        if (from == X_BAR) == (self.pips[X_BAR] > 0) {
            self.can_move_internally(from, die)
        } else {
            false
        }
    }

    pub(crate) fn try_move_single_checker(&self, from: usize, die: usize) -> Option<Position> {
        if self.can_move(from, die) {
            Some(self.clone_and_move_single_checker(from, die))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::position::*;
    use std::collections::HashMap;

    #[test]
    fn x_off() {
        let given = pos! {x 3:15; o 1:1};
        assert_eq!(given.x_off(), 0);
        let given = pos! {x 3:10; o 1:1};
        assert_eq!(given.x_off(), 5);
    }

    #[test]
    fn o_off() {
        let given = pos! {x 1:1; o 3:15};
        assert_eq!(given.o_off(), 0);
        let given = pos! {x 1:1; o 3:10};
        assert_eq!(given.o_off(), 5);
    }

    #[test]
    fn x_bar() {
        let given = pos! {x 3:15; o 1:1};
        assert_eq!(given.x_bar(), 0);
        let given = pos! {x X_BAR:2, 3:10; o 1:1};
        assert_eq!(given.x_bar(), 2);
    }

    #[test]
    fn o_bar() {
        let given = pos! {x 1:1; o 3:15};
        assert_eq!(given.o_bar(), 0);
        let given = pos! {x 1:1; o 3:10, O_BAR:1};
        assert_eq!(given.o_bar(), 1);
    }

    #[test]
    fn game_state_bg_when_on_bar() {
        let given = pos!(x 25:1, 1:14; o);
        assert_eq!(given.game_state(), GameOver(LoseBg));
        assert_eq!(
            given.switch_sides().game_state(),
            GameOver(LoseBg.reverse())
        );
    }

    #[test]
    fn game_state_bg_when_not_on_bar() {
        let given = pos!(x 19:15; o);
        assert_eq!(given.game_state(), GameOver(LoseBg));
        assert_eq!(
            given.switch_sides().game_state(),
            GameOver(LoseBg.reverse())
        );
    }

    #[test]
    fn game_state_gammon() {
        let given = pos!(x 18:15; o);
        assert_eq!(given.game_state(), GameOver(LoseGammon));
        assert_eq!(
            given.switch_sides().game_state(),
            GameOver(LoseGammon.reverse())
        );
    }

    #[test]
    fn game_state_normal() {
        let given = pos!(x 19:14; o);
        assert_eq!(given.game_state(), GameOver(LoseNormal));
        assert_eq!(
            given.switch_sides().game_state(),
            GameOver(LoseNormal.reverse())
        );
    }

    #[test]
    fn game_state_ongoing() {
        let given = pos!(x 19:14; o 1:4);
        assert_eq!(given.game_state(), Ongoing);
        assert_eq!(given.switch_sides().game_state(), Ongoing);
    }

    #[test]
    fn all_positions_after_moving_double() {
        // Given
        let pos = pos!(x X_BAR:2, 4:1, 3:1; o 24:2);
        // When
        let positions = pos.all_positions_after_moving(&Dice::new(3, 3));
        // Then
        let expected1 = pos!(x 1:2; o 6:2, 21:1, 22:1);
        let expected2 = pos!(x 1:2; o 3:1, 9:1, 21:1, 22:1);
        let expected3 = pos!(x 1:2; o 3:1, 6:1, 22:1, 24:1);
        assert_eq!(positions, [expected1, expected2, expected3]);
    }

    #[test]
    fn all_positions_after_moving_regular() {
        let pos = pos!(x X_BAR:1; o 22:1);
        // When
        let positions = pos.all_positions_after_moving(&Dice::new(2, 3));
        // Then
        let expected1 = pos!(x X_BAR:1; o 5:1);
        let expected2 = pos!(x 3:1; o 5:1);
        assert_eq!(positions, [expected1, expected2]);
    }

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
        let actual = pos!(x X_BAR:2, 3:2, 1:1; o 24:5, 23:4, 22:6);
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
        let actual = format!("{:?}", pos!(x X_BAR:2, 3:5, 1:1; o 24:7, 23:4, O_BAR:3),);
        let expected = "Position:\nx: {bar:2, 3:5, 1:1, off:7}\no: {off:1, 24:7, 23:4, bar:3}";
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod private_tests {
    use crate::position::{Position, O_BAR, STARTING};
    use std::collections::HashMap;

    #[test]
    fn starting_position_is_correct_and_symmetric() {
        let expected = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        assert_eq!(STARTING, expected);
        assert_eq!(STARTING, STARTING.switch_sides());
    }

    #[test]
    fn move_single_checker_regular_move() {
        let before = pos!(x 4:10; o);
        let actual = before.clone_and_move_single_checker(4, 2);
        let expected = pos!(x 4:9, 2:1; o);
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_hit_opponent() {
        let before = pos!(x 4:10; o 2:1);
        let actual = before.clone_and_move_single_checker(4, 2);
        let expected = pos!(x 4:9, 2:1; o O_BAR:1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_bearoff_regular() {
        let before = pos!(x 4:10; o);
        let actual = before.clone_and_move_single_checker(4, 4);
        let expected = pos!(x 4:9; o);
        assert_eq!(actual, expected);
    }

    #[test]
    fn move_single_checker_bearoff_from_lower_pip() {
        let before = pos!(x 4:10; o);
        let actual = before.clone_and_move_single_checker(4, 5);
        let expected = pos!(x 4:9; o);
        assert_eq!(actual, expected);
    }

    #[test]
    fn cannot_move_no_checker() {
        let given = pos!(x 4:10; o);
        assert!(!given.can_move_in_board(5, 2));
    }

    #[test]
    fn cannot_move_opposing_checker() {
        let given = Position::from(&HashMap::new(), &HashMap::from([(4, 10)]));
        assert!(!given.can_move_in_board(4, 2));
    }

    #[test]
    fn cannot_move_would_land_on_two_opposing_checkers() {
        let given = pos!(x 4:10; o 2:2);
        assert!(!given.can_move_in_board(4, 2));
    }

    #[test]
    fn can_move_will_land_on_one_opposing_checker() {
        let given = pos!(x 4:10; o 2:1);
        assert!(given.can_move_in_board(4, 2));
    }

    #[test]
    fn can_move_will_land_on_checkers() {
        let given = pos!(x 4:10; o 2:1);
        assert!(given.can_move_in_board(4, 2));
    }

    #[test]
    fn cannot_move_bear_off_illegal_because_other_checkers() {
        let given = pos!(x 10:2, 4:10; o);
        assert!(!given.can_move_in_board(4, 4));
    }

    #[test]
    fn can_move_will_bear_off_exactly() {
        let given = pos!(x 4:10; o);
        assert!(given.can_move_in_board(4, 4));
    }

    #[test]
    fn cannot_move_bear_off_skipping_illegal_because_other_checkers() {
        let given = pos!(x 10:2, 4:10; o);
        assert!(!given.can_move_in_board(4, 6));
    }

    #[test]
    fn can_move_will_bear_off_skipping() {
        let given = pos!(x 4:10; o);
        assert!(given.can_move_in_board(4, 6));
    }
}
