mod conversion;
mod double_moves;
mod mixed_moves;

use crate::dice::Dice;
use crate::position::GameResult::*;
use crate::position::GameState::*;
use crate::position::OngoingPhase::{Contact, Race};
use std::cmp::min;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Write;

const NUM_OF_CHECKERS: u8 = 15;
pub const X_BAR: usize = 25;
pub const O_BAR: usize = 0;

/// It helps performance during move generation to initialize vectors with a given capacity.
/// It also helps the compiler optimizing, when this number is the same in all places.
/// A good capacity is 128 or 256 on Apple silicon. Smaller numbers mean more reallocations.
/// Bigger numbers mean too much memory wasted.
const MOVES_CAPACITY: usize = 128;

pub const STARTING: Position = Position {
    pips: [
        0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
    ],
    x_off: 0,
    o_off: 0,
};

#[derive(Clone, Debug, PartialEq)]
pub enum GameResult {
    WinNormal,
    WinGammon,
    WinBg,
    LoseNormal,
    LoseGammon,
    LoseBg,
}

impl GameResult {
    pub fn reverse(&self) -> Self {
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
pub enum GameState {
    Ongoing,
    GameOver(GameResult),
}

#[derive(Debug, PartialEq)]
pub enum OngoingPhase {
    Contact,
    Race,
}

#[derive(Debug, PartialEq)]
pub enum GamePhase {
    Ongoing(OngoingPhase),
    GameOver(GameResult),
}

/// A single position in backgammon without match information.
/// We assume two players "x" and "o".
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    // Array positions 25 and 0 are the bar.
    // The other array positions are the pips from the point of view of x, moving from 24 to 0.
    // A positive number means x has that many checkers on that point. Negative for o.
    // Both x_off and o_off are never negative.
    pub(crate) pips: [i8; 26],
    x_off: u8,
    o_off: u8,
}

impl Position {
    #[inline]
    pub(crate) fn x_off(&self) -> u8 {
        self.x_off
    }

    #[inline]
    pub(crate) fn o_off(&self) -> u8 {
        self.o_off
    }

    #[inline]
    /// Will return positive value for checkers of `x`, negative value for checkers of `o`.
    pub fn pip(&self, pip: usize) -> i8 {
        self.pips[pip]
    }

    #[inline]
    pub fn has_lost(&self) -> bool {
        self.o_off == NUM_OF_CHECKERS
    }

    #[inline]
    pub fn game_state(&self) -> GameState {
        debug_assert!(
            self.x_off < NUM_OF_CHECKERS || self.o_off < NUM_OF_CHECKERS,
            "Not both sides can win at the same time"
        );
        if self.x_off == NUM_OF_CHECKERS {
            if self.o_off > 0 {
                GameOver(WinNormal)
            } else if self.pips[O_BAR..7].iter().any(|pip| pip < &0) {
                GameOver(WinBg)
            } else {
                GameOver(WinGammon)
            }
        } else if self.o_off == NUM_OF_CHECKERS {
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

    /// Returns more info than `game_state` - not only whether the game is still ongoing, but also
    /// whether we are already in the race phase.
    ///
    /// This is important for choosing the proper neural net.
    #[inline]
    pub fn game_phase(&self) -> GamePhase {
        match self.game_state() {
            GameOver(result) => GamePhase::GameOver(result),
            Ongoing => {
                // The index of my checker which is closest to pip 24
                let last_own_checker = self
                    .pips
                    .iter()
                    .rposition(|&p| p > 0)
                    .expect("There must be a checker on a pip, otherwise the game is over");
                // The index of opponent's checker which is closest to 1
                let last_opponent_checker = self
                    .pips
                    .iter()
                    .position(|&p| p < 0)
                    .expect("There must be a checker on a pip, otherwise the game is over");
                if last_own_checker > last_opponent_checker {
                    GamePhase::Ongoing(Contact)
                } else {
                    GamePhase::Ongoing(Race)
                }
            }
        }
    }

    /// The return values have switched the sides of the players.
    pub fn all_positions_after_moving(&self, dice: &Dice) -> Vec<Position> {
        debug_assert!(self.o_off < NUM_OF_CHECKERS && self.x_off < NUM_OF_CHECKERS);
        let mut new_positions = match dice {
            Dice::Double(die) => self.all_positions_after_double_move(*die),
            Dice::Mixed(dice) => self.all_positions_after_mixed_move(dice),
        };
        for position in new_positions.iter_mut() {
            *position = position.sides_switched();
        }
        new_positions
    }

    #[inline]
    pub fn sides_switched(&self) -> Position {
        let mut pips = [0; 26];
        for (i, pip) in self.pips.iter().enumerate() {
            pips[25 - i] = -pip;
        }
        Position {
            x_off: self.o_off,
            o_off: self.x_off,
            pips,
        }
    }
}

impl From<Position> for [i8; 26] {
    fn from(value: Position) -> Self {
        value.pips
    }
}

impl TryFrom<[i8; 26]> for Position {
    type Error = &'static str;

    /// Use positive numbers for checkers of `x`. Use negative number for checkers of `o`.
    /// Index `25` is the bar for `x`, index `0` is the the bar for `o`.
    /// Checkers already off the board are calculated based on the input array.
    /// Will return an error if the sum of checkers for `x` or `o` is bigger than 15.
    fn try_from(pips: [i8; 26]) -> Result<Self, Self::Error> {
        let x_off: i8 =
            (NUM_OF_CHECKERS as i8) - pips.iter().filter(|p| p.is_positive()).sum::<i8>();
        let o_off: i8 =
            (NUM_OF_CHECKERS as i8) + pips.iter().filter(|p| p.is_negative()).sum::<i8>();

        if x_off < 0 {
            Err("Player x has more than 15 checkers on the board.")
        } else if o_off < 0 {
            Err("Player o has more than 15 checkers on the board.")
        } else if pips[X_BAR].is_negative() {
            Err("Index 25 is the bar for player x, number of checkers needs to be positive.")
        } else if pips[O_BAR].is_positive() {
            Err("Index 0 is the bar for player o, number of checkers needs to be negative.")
        } else {
            Ok(Position {
                pips,
                x_off: x_off as u8,
                o_off: o_off as u8,
            })
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
                // mixed move
                self.pips[from - die] += 1;
            }
        } else {
            // bear off
            self.x_off += 1;
        }
    }

    /// Only call if this move is legal.
    fn clone_and_move_single_checker(&self, from: usize, die: usize) -> Position {
        let mut new = *self;
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

    #[inline]
    fn can_move_internally(&self, from: usize, die: usize) -> bool {
        return if self.pips[from] < 1 {
            // no checker to move
            false
        } else if from > die {
            // mixed move, no bear off
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

    /// Tests whether we can move a checker from a certain pip.
    /// This method only does proper checks for non bearoff moves.
    /// It returns `true` for all possible bear offs.
    fn can_move_when_bearoff_is_legal(&self, from: usize, die: usize) -> bool {
        if self.pips[from] < 1 {
            // no checker to move
            false
        } else if from > die {
            // mixed move, no bear off
            let number_of_opposing_checkers = self.pips[from - die];
            number_of_opposing_checkers > -2
        } else {
            true
        }
    }

    /// When iterating over the pips to find checkers to move, we can ignore certain pips because
    /// moving from them is impossible.
    ///
    /// An example is: If there is a checker out of the home board, we can't bear off.
    /// So for example with a die of 4, the smallest pip to check is the 5.
    fn smallest_pip_to_check(&self, die: usize) -> usize {
        match self.pips.iter().rposition(|&p| p > 0) {
            None => X_BAR + 1, // no checkers on the board
            Some(biggest_checker) => {
                if biggest_checker > 6 {
                    // bear off is impossible
                    die + 1
                } else {
                    // bear off might be possible
                    min(biggest_checker, die)
                }
            }
        }
    }

    pub fn try_move_single_checker(&self, from: usize, die: usize) -> Option<Position> {
        if self.can_move(from, die) {
            Some(self.clone_and_move_single_checker(from, die))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::pos;
    use crate::position::*;

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
    fn game_state_bg_when_on_bar() {
        let given = pos!(x 25:1, 1:14; o);
        assert_eq!(given.game_state(), GameOver(LoseBg));
        assert_eq!(
            given.sides_switched().game_state(),
            GameOver(LoseBg.reverse())
        );
    }

    #[test]
    fn game_state_bg_when_not_on_bar() {
        let given = pos!(x 19:15; o);
        assert_eq!(given.game_state(), GameOver(LoseBg));
        assert_eq!(
            given.sides_switched().game_state(),
            GameOver(LoseBg.reverse())
        );
    }

    #[test]
    fn game_state_gammon() {
        let given = pos!(x 18:15; o);
        assert_eq!(given.game_state(), GameOver(LoseGammon));
        assert_eq!(
            given.sides_switched().game_state(),
            GameOver(LoseGammon.reverse())
        );
    }

    #[test]
    fn game_state_normal() {
        let given = pos!(x 19:14; o);
        assert_eq!(given.game_state(), GameOver(LoseNormal));
        assert!(given.has_lost());
        assert_eq!(
            given.sides_switched().game_state(),
            GameOver(LoseNormal.reverse())
        );
        assert!(!given.sides_switched().has_lost());
    }

    #[test]
    fn game_state_ongoing() {
        let given = pos!(x 19:14; o 1:4);
        assert_eq!(given.game_state(), Ongoing);
        assert_eq!(given.sides_switched().game_state(), Ongoing);
    }

    #[test]
    fn game_phase_win_or_lose_normal() {
        let given = pos!(x 1:1; o);
        assert_eq!(given.game_phase(), GamePhase::GameOver(LoseNormal));
        assert_eq!(
            given.sides_switched().game_phase(),
            GamePhase::GameOver(WinNormal)
        );
    }

    #[test]
    fn game_phase_win_or_lose_gammon() {
        let given = pos!(x 12:15; o);
        assert_eq!(given.game_phase(), GamePhase::GameOver(LoseGammon));
        assert_eq!(
            given.sides_switched().game_phase(),
            GamePhase::GameOver(WinGammon)
        );
    }

    #[test]
    fn game_phase_win_or_lose_bg() {
        let given = pos!(x 20:15; o);
        assert_eq!(given.game_phase(), GamePhase::GameOver(LoseBg));
        assert_eq!(
            given.sides_switched().game_phase(),
            GamePhase::GameOver(WinBg)
        );
    }

    #[test]
    fn game_phase_contact() {
        let given = pos!(x 12:1; o 2:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Contact));
    }

    #[test]
    fn game_phase_contact_enclosing() {
        let given = pos!(x 12:1; o 20:1, 2:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Contact));

        let given = pos!(x 20:1, 2:1; o 12:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Contact));
    }

    #[test]
    fn game_phase_contact_when_x_on_bar() {
        let given = pos!(x X_BAR:1; o 2:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Contact));
    }

    #[test]
    fn game_phase_contact_when_o_on_bar() {
        let given = pos!(x 1:1; o O_BAR:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Contact));
    }

    #[test]
    fn game_phase_race() {
        let given = pos!(x 1:1; o 2:1);
        assert_eq!(given.game_phase(), GamePhase::Ongoing(Race));
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
        assert_eq!(positions, [expected3, expected2, expected1]);
    }

    #[test]
    fn all_positions_after_moving_mixed() {
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
        let actual = original.sides_switched();
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
    fn try_from_legal() {
        // Given
        let mut pips = [0_i8; 26];
        pips[X_BAR] = 2;
        pips[10] = 10;
        pips[11] = -11;
        pips[O_BAR] = -3;
        // When
        let position = Position::try_from(pips);
        // Then
        let position = position.unwrap();
        assert_eq!(position.pip(X_BAR), 2);
        assert_eq!(position.pip(10), 10);
        assert_eq!(position.x_off, 3);
        assert_eq!(position.pip(11), -11);
        assert_eq!(position.pip(O_BAR), -3);
        assert_eq!(position.o_off, 1);
    }

    #[test]
    fn try_from_fails_too_many_x_checkers() {
        // Given
        let mut pips = [0_i8; 26];
        pips[X_BAR] = 10;
        pips[10] = 10;
        pips[11] = -10;
        // When
        let position = Position::try_from(pips);
        // Then
        assert_eq!(
            position,
            Err("Player x has more than 15 checkers on the board.")
        );
    }

    #[test]
    fn try_from_fails_too_many_o_checkers() {
        // Given
        let mut pips = [0_i8; 26];
        pips[10] = 10;
        pips[11] = -10;
        pips[O_BAR] = -10;
        // When
        let position = Position::try_from(pips);
        // Then
        assert_eq!(
            position,
            Err("Player o has more than 15 checkers on the board.")
        );
    }

    #[test]
    fn try_from_fails_o_checker_on_x_bar() {
        // Given
        let mut pips = [0_i8; 26];
        pips[X_BAR] = -10;
        pips[10] = 1;
        pips[11] = -1;
        // When
        let position = Position::try_from(pips);
        // Then
        assert_eq!(
            position,
            Err("Index 25 is the bar for player x, number of checkers needs to be positive.")
        );
    }

    #[test]
    fn try_from_fails_x_checker_on_o_bar() {
        // Given
        let mut pips = [0_i8; 26];
        pips[10] = 1;
        pips[11] = -1;
        pips[O_BAR] = 10;
        // When
        let position = Position::try_from(pips);
        // Then
        assert_eq!(
            position,
            Err("Index 0 is the bar for player o, number of checkers needs to be negative.")
        );
    }

    #[test]
    fn debug() {
        let actual = format!("{:?}", pos!(x X_BAR:2, 3:5, 1:1; o 24:7, 23:4, O_BAR:3),);
        let expected = "Position:\nx: {bar:2, 3:5, 1:1, off:7}\no: {off:1, 24:7, 23:4, bar:3}";
        assert_eq!(actual, expected);
    }

    #[test]
    fn number_of_moves_for_various_positions_and_dice() {
        // Thanks to Øystein for his test positions
        let positions = [
            ("4HPwATDgc/ABMA", (4, 4), 52),
            ("4HPwATDgc/ABMA", (3, 1), 16),
            ("4HPwATDgc/ABMA", (1, 3), 16),
            ("0HPwATDgc/ABMA", (6, 4), 15),
            ("0HPwATDgc/ABMA", (4, 6), 15),
            ("4DnyATDgc/ABMA", (6, 4), 14),
            ("4DnyATDgc/ABMA", (4, 6), 14),
            ("AACAkCRJqqoAAA", (1, 1), 2220),
            /* From The Bar */
            ("4HPwATDgc/ABUA", (6, 6), 0),
            ("4HPwATDgc/ABUA", (5, 6), 4),
            ("4HPwATDgc/ABUA", (5, 2), 7),
            ("0HPwATDgc/ABUA", (5, 2), 8),
            ("4HPwATDgc/ABYA", (5, 2), 1),
            ("sHPwATDgc/ABYA", (5, 2), 1),
            ("hnPwATDgc/ABYA", (5, 2), 1),
            ("sHPwATDgc/ABYA", (2, 2), 12),
            ("sHPwATDgOfgAcA", (2, 2), 4),
            ("sHPwATDgHHwAeA", (2, 2), 1),
            ("sHPwATDgHDwAfA", (2, 2), 1),
            ("sHPwATDgHDwAfA", (2, 1), 1),
            ("sHPwATDgHDwAfA", (6, 1), 1),
            ("xOfgATDgc/ABUA", (4, 3), 10),
            ("lOfgATDgc/ABUA", (4, 3), 10),
            /* Unable to play full roll */
            ("sNvBATBw38ABMA", (6, 6), 1),
            ("YNsWADZsuzsAAA", (6, 5), 1),
            ("YNsWADNm7zkAAA", (6, 5), 1),
            ("4BwcMBvgAYABAA", (4, 3), 1),
            ("4DgcMBvgAYABAA", (4, 3), 1),
            ("wAYAMBsAAAQAAA", (4, 3), 1),
            ("GBsAmA0EACAAAA", (4, 3), 2),
            ("MBsAsA0EACAAAA", (4, 3), 2),
            /* Bearoff */
            ("2G4bADDOAgAAAA", (5, 1), 2),
            ("2G4bADDObgAAAA", (4, 2), 7),
            ("AwAACAAAAAAAAA", (4, 2), 1),
            ("AwAAYDsAAAAAAA", (6, 5), 1),
            ("AwAAYDsAAAAAAA", (6, 2), 3),
            ("2+4OAADs3hcAAA", (4, 3), 12),
            ("tN0dAATb3AMAAA", (4, 2), 9),
            ("tN0dAATb3AMAAA", (2, 2), 38),
            ("2L07AAC274YAAA", (6, 5), 3),
            ("2L07AAC23wYBAA", (6, 5), 2),
            ("27ZFAAR7swEAAA", (6, 2), 4),
            ("27ZFAAR7swEAAA", (2, 6), 4),
            ("v0MChgK7HwgAAA", (5, 6), 1),
            ("u20DAAP77hEAAA", (6, 3), 3),
            ("u20DYAD77hEAAA", (6, 3), 3),
            ("ABDAEBIAAAAAAA", (6, 2), 1),
        ];
        fn number_of_moves(position: &Position, dice: &Dice) -> usize {
            let all = position.all_positions_after_moving(dice);
            if all.len() == 1 && all.first().unwrap().sides_switched() == *position {
                0
            } else {
                all.len()
            }
        }
        for (id, dice, number) in positions {
            let position = Position::from_id(id.to_string());
            let dice = Dice::new(dice.0, dice.1);
            assert_eq!(
                number_of_moves(&position, &dice),
                number,
                "failing position is {}",
                id
            );
        }
    }
}

#[cfg(test)]
mod private_tests {
    use crate::pos;
    use crate::position::{Position, O_BAR, STARTING};
    use std::collections::HashMap;

    #[test]
    fn starting_position_is_correct_and_symmetric() {
        let expected = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        assert_eq!(STARTING, expected);
        assert_eq!(STARTING, STARTING.sides_switched());
    }

    #[test]
    fn move_single_checker_mixed_move() {
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
    fn move_single_checker_bearoff_mixed() {
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
        let given = Position::from_hash_maps(&HashMap::new(), &HashMap::from([(4, 10)]));
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
