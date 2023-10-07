mod double_moves;
mod regular_moves;
use base64::{engine::general_purpose, Engine as _};

use crate::dice::Dice;
use crate::position::GameResult::*;
use crate::position::GameState::*;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Write;
use std::ops::Add;

const NUM_OF_CHECKERS: u8 = 15;
pub const X_BAR: usize = 25;
pub const O_BAR: usize = 0;

pub const STARTING: Position = Position {
    pips: [
        0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
    ],
    x_off: 0,
    o_off: 0,
};

#[derive(Debug, PartialEq)]
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
    pub fn pip(&self, pip: usize) -> i8 {
        debug_assert!((1..=25).contains(&pip));
        self.pips[pip]
    }

    pub fn has_lost(&self) -> bool {
        self.o_off == NUM_OF_CHECKERS
    }

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

    /// The return values have switched the sides of the players.
    pub fn all_positions_after_moving(&self, dice: &Dice) -> Vec<Position> {
        debug_assert!(self.o_off < NUM_OF_CHECKERS && self.x_off < NUM_OF_CHECKERS);
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
        Position::try_from(pips).expect("Need legal position")
    }

    pub fn position_id(&self) -> String {
        let key = self.encode();
        let b64 = general_purpose::STANDARD.encode(key);
        b64[..14].to_string()
    }

    pub fn from_id(id: String) -> Position {
        let key = general_purpose::STANDARD.decode(id.add("==")).unwrap();
        Position::decode(key.try_into().unwrap())
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

    pub fn try_move_single_checker(&self, from: usize, die: usize) -> Option<Position> {
        if self.can_move(from, die) {
            Some(self.clone_and_move_single_checker(from, die))
        } else {
            None
        }
    }

    fn encode(&self) -> [u8; 10] {
        let mut key = [0u8; 10];
        let mut bit_index = 0;

        // Encoding the position for the player not on roll
        for point in (1..=24).rev() {
            for _ in 0..-self.pips[point] {
                key[bit_index / 8] |= 1 << (bit_index % 8);
                bit_index += 1; // Appending a 1
            }
            bit_index += 1; // Appending a 0
        }
        for _ in 0..self.pips[O_BAR] {
            key[bit_index / 8] |= 1 << (bit_index % 8);
            bit_index += 1; // Appending a 1
        }
        bit_index += 1; // Appending a 0

        // Encoding the position for the player on roll
        for point in 1..=24 {
            for _ in 0..self.pips[point] {
                key[bit_index / 8] |= 1 << (bit_index % 8);
                bit_index += 1; // Appending a 1
            }
            bit_index += 1; // Appending a 0
        }
        for _ in 0..self.pips[X_BAR] {
            key[bit_index / 8] |= 1 << (bit_index % 8);
            bit_index += 1; // Appending a 1
        }

        key
    }

    fn decode(key: [u8; 10]) -> Position {
        let mut bit_index = 0;
        let mut pips = [0i8; 26];

        let mut x_bar = 0;
        let mut o_bar = 0;
        let mut x_pieces = 0;
        let mut o_pieces = 0;

        for point in (0..24).rev() {
            while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
                pips[point + 1] -= 1;
                o_pieces += 1;
                bit_index += 1;
            }
            bit_index += 1; // Appending a 0
        }

        while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
            o_bar += 1;
            bit_index += 1;
        }

        bit_index += 1; // Appending a 0

        for point in 0..24 {
            while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
                pips[point + 1] += 1;
                x_pieces += 1;
                bit_index += 1;
            }
            bit_index += 1; // Appending a 0
        }

        while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
            x_bar += 1;
            bit_index += 1;
        }

        pips[X_BAR] = x_bar;
        pips[O_BAR] = -o_bar;

        Position {
            pips,
            x_off: (NUM_OF_CHECKERS as i8 - x_pieces - x_bar) as u8,
            o_off: (NUM_OF_CHECKERS as i8 - o_pieces - o_bar) as u8,
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
        assert!(given.has_lost());
        assert_eq!(
            given.switch_sides().game_state(),
            GameOver(LoseNormal.reverse())
        );
        assert!(!given.switch_sides().has_lost());
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
        assert_eq!(position.x_bar(), 2);
        assert_eq!(position.pip(10), 10);
        assert_eq!(position.x_off, 3);
        assert_eq!(position.pip(11), -11);
        assert_eq!(position.o_bar(), 3);
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
    fn start_id() {
        let game = STARTING;
        let id = game.position_id();
        assert_eq!(id, "4HPwATDgc/ABMA");
    }

    #[test]
    fn matching_ids() {
        let pids = [
            "4HPwATDgc/ABMA", // starting position
            "jGfkASjg8wcBMA", // random position
            "zGbiIQgxH/AAWA", // X bar
            "zGbiIYCYD3gALA", // O off
        ];
        for pid in pids {
            let game = super::Position::from_id(pid.to_string());
            assert_eq!(pid, game.position_id());
        }
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
            if all.len() == 1 && all.first().unwrap().switch_sides() == *position {
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
