use crate::position::{NUM_OF_CHECKERS, O_BAR, Position, X_BAR};
use base64::Engine;
use base64::engine::general_purpose;
use std::collections::HashMap;

/// Simple way to create positions for testing.
///
/// The starting position would be:
/// pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2)
/// The order is not important, so this is equivalent:
/// pos!(x 24:2, 13:5, 8:3, 6:5; o 1:2, 12:5, 17:3, 19:5)
#[macro_export]
macro_rules! pos {
    ( x $( $x_pip:tt:$x_checkers:tt ), * ;o $( $o_pip:tt:$o_checkers:tt ), * ) => {
        {
            #[allow(unused_mut)]
            let mut x = std::collections::HashMap::new();
            $(
                x.insert($x_pip as usize, $x_checkers as u8);
            )*

            #[allow(unused_mut)]
            let mut o = std::collections::HashMap::new();
            $(
                o.insert($o_pip as usize, $o_checkers as u8);
            )*

            $crate::position::Position::from_hash_maps(&x, &o)
        }
    };
}

/// GnuBG Position ID.
/// Details: https://www.gnu.org/software/gnubg/manual/html_node/A-technical-description-of-the-Position-ID.html
impl Position {
    pub fn position_id(&self) -> String {
        let key = self.encode();
        general_purpose::STANDARD_NO_PAD.encode(key)
    }

    pub fn from_id(id: &str) -> Position {
        let mut bytes = [0; 10];
        general_purpose::STANDARD_NO_PAD
            .decode_slice(id, &mut bytes)
            .expect("Position encoding expects valid id.");
        Position::decode(bytes)
    }

    fn encode(&self) -> [u8; 10] {
        let mut key = [0u8; 10];
        let mut bit_index = 0;

        // Encoding the position for the player not on roll
        for point in (O_BAR..X_BAR).rev() {
            for _ in 0..-self.pips[point] {
                key[bit_index / 8] |= 1 << (bit_index % 8);
                bit_index += 1; // Appending a 1
            }
            bit_index += 1; // Appending a 0
        }

        // Encoding the position for the player on roll
        (O_BAR + 1..X_BAR + 1).for_each(|point| {
            (0..self.pips[point]).for_each(|_| {
                key[bit_index / 8] |= 1 << (bit_index % 8);
                bit_index += 1; // Appending a 1
            });
            bit_index += 1; // Appending a 0
        });

        key
    }

    fn decode(key: [u8; 10]) -> Position {
        let mut bit_index = 0;
        let mut pips = [0i8; 26];

        let mut x_pieces = 0;
        let mut o_pieces = 0;

        (O_BAR..X_BAR).rev().for_each(|point| {
            while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
                pips[point] -= 1;
                o_pieces += 1;
                bit_index += 1;
            }
            bit_index += 1; // Appending a 0
        });

        (O_BAR + 1..X_BAR + 1).for_each(|point| {
            while (key[bit_index / 8] >> (bit_index % 8)) & 1 == 1 {
                pips[point] += 1;
                x_pieces += 1;
                bit_index += 1;
            }
            bit_index += 1; // Appending a 0
        });

        Position {
            pips,
            x_off: NUM_OF_CHECKERS - x_pieces,
            o_off: NUM_OF_CHECKERS - o_pieces,
        }
    }

    pub fn from_hash_maps(x: &HashMap<usize, u8>, o: &HashMap<usize, u8>) -> Position {
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
}

#[cfg(test)]
mod tests {
    use crate::position::{O_BAR, Position, STARTING, X_BAR};

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
            let game = Position::from_id(pid);
            assert_eq!(pid, game.position_id());
        }
    }

    #[test]
    fn matching_positions() {
        let pos1 = pos!(x 24:1, X_BAR:2; o 1:3, O_BAR: 4);
        let pos2 = pos!(x 2:10, 1:5; o 24:9, 23:6);
        for position in [pos1, pos2] {
            let id = position.position_id();
            assert_eq!(position, Position::from_id(&id));
        }
    }
}
