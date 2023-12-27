use crate::position::{Position, NUM_OF_CHECKERS, O_BAR, X_BAR};
use base64::engine::general_purpose;
use base64::Engine;
use std::collections::HashMap;
use std::ops::Add;

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

/// GnuBG Position ID
impl Position {
    pub fn position_id(&self) -> String {
        let key = self.encode();
        let b64 = general_purpose::STANDARD.encode(key);
        b64[..14].to_string()
    }

    pub fn from_id(id: String) -> Position {
        let key = general_purpose::STANDARD.decode(id.add("==")).unwrap();
        Position::decode(key.try_into().unwrap())
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
    use crate::position::STARTING;

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
}
