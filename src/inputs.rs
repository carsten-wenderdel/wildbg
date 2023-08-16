use crate::position::{Position, X_BAR};
use std::fmt;

pub(crate) const NUM_INPUTS: usize = 202;

/// Custom format, for ideas see https://stackoverflow.com/questions/32428237/board-encoding-in-tesauros-td-gammon
pub struct Inputs {
    x_inputs: [PipInput; 25],
    o_inputs: [PipInput; 25],
    x_off: u8,
    o_off: u8,
}

/// Used when writing CSV data to a file
impl fmt::Display for Inputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{};{}", self.x_off, self.o_off).unwrap();
        for input in &self.x_inputs {
            write!(f, ";{};{};{};{}", input.p1, input.p2, input.p3, input.p4).unwrap();
        }
        for input in &self.o_inputs {
            write!(f, ";{};{};{};{}", input.p1, input.p2, input.p3, input.p4).unwrap();
        }
        Ok(())
    }
}

impl Inputs {
    pub fn to_vec(&self) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::with_capacity(NUM_INPUTS);
        vec.push(self.x_off as f32);
        vec.push(self.o_off as f32);
        for input in &self.x_inputs {
            vec.push(input.p1 as f32);
            vec.push(input.p2 as f32);
            vec.push(input.p3 as f32);
            vec.push(input.p4 as f32);
        }
        for input in &self.o_inputs {
            vec.push(input.p1 as f32);
            vec.push(input.p2 as f32);
            vec.push(input.p3 as f32);
            vec.push(input.p4 as f32);
        }
        vec
    }

    pub fn csv_header() -> String {
        let mut string = String::new();
        string.push_str("x_off;o_off;x_bar-1;x_bar-2;x_bar-3;x_bar-4");
        for pip in 1..25 {
            for case in 1..5 {
                string = string + ";x" + &pip.to_string() + "-" + &case.to_string();
            }
        }
        string += ";o_bar-1;o_bar-2;o_bar-3;o_bar-4";
        for pip in 1..25 {
            for case in 1..5 {
                string = string + ";o" + &pip.to_string() + "-" + &case.to_string();
            }
        }
        string
    }

    pub fn from_position(pos: &Position) -> Self {
        let mut x_inputs = [NO_CHECKERS; 25];
        let mut o_inputs = [NO_CHECKERS; 25];
        // on the bar:
        x_inputs[0] = PipInput::from_pip(pos.x_bar());
        o_inputs[0] = PipInput::from_pip(pos.o_bar());
        // on the board:
        for i in 1..X_BAR {
            let pip = pos.pip(i);
            #[allow(clippy::comparison_chain)]
            if pip > 0 {
                x_inputs[i] = PipInput::from_pip(pip as u8);
            } else if pip < 0 {
                o_inputs[i] = PipInput::from_pip(-pip as u8);
            }
        }
        Inputs {
            x_inputs,
            o_inputs,
            x_off: pos.x_off(),
            o_off: pos.o_off(),
        }
    }
}

struct PipInput {
    p1: u8,
    p2: u8,
    p3: u8,
    p4: u8,
}

const NO_CHECKERS: PipInput = PipInput {
    p1: 0,
    p2: 0,
    p3: 0,
    p4: 0,
};

impl PipInput {
    fn from_pip(pip: u8) -> Self {
        match pip {
            0 => NO_CHECKERS,
            1 => Self {
                p1: 1,
                p2: 0,
                p3: 0,
                p4: 0,
            },
            2 => Self {
                p1: 0,
                p2: 1,
                p3: 0,
                p4: 0,
            },
            p => Self {
                p1: 0,
                p2: 0,
                p3: 1,
                p4: p - 3,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::inputs::{Inputs, NUM_INPUTS};
    use crate::pos;
    use crate::position::{Position, O_BAR};
    use std::collections::HashMap;

    #[test]
    fn inputs_display() {
        let pos = pos!(x 1:1, 2:2, 3:3, 4:4, 5:5; o 24:1, O_BAR: 1);
        let pos_switched = pos.switch_sides();
        let inputs = Inputs::from_position(&pos);
        let inputs_switched = Inputs::from_position(&pos_switched);
        assert_eq!(
            inputs.to_string(),
            "0;13;0;0;0;0;1;0;0;0;0;1;0;0;0;0;1;0;0;0;1;1;0;0;1;2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0"
        );
        assert_eq!(
            inputs_switched.to_string(),
            "13;0;1;0;0;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;2;0;0;1;1;0;0;1;0;0;1;0;0;1;0;0;0"
        );
    }

    #[test]
    fn header_has_same_number_of_columns_as_inputs() {
        let pos = pos!(x 1:1; o 2:2);
        let inputs = Inputs::from_position(&pos);
        let inputs_semicolons = inputs.to_string().matches(';').count();

        let header = Inputs::csv_header();
        let header_semicolons = header.matches(';').count();

        assert_eq!(inputs_semicolons, NUM_INPUTS - 1);
        assert_eq!(header_semicolons, NUM_INPUTS - 1);
    }

    #[test]
    fn no_empty_column_in_header() {
        assert_eq!(Inputs::csv_header().matches(";;").count(), 0)
    }
}
