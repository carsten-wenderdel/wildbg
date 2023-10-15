use crate::position::{Position, X_BAR};

/// Custom format, for ideas see https://stackoverflow.com/questions/32428237/board-encoding-in-tesauros-td-gammon
pub trait InputsGen {
    /// The number of inputs for the neural network.
    fn num_inputs(&self) -> usize;

    /// The inputs for the neural network.
    ///
    /// The length of the vector matches `num_inputs`.
    fn input_vec(&self, pos: &Position) -> Vec<f32>;

    /// The header for CSV files with training data.
    ///
    /// As delimiter is `;` used.
    /// The number of elements matches `num_inputs`;
    fn csv_header(&self) -> String;

    /// A line with outputs for the neural network.
    ///
    /// As delimiter is `;` used.
    /// The elements are floating point numbers converted to strings.
    /// The number of elements matches `num_inputs`;
    fn csv_line(&self, pos: &Position) -> String {
        self.input_vec(pos)
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(";")
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

pub struct ContactInputsGen {}

impl InputsGen for ContactInputsGen {
    fn num_inputs(&self) -> usize {
        202
    }

    fn csv_header(&self) -> String {
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

    fn input_vec(&self, pos: &Position) -> Vec<f32> {
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

        let mut vec: Vec<f32> = Vec::with_capacity(self.num_inputs());
        vec.push(pos.x_off() as f32);
        vec.push(pos.o_off() as f32);
        for input in x_inputs {
            vec.push(input.p1 as f32);
            vec.push(input.p2 as f32);
            vec.push(input.p3 as f32);
            vec.push(input.p4 as f32);
        }
        for input in o_inputs {
            vec.push(input.p1 as f32);
            vec.push(input.p2 as f32);
            vec.push(input.p3 as f32);
            vec.push(input.p4 as f32);
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use crate::inputs::{ContactInputsGen, InputsGen};
    use crate::pos;
    use crate::position::{Position, O_BAR};
    use std::collections::HashMap;

    #[test]
    fn inputs_display() {
        let pos = pos!(x 1:1, 2:2, 3:3, 4:4, 5:5; o 24:1, O_BAR: 1);
        let pos_switched = pos.switch_sides();
        let inputs_gen = ContactInputsGen {};
        let inputs = inputs_gen.csv_line(&pos);
        let inputs_switched = inputs_gen.csv_line(&pos_switched);
        assert_eq!(
            inputs,
            "0;13;0;0;0;0;1;0;0;0;0;1;0;0;0;0;1;0;0;0;1;1;0;0;1;2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0"
        );
        assert_eq!(
            inputs_switched,
            "13;0;1;0;0;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;2;0;0;1;1;0;0;1;0;0;1;0;0;1;0;0;0"
        );
    }

    #[test]
    fn header_has_same_number_of_columns_as_inputs() {
        let pos = pos!(x 1:1; o 2:2);
        let inputs_gen = ContactInputsGen {};
        let inputs = inputs_gen.csv_line(&pos);
        let inputs_semicolons = inputs.matches(';').count();

        let header = inputs_gen.csv_header();
        let header_semicolons = header.matches(';').count();

        assert_eq!(inputs_semicolons, inputs_gen.num_inputs() - 1);
        assert_eq!(header_semicolons, inputs_gen.num_inputs() - 1);
    }

    #[test]
    fn no_empty_column_in_header() {
        assert_eq!(ContactInputsGen {}.csv_header().matches(";;").count(), 0)
    }
}
