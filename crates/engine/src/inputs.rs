use crate::position::{Position, X_BAR};

pub trait InputsGen {
    /// The number of inputs for the neural network.
    const NUM_INPUTS: usize;

    /// The inputs for the neural network.
    ///
    /// The length of the vector matches `num_inputs`.
    fn input_vec(&self, pos: &Position) -> Vec<f32>;

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

/// 4 inputs representing a single pip from the point of view of one player.
///
/// Custom format, probably same as GnuBG
/// For ideas see https://stackoverflow.com/questions/32428237/board-encoding-in-tesauros-td-gammon
#[inline(always)]
fn td_inputs(pip: &i8) -> [f32; 4] {
    match pip {
        1 => [1.0, 0.0, 0.0, 0.0],
        2 => [0.0, 1.0, 0.0, 0.0],
        &p if p > 0 => [0.0, 0.0, 1.0, p as f32 - 3.0],
        _ => [0.0; 4], // both for no checker and for opponent's checker
    }
}

pub struct ContactInputsGen {}

impl InputsGen for ContactInputsGen {
    const NUM_INPUTS: usize = 202;

    fn input_vec(&self, pos: &Position) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::with_capacity(Self::NUM_INPUTS);
        vec.push(pos.x_off() as f32);
        vec.push(pos.o_off() as f32);

        // The inputs for the own player `x`
        // In an earlier implementation we messed up the order of the inputs
        // If one day there will be more inputs, streamline the next three lines:
        // X_BAR
        vec.extend_from_slice(&td_inputs(&pos.pips[X_BAR]));
        for td_inputs in pos.pips[1..X_BAR].iter().map(td_inputs) {
            vec.extend_from_slice(&td_inputs);
        }

        // The inputs for the opponent `o`.
        for td_inputs in pos.pips[0..X_BAR].iter().map(|p| td_inputs(&-p)) {
            vec.extend_from_slice(&td_inputs);
        }
        vec
    }
}

pub struct RaceInputsGen {}

impl InputsGen for RaceInputsGen {
    const NUM_INPUTS: usize = 186;

    fn input_vec(&self, pos: &Position) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::with_capacity(Self::NUM_INPUTS);
        vec.push(pos.x_off() as f32);
        vec.push(pos.o_off() as f32);

        // The inputs for the own player `x`. No checkers on bar or on 24 during race.
        for td_inputs in pos.pips[1..24].iter().map(td_inputs) {
            vec.extend_from_slice(&td_inputs);
        }

        // The inputs for the opponent `o`. No checkers on bar or on 1 during race.
        for td_inputs in pos.pips[2..X_BAR].iter().map(|p| td_inputs(&-p)) {
            vec.extend_from_slice(&td_inputs);
        }
        vec
    }
}

#[cfg(test)]
mod contact_tests {
    use crate::inputs::{ContactInputsGen, InputsGen};
    use crate::pos;
    use crate::position::{Position, O_BAR};
    use std::collections::HashMap;

    #[test]
    fn contact_cvs_line() {
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
}

#[cfg(test)]
mod race_tests {
    use crate::inputs::{InputsGen, RaceInputsGen};
    use crate::pos;
    use crate::position::Position;
    use std::collections::HashMap;

    #[test]
    fn race_cvs_line() {
        let pos = pos!(x 1:1, 2:2, 3:3, 4:4, 5:5; o 24:1);
        let pos_switched = pos.switch_sides();
        let inputs_gen = RaceInputsGen {};
        let inputs = inputs_gen.csv_line(&pos);
        let inputs_switched = inputs_gen.csv_line(&pos_switched);
        assert_eq!(
            inputs,
            "0;14;1;0;0;0;0;1;0;0;0;0;1;0;0;0;1;1;0;0;1;2;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0"
        );
        assert_eq!(
            inputs_switched,
            "14;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;2;0;0;1;1;0;0;1;0;0;1;0;0;1;0;0;0"
        );
    }
}
