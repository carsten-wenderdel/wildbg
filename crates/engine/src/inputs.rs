use crate::position::{Position, X_BAR};

pub trait InputsGen {
    /// The number of inputs for the neural network.
    const NUM_INPUTS: usize;

    /// Fill the given vector with the neural net inputs for a single position.
    /// The returned vector is the same as the given vector, the length should have been
    /// increased by `NUM_INPUTS`.
    ///
    /// This is the only method that needs to be implemented.
    fn fill_inputs(&self, pos: &Position, vec: &mut Vec<f32>);

    /// The neural net inputs for a single position.
    ///
    /// The length of the returned vector matches `NUM_INPUTS`.
    fn inputs_for_single(&self, pos: &Position) -> Vec<f32> {
        self.inputs_for_all(&[*pos])
    }

    /// A single vector with neural net inputs for all positions. This is useful for batch evaluation.
    ///
    /// The length of the returned vector is `NUM_INPUTS * positions.len()`.
    fn inputs_for_all(&self, positions: &[Position]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(Self::NUM_INPUTS * positions.len());
        positions
            .iter()
            .for_each(|pos| self.fill_inputs(pos, &mut vec));
        vec
    }
}

/// Inputs for one pip. One entry for every legal number of checkers (-15 to 15).
///
/// Custom format, probably same as GnuBG
/// For ideas see https://stackoverflow.com/questions/32428237/board-encoding-in-tesauros-td-gammon
const TD_INPUTS: [[f32; 4]; 31] = [
    [0.; 4], // opponent checkers (-15)
    [0.; 4], // opponent checkers (-14)
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],
    [0.; 4],          // opponent checker (-1)
    [0.; 4],          // no checker
    [1., 0., 0., 0.], // own checker (1)
    [0., 1., 0., 0.], // own checkers (2)
    [0., 0., 1., 0.],
    [0., 0., 1., 1.],
    [0., 0., 1., 2.],
    [0., 0., 1., 3.],
    [0., 0., 1., 4.],
    [0., 0., 1., 5.],
    [0., 0., 1., 6.],
    [0., 0., 1., 7.],
    [0., 0., 1., 8.],
    [0., 0., 1., 9.],
    [0., 0., 1., 10.],
    [0., 0., 1., 11.],
    [0., 0., 1., 12.], // own checkers (15)
];

/// 4 inputs representing a single pip from the point of view of one player.
#[inline]
fn td_inputs(number_of_checkers: isize) -> &'static [f32; 4] {
    // We need to add `15` to `pip` to make sure that the index is non negative.
    let array_index = (number_of_checkers + 15) as usize;
    // Using a lookup table for the 31 different cases (-15 to 15) is much faster than for example a match statement.
    TD_INPUTS
        .get(array_index)
        .expect("number of pips needs to be between -15 and 15")
}

pub struct ContactInputsGen {}

impl InputsGen for ContactInputsGen {
    const NUM_INPUTS: usize = 202;

    fn fill_inputs(&self, pos: &Position, vec: &mut Vec<f32>) {
        vec.push(pos.x_off() as f32);
        vec.push(pos.o_off() as f32);

        // The inputs for the own player `x`
        // In an earlier implementation we messed up the order of the inputs
        // If one day there will be more inputs, streamline the next three lines:
        vec.extend_from_slice(td_inputs(pos.pips[X_BAR] as isize));
        pos.pips[1..X_BAR].iter().for_each(|p| {
            vec.extend_from_slice(td_inputs(*p as isize));
        });

        // The inputs for the opponent `o`.
        pos.pips[0..X_BAR].iter().for_each(|p| {
            vec.extend_from_slice(td_inputs(-(*p as isize)));
        });
    }
}

pub struct RaceInputsGen {}

impl InputsGen for RaceInputsGen {
    const NUM_INPUTS: usize = 186;

    fn fill_inputs(&self, pos: &Position, vec: &mut Vec<f32>) {
        vec.push(pos.x_off() as f32);
        vec.push(pos.o_off() as f32);

        // The inputs for the own player `x`. No checkers on bar or on 24 during race.
        pos.pips[1..24].iter().for_each(|p| {
            vec.extend_from_slice(td_inputs(*p as isize));
        });

        // The inputs for the opponent `o`. No checkers on bar or on 1 during race.
        pos.pips[2..X_BAR].iter().for_each(|p| {
            vec.extend_from_slice(td_inputs(-(*p as isize)));
        });
    }
}

#[cfg(test)]
mod input_tests {
    use crate::inputs::td_inputs;

    #[test]
    fn test_td_inputs() {
        for pip in -15..16 {
            let inputs = td_inputs(pip);

            // Check input one
            if pip == 1 {
                assert_eq!(inputs[0], 1.0);
            } else {
                assert_eq!(inputs[0], 0.0);
            }

            // Check input two
            if pip == 2 {
                assert_eq!(inputs[1], 1.0);
            } else {
                assert_eq!(inputs[1], 0.0);
            }

            // Check input three
            if pip > 2 {
                assert_eq!(inputs[2], 1.0);
            } else {
                assert_eq!(inputs[2], 0.0);
            }

            // Check input four
            if pip > 2 {
                assert_eq!(inputs[3], pip as f32 - 3.0);
            } else {
                assert_eq!(inputs[3], 0.0);
            }
        }
    }
}

#[cfg(test)]
mod contact_tests {
    use crate::inputs::{ContactInputsGen, InputsGen};
    use crate::pos;
    use crate::position::O_BAR;

    #[test]
    fn contact_cvs_line() {
        let pos = pos!(x 1:1, 2:2, 3:3, 4:4, 5:5; o 24:1, O_BAR: 1);
        let inputs_gen = ContactInputsGen {};

        let inputs = inputs_gen
            .inputs_for_single(&pos)
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let inputs_switched = inputs_gen
            .inputs_for_single(&pos.switch_sides())
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");

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

    #[test]
    fn race_cvs_line() {
        let pos = pos!(x 1:1, 2:2, 3:3, 4:4, 5:5; o 24:1);
        let inputs_gen = RaceInputsGen {};

        let inputs = inputs_gen
            .inputs_for_single(&pos)
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let inputs_switched = inputs_gen
            .inputs_for_single(&pos.switch_sides())
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");

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
