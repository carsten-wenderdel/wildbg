use engine::inputs::InputsGen;
use engine::position::Position;
use engine::probabilities::Probabilities;
use serde::{Deserialize, Serialize};

/// Position ID and 5 probabilities meant to be serialized to CSV to keep training data for a longer time.
///
/// We don't use the 6 probabilities format to be more compatible with other backgammon programs.
/// `win` includes the chance to win gammon or backgammon.
/// `win_g` and `lose_g` include the chance to win or lose backgammon.
#[derive(Debug, Deserialize, Serialize)]
pub struct PositionRecord {
    position_id: String,
    win: f32,
    win_g: f32,
    win_bg: f32,
    lose_g: f32,
    lose_bg: f32,
}

impl PositionRecord {
    pub fn new(position: &Position, probabilities: &Probabilities) -> Self {
        // We only round if the values are the result of an addition.
        PositionRecord {
            position_id: position.position_id(),
            win: round_if_near_0_or_1(
                probabilities.win_normal + probabilities.win_gammon + probabilities.win_bg,
            ),
            win_g: round_if_near_0_or_1(probabilities.win_gammon + probabilities.win_bg),
            win_bg: probabilities.win_bg,
            lose_g: round_if_near_0_or_1(probabilities.lose_gammon + probabilities.lose_bg),
            lose_bg: probabilities.lose_bg,
        }
    }

    pub fn csv_header() -> Vec<String> {
        vec![
            "position_id".to_owned(),
            "win".to_owned(),
            "win_g".to_owned(),
            "win_bg".to_owned(),
            "lose_g".to_owned(),
            "lose_bg".to_owned(),
        ]
    }
}

#[derive(Debug, Serialize)]
pub struct InputsRecord {
    win_normal: f32,
    win_gammon: f32,
    win_bg: f32,
    lose_normal: f32,
    lose_gammon: f32,
    lose_bg: f32,
    inputs: Vec<f32>,
}

impl InputsRecord {
    pub fn new<T: InputsGen>(record: &PositionRecord, inputs_gen: &T) -> Self {
        let position = Position::from_id(&record.position_id);
        let mut record = InputsRecord {
            win_normal: round_if_near_0_or_1(record.win - record.win_g),
            win_gammon: round_if_near_0_or_1(record.win_g - record.win_bg),
            win_bg: round_if_near_0_or_1(record.win_bg),
            lose_normal: round_if_near_0_or_1(1.0 - record.win - record.lose_g),
            lose_gammon: round_if_near_0_or_1(record.lose_g - record.lose_bg),
            lose_bg: round_if_near_0_or_1(record.lose_bg),
            inputs: inputs_gen.inputs_for_single(&position),
        };
        // The sum should be 1.0, but because of floating point inaccuracies, it might be a bit off.
        let sum = record.win_normal
            + record.win_gammon
            + record.win_bg
            + record.lose_normal
            + record.lose_gammon
            + record.lose_bg;
        if sum != 1.0 {
            record.win_normal /= sum;
            record.win_gammon /= sum;
            record.win_bg /= sum;
            record.lose_normal /= sum;
            record.lose_gammon /= sum;
            record.lose_bg /= sum;
        }
        record
    }
}

/// Because of floating point arithmetic we might end up with numbers which should be 0 or 1,
/// but are slightly off. Let's fix it this way:
fn round_if_near_0_or_1(probability: f32) -> f32 {
    let margin = 0.00001;
    if probability.abs() < margin {
        0.0
    } else if (1.0 - probability).abs() < margin {
        1.0
    } else {
        probability
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::inputs::ContactInputsGen;

    #[test]
    fn test_inputs_record_probabilities_sum_to_one() {
        let record = PositionRecord {
            position_id: "4HPwATDgc/ABMA".to_string(), // starting position
            win: 0.6,
            win_g: 0.2,
            win_bg: 0.05,
            lose_g: 0.1,
            lose_bg: 0.02,
        };
        let inputs_gen = ContactInputsGen {};
        let inputs_record = InputsRecord::new(&record, &inputs_gen);

        let sum = inputs_record.win_normal
            + inputs_record.win_gammon
            + inputs_record.win_bg
            + inputs_record.lose_normal
            + inputs_record.lose_gammon
            + inputs_record.lose_bg;

        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_inputs_record_correct_probabilities() {
        let record = PositionRecord {
            position_id: "4HPwATDgc/ABMA".to_string(), // starting position
            win: 0.6,
            win_g: 0.2,
            win_bg: 0.05,
            lose_g: 0.1,
            lose_bg: 0.02,
        };
        let inputs_gen = ContactInputsGen {};
        let inputs_record = InputsRecord::new(&record, &inputs_gen);

        // win_normal = win - win_g = 0.6 - 0.2 = 0.4
        assert!((inputs_record.win_normal - 0.4).abs() < 1e-6);

        // win_gammon = win_g - win_bg = 0.2 - 0.05 = 0.15
        assert!((inputs_record.win_gammon - 0.15).abs() < 1e-6);

        // win_bg = 0.05
        assert!((inputs_record.win_bg - 0.05).abs() < 1e-6);

        // lose_normal = 1.0 - win - lose_g = 1.0 - 0.6 - 0.1 = 0.3
        assert!((inputs_record.lose_normal - 0.3).abs() < 1e-6);

        // lose_gammon = lose_g - lose_bg = 0.1 - 0.02 = 0.08
        assert!((inputs_record.lose_gammon - 0.08).abs() < 1e-6);

        // lose_bg = 0.02
        assert!((inputs_record.lose_bg - 0.02).abs() < 1e-6);
    }
}
