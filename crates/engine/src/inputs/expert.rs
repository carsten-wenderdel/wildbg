use crate::position::Position;

/// Expert features, for later usage as inputs for the neural nets.
///
/// The current neural nets don't use expert features yet.
pub trait ExpertInputs {
    fn pip_count(&self) -> f32;
}

impl ExpertInputs for Position {
    fn pip_count(&self) -> f32 {
        self.pips
            .iter()
            .enumerate()
            .filter(|&(_, p)| *p >= 0)
            // The biggest pip count would be 15 * 25 = 375. That's too big for u8, but i16 works fine.
            // Using a small data type is good for SIMD, it makes the multiplications faster.
            .map(|(i, &pip)| i as i16 * pip as i16)
            .sum::<i16>() as f32
    }
}

#[cfg(test)]
mod tests {
    use crate::inputs::expert::ExpertInputs;
    use crate::pos;
    use crate::position::{STARTING, X_BAR};

    #[test]
    fn pip_count_starting_pos() {
        assert_eq!(STARTING.pip_count(), 167.);
    }

    #[test]
    fn pip_count_some_pips_on_bar() {
        assert_eq!(pos![x 1:13, 24:1; o 10:1].pip_count(), 37.);
    }

    #[test]
    fn pip_count_with_all_pips_on_bar() {
        // We have to switch sides because the macro can't handle all of x checkers being on the bar.
        assert_eq!(pos![x X_BAR:15; o 10:15].pip_count(), 375.);
    }
}
