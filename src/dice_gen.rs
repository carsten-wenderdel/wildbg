pub(crate) trait DiceGen {
    /// Returns two random dice with values each between 1 and 6. Not sorted by size.
    fn roll(&mut self) -> (usize, usize);
}

pub(crate) struct FastrandDice {
    generator: fastrand::Rng,
}

impl DiceGen for FastrandDice {
    /// Returns two random dice with values each between 1 and 6. Not sorted by size.
    fn roll(&mut self) -> (usize, usize) {
        let random = self.generator.usize(0..36);
        let die1 = random / 6 + 1;
        let die2 = random % 6 + 1;
        (die1, die2)
    }
}

impl FastrandDice {
    #[allow(dead_code)]
    pub fn new() -> FastrandDice {
        FastrandDice {
            generator: fastrand::Rng::new(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_seed(seed: u64) -> FastrandDice {
        FastrandDice {
            generator: fastrand::Rng::with_seed(seed),
        }
    }
}

#[cfg(test)]
/// Use this for unit tests where you want to control the dice.
pub(crate) struct DiceGenMock {
    dice: Vec<(usize, usize)>,
    no_calls: usize, // how often `roll` has been called.
}

#[cfg(test)]
impl DiceGen for DiceGenMock {
    fn roll(&mut self) -> (usize, usize) {
        let dice = self.dice[self.no_calls];
        self.no_calls += 1;
        dice
    }
}

#[cfg(test)]
impl DiceGenMock {
    pub(crate) fn new(dice: &[(usize, usize)]) -> DiceGenMock {
        DiceGenMock {
            dice: dice.to_vec(),
            no_calls: 0,
        }
    }

    pub(crate) fn assert_all_dice_were_used(&self) {
        assert_eq!(
            self.dice.len(),
            self.no_calls,
            "Not all dice of the mock have been used"
        );
    }
}

#[cfg(test)]
mod fastrand_dice_tests {
    use crate::dice_gen::{DiceGen, FastrandDice};

    #[test]
    fn all_numbers_are_occurring() {
        // Given
        let mut gen = FastrandDice::with_seed(123);
        let mut count1 = [0, 0, 0, 0, 0, 0];
        let mut count2 = count1;
        // When
        for _ in 0..100_000 {
            let (die1, die2) = gen.roll();
            count1[die1 - 1] += 1;
            count2[die2 - 1] += 1;
        }
        // Then
        for i in 0..6 {
            assert!(count1[i] > 16_300);
            assert!(count2[i] > 16_300);
            assert!(count1[i] < 17_000);
            assert!(count2[i] < 17_000);
        }
    }
}

#[cfg(test)]
mod dice_gen_mock_tests {
    use crate::dice_gen::{DiceGen, DiceGenMock};

    #[test]
    // #[should_panic]
    fn roll_returns_given_dice() {
        let mut dice_gen = DiceGenMock::new(&[(3, 2), (1, 6)]);
        assert_eq!(dice_gen.roll(), (3, 2));
        assert_eq!(dice_gen.roll(), (1, 6));
    }

    #[test]
    // #[should_panic]
    #[should_panic(expected = "Not all dice of the mock have been used")]
    fn assert_all_dice_were_used() {
        let mut dice_gen = DiceGenMock::new(&[(3, 2), (1, 6)]);
        dice_gen.roll();
        dice_gen.assert_all_dice_were_used();
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 1 but the index is 1")]
    fn panics_when_roll_is_called_too_often() {
        let mut dice_gen = DiceGenMock::new(&[(3, 2)]);
        dice_gen.roll();
        dice_gen.roll();
    }
}
