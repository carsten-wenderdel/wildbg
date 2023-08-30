#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Dice {
    Regular(RegularDice),
    Double(usize),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RegularDice {
    pub(crate) big: usize,
    pub(crate) small: usize,
}

pub(crate) const ALL_36: [Dice; 36] = Dice::all_36();
pub(crate) const ALL_1296: [(Dice, Dice); 1296] = Dice::all_1296();

impl Dice {
    #[inline(always)]
    pub const fn new(die1: usize, die2: usize) -> Self {
        if die1 == die2 {
            Dice::Double(die1)
        } else {
            Dice::Regular(RegularDice::new(die1, die2))
        }
    }

    /// Contains all 36 possibilities of dice. Regular dice will appear twice.
    const fn all_36() -> [Dice; 36] {
        let mut dice = [Dice::Double(1); 36]; // Dummy values, will be replaced

        // for loops don't work with `const fn`
        let mut i = 0_usize;
        while i < 6 {
            let mut j = 0_usize;
            while j < 6 {
                dice[i * 6 + j] = Dice::new(i + 1, j + 1);
                j += 1;
            }
            i += 1;
        }
        dice
    }

    /// Contains all 1296 possibilities of the first two rolls. Regular dice will appear multiple times.
    const fn all_1296() -> [(Dice, Dice); 1296] {
        let dummy_value = (Dice::Double(1), Dice::Double(1));
        let mut dice = [dummy_value; 1296];
        let all_36 = ALL_36;

        // for loops don't work with `const fn`
        let array_length = 36;
        let mut i = 0_usize;
        while i < array_length {
            let mut j = 0_usize;
            while j < array_length {
                dice[i * array_length + j] = (all_36[i], all_36[j]);
                j += 1;
            }
            i += 1;
        }
        dice
    }
}

impl RegularDice {
    #[inline(always)]
    pub(crate) const fn new(die1: usize, die2: usize) -> Self {
        let (big, small) = if die1 > die2 {
            (die1, die2)
        } else {
            (die2, die1)
        };
        Self { big, small }
    }
}

pub(crate) trait DiceGen {
    /// Returns two random dice with values each between 1 and 6. Not sorted by size.
    fn roll(&mut self) -> Dice;
}

pub(crate) struct FastrandDice {
    generator: fastrand::Rng,
}

impl DiceGen for FastrandDice {
    /// Returns two random dice with values each between 1 and 6. Not sorted by size.
    fn roll(&mut self) -> Dice {
        let random = self.generator.usize(0..36);
        let die1 = random / 6 + 1;
        let die2 = random % 6 + 1;
        Dice::new(die1, die2)
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
mod dice_tests {
    use crate::dice::Dice;

    #[test]
    fn all_36() {
        let all_36 = Dice::all_36();

        let smallest_double = Dice::new(1, 1);
        assert_eq!(all_36.iter().filter(|x| x == &&smallest_double).count(), 1);

        let biggest_double = Dice::new(6, 6);
        assert_eq!(all_36.iter().filter(|x| x == &&biggest_double).count(), 1);

        let regular = Dice::new(1, 6);
        assert_eq!(all_36.iter().filter(|x| x == &&regular).count(), 2);
    }
    #[test]
    fn all_1296() {
        let all_1296 = Dice::all_1296();

        let double_double = (Dice::new(1, 1), Dice::new(6, 6));
        assert_eq!(all_1296.iter().filter(|x| x == &&double_double).count(), 1);

        let double_regular = (Dice::new(2, 2), Dice::new(5, 4));
        assert_eq!(all_1296.iter().filter(|x| x == &&double_regular).count(), 2);

        let regular_double = (Dice::new(2, 3), Dice::new(5, 5));
        assert_eq!(all_1296.iter().filter(|x| x == &&regular_double).count(), 2);

        let regular_regular = (Dice::new(2, 3), Dice::new(3, 5));
        assert_eq!(
            all_1296.iter().filter(|x| x == &&regular_regular).count(),
            4
        );
    }
}

#[cfg(test)]
/// Use this for unit tests where you want to control the dice.
pub(crate) struct DiceGenMock {
    dice: Vec<Dice>,
    no_calls: usize, // how often `roll` has been called.
}

#[cfg(test)]
impl DiceGen for DiceGenMock {
    fn roll(&mut self) -> Dice {
        let dice = self.dice[self.no_calls];
        self.no_calls += 1;
        dice
    }
}

#[cfg(test)]
impl DiceGenMock {
    pub(crate) fn new(dice: &[Dice]) -> DiceGenMock {
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
    use crate::dice::{Dice, DiceGen, FastrandDice};
    use std::cmp::Ordering;

    #[test]
    fn all_numbers_are_occurring() {
        // Given
        let mut gen = FastrandDice::with_seed(123);
        let mut count = [[0_u32; 6]; 6];
        // When
        for _ in 0..360_000 {
            let dice = gen.roll();
            match dice {
                Dice::Double(die) => {
                    count[die - 1][die - 1] += 1;
                }
                Dice::Regular(dice) => {
                    count[dice.big - 1][dice.small - 1] += 1;
                }
            }
        }

        // Check regular rolls
        for i in 0..6 {
            for j in 0..6 {
                let count = count[i][j];
                match i.cmp(&j) {
                    Ordering::Less => {
                        // This makes sure that dice.big is not smaller than dice.small
                        assert_eq!(count, 0);
                    }
                    Ordering::Equal => {
                        assert!(count > 9_000 && count < 11_000);
                    }
                    Ordering::Greater => {
                        assert!(count > 18_000 && count < 22_000);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dice_gen_mock_tests {
    use crate::dice::{Dice, DiceGen, DiceGenMock};

    #[test]
    // #[should_panic]
    fn roll_returns_given_dice() {
        let mut dice_gen = DiceGenMock::new(&[Dice::new(3, 2), Dice::new(1, 6)]);
        assert_eq!(dice_gen.roll(), Dice::new(3, 2));
        assert_eq!(dice_gen.roll(), Dice::new(1, 6));
    }

    #[test]
    // #[should_panic]
    #[should_panic(expected = "Not all dice of the mock have been used")]
    fn assert_all_dice_were_used() {
        let mut dice_gen = DiceGenMock::new(&[Dice::new(3, 2), Dice::new(1, 6)]);
        dice_gen.roll();
        dice_gen.assert_all_dice_were_used();
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 1 but the index is 1")]
    fn panics_when_roll_is_called_too_often() {
        let mut dice_gen = DiceGenMock::new(&[Dice::new(3, 2)]);
        dice_gen.roll();
        dice_gen.roll();
    }
}
