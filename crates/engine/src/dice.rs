/// Contains a legal pair of dice (values between 1 and 6).
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dice {
    Regular(RegularDice),
    Double(usize),
}

/// Contains two different values between 1 and six. `big` is bigger than `small`.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegularDice {
    pub(crate) big: usize,
    pub(crate) small: usize,
}

/// Contains all 441 possibilities of for two rolls of dice.
///
/// As `usize` is also returned how often those dice on average appear in 1296 rolls.
/// Two double rolls appear 1 time, two regular rolls appear 4 times.
/// A combination of a double and a regular roll appears 2 times.
pub const ALL_441: [([Dice; 2], usize); 441] = Dice::all_441();

impl Dice {
    #[inline(always)]
    pub const fn new(die1: usize, die2: usize) -> Self {
        if die1 == die2 {
            Dice::Double(die1)
        } else {
            Dice::Regular(RegularDice::new(die1, die2))
        }
    }

    /// Contains all 21 possibilities of dice and a number of how often they appear in 36 rolls.
    ///
    /// So doubles appear 1 time, regular rolls appear 2 times.
    const fn all_441() -> [([Dice; 2], usize); 441] {
        let mut dice_441 = [([Dice::Double(1), Dice::Double(1)], 0_usize); 441]; // Dummy values, will be replaced
        let dice_21 = Self::all_21();

        let mut i = 0_usize;
        while i < 21 {
            let mut j = 0_usize;
            while j < 21 {
                let a = dice_21[i];
                let b = dice_21[j];
                dice_441[i * 21 + j] = ([a.0, b.0], a.1 * b.1);
                j += 1;
            }
            i += 1;
        }
        dice_441
    }

    /// Contains all 21 possibilities of dice and a number of how often they appear in 36 rolls.
    ///
    /// So doubles appear 1 time, regular rolls appear 2 times.
    const fn all_21() -> [(Dice, usize); 21] {
        let mut dice = [(Dice::Double(1), 0_usize); 21]; // Dummy values, will be replaced

        // for loops don't work with `const fn`
        let mut i = 0_usize;
        let mut dice_index = 0_usize;
        while i < 6 {
            let mut j = i + 1;
            dice[dice_index] = (Dice::new(i + 1, i + 1), 1);
            dice_index += 1;
            while j < 6 {
                dice[dice_index] = (Dice::new(i + 1, j + 1), 2);
                j += 1;
                dice_index += 1;
            }
            i += 1;
        }
        dice
    }
}

impl TryFrom<(usize, usize)> for Dice {
    type Error = &'static str;

    fn try_from(value: (usize, usize)) -> Result<Self, Self::Error> {
        if value.0 < 1 || value.0 > 6 || value.1 < 1 || value.1 > 6 {
            Err("Dice values must be between 1 and 6.")
        } else {
            Ok(Dice::new(value.0, value.1))
        }
    }
}

impl RegularDice {
    #[inline(always)]
    pub fn small(&self) -> usize {
        self.small
    }

    #[inline(always)]
    pub fn big(&self) -> usize {
        self.big
    }
    #[inline(always)]
    pub const fn new(die1: usize, die2: usize) -> Self {
        let (big, small) = if die1 > die2 {
            (die1, die2)
        } else {
            (die2, die1)
        };
        Self { big, small }
    }
}

pub trait DiceGen {
    /// Returns dice
    fn roll(&mut self) -> Dice;
}

pub struct FastrandDice {
    generator: fastrand::Rng,
}

impl DiceGen for FastrandDice {
    /// Returns random dice
    fn roll(&mut self) -> Dice {
        let random = self.generator.usize(0..36);
        let die1 = random / 6 + 1;
        let die2 = random % 6 + 1;
        Dice::new(die1, die2)
    }
}

impl FastrandDice {
    #[allow(clippy::new_without_default)]
    pub fn new() -> FastrandDice {
        FastrandDice {
            generator: fastrand::Rng::new(),
        }
    }

    pub fn with_seed(seed: u64) -> FastrandDice {
        FastrandDice {
            generator: fastrand::Rng::with_seed(seed),
        }
    }

    /// Returns a seed which is depending on the state of `FastrandDice`.
    ///
    /// If `FastrandDice` was created with a certain seed and immediately this function is called,
    /// then it will always return the same value. Helpful for tests and benchmarks.
    pub fn seed(&mut self) -> u64 {
        self.generator.u64(..)
    }

    /// Returns a random seed, not depending on state. Don't use it in tests or benchmarks.
    pub fn random_seed() -> u64 {
        fastrand::u64(..)
    }
}

#[cfg(test)]
mod dice_tests {
    use crate::dice::Dice::{Double, Regular};
    use crate::dice::{Dice, ALL_441};
    use std::collections::HashSet;

    #[test]
    fn all_441() {
        let all_441 = ALL_441;
        let sum_of_numbers: usize = all_441.iter().map(|element| element.1).sum();
        assert_eq!(sum_of_numbers, 1296);
        for (dice, number) in all_441 {
            match (dice[0], dice[1]) {
                (Regular(_), Regular(_)) => assert_eq!(number, 4),
                (Double(_), Regular(_)) => assert_eq!(number, 2),
                (Regular(_), Double(_)) => assert_eq!(number, 2),
                (Double(_), Double(_)) => assert_eq!(number, 1),
            }
        }
    }

    #[test]
    fn all_441_has_distinct_dice() {
        let all_441 = ALL_441;
        let mut dice_set: HashSet<[Dice; 2]> = HashSet::new();
        for (dice, _) in all_441 {
            match dice[0] {
                Regular(dice) => assert!(dice.small > 0 && dice.big < 7),
                Double(die) => assert!(die > 0 && die < 7),
            }
            dice_set.insert(dice);
        }
        assert_eq!(dice_set.len(), 441);
    }
}

/// Use this for unit tests where you want to control the dice.
pub struct DiceGenMock {
    dice: Vec<Dice>,
    no_calls: usize, // how often `roll` has been called.
}

impl DiceGen for DiceGenMock {
    fn roll(&mut self) -> Dice {
        let dice = self.dice[self.no_calls];
        self.no_calls += 1;
        dice
    }
}

impl DiceGenMock {
    pub fn new(dice: &[Dice]) -> DiceGenMock {
        DiceGenMock {
            dice: dice.to_vec(),
            no_calls: 0,
        }
    }

    pub fn assert_all_dice_were_used(&self) {
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

    #[test]
    fn initialized_with_identical_seeds_results_in_identical_behavior() {
        let mut dice_gen1 = FastrandDice::with_seed(100);
        let mut dice_gen2 = FastrandDice::with_seed(100);

        assert_eq!(dice_gen1.seed(), dice_gen2.seed());
        assert_eq!(dice_gen1.roll(), dice_gen2.roll());
    }

    #[test]
    fn initialized_with_different_seeds_results_in_different_behavior() {
        let mut dice_gen1 = FastrandDice::with_seed(100);
        let mut dice_gen2 = FastrandDice::with_seed(123);

        // Well, as there are only 21 dice, sometimes we will see the same behavior. But mostly not.
        assert_ne!(dice_gen1.seed(), dice_gen2.seed());
        assert_ne!(dice_gen1.roll(), dice_gen2.roll());
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
