use crate::dice::Dice;

/// Implements all (pseudo) randomness that happens in wildbg. Example use case: rollouts.
///
/// Not only dice rolls are implemented also other randomness for the crate `coach`.
pub trait DiceGen {
    /// Returns pseudo random dice
    fn roll(&mut self) -> Dice;

    /// Returns different dice, no double roll. This is useful for the first move of the game.
    fn roll_mixed(&mut self) -> Dice {
        // There are more efficient ways to do that, but it's rarely used.
        // So let's keep it simple.
        // Also this works for all implementing types.
        loop {
            let dice = self.roll();
            if let Dice::Mixed(_) = dice {
                return dice;
            }
        }
    }

    /// Returns a number at least zero but smaller than the length of array `choices`.
    ///
    /// The chance that `i` is returned is proportional to the number in `choices[i]`.
    /// Example: When choices is [1.0, 2.0, 2.0], then in 20% of all cases '0` is returned and in
    fn choose_index(&mut self, chances: &[f32]) -> usize;
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

    fn choose_index(&mut self, chances: &[f32]) -> usize {
        // This is all no very exact because of rounding errors, but good enough for our use case
        let big_number: u32 = 1_000_000;
        let random_number = self.generator.u32(0..big_number);
        // `choice` will be random number between 0 and the sum of `choices`.
        let threshold: f32 =
            (chances.iter().sum::<f32>() / big_number as f32) * random_number as f32;
        let mut sum: f32 = 0.0;
        for (index, &value) in chances.iter().enumerate() {
            sum += value;
            if sum > threshold {
                return index;
            }
        }
        chances.len() - 1
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

    fn choose_index(&mut self, _chances: &[f32]) -> usize {
        0
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
    use crate::dice::Dice;
    use crate::dice_gen::{DiceGen, FastrandDice};
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
                Dice::Mixed(dice) => {
                    count[dice.big - 1][dice.small - 1] += 1;
                }
            }
        }

        // Check mixed rolls
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

    #[test]
    fn choose() {
        let mut counts = [0; 3];
        let mut dice_gen = FastrandDice::with_seed(123);
        let choices = vec![0.2, 0.5, 0.3];
        for _ in 0..100_000 {
            counts[dice_gen.choose_index(&choices)] += 1;
        }
        assert!(counts[0] > 19_500 && counts[0] < 20_500);
        assert!(counts[1] > 49_500 && counts[1] < 50_500);
        assert!(counts[2] > 29_500 && counts[2] < 30_500);
    }
}

#[cfg(test)]
mod dice_gen_mock_tests {
    use crate::dice::Dice;
    use crate::dice_gen::{DiceGen, DiceGenMock};

    #[test]
    fn roll_returns_given_dice() {
        let mut dice_gen = DiceGenMock::new(&[Dice::new(3, 2), Dice::new(1, 6)]);
        assert_eq!(dice_gen.roll(), Dice::new(3, 2));
        assert_eq!(dice_gen.roll(), Dice::new(1, 6));
    }

    #[test]
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
