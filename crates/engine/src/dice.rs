/// Contains a legal pair of dice (values between 1 and 6).
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dice {
    Mixed(MixedDice),
    Double(usize),
}

/// Contains two different values between 1 and six. `big` is bigger than `small`.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct MixedDice {
    pub(crate) big: usize,
    pub(crate) small: usize,
}

/// Contains all 441 possibilities of for two rolls of dice.
///
/// As `usize` is also returned how often those dice on average appear in 1296 rolls.
/// Two double rolls appear 1 time, two mixed rolls appear 4 times.
/// A combination of a double and a mixed roll appears 2 times.
pub const ALL_441: [([Dice; 2], usize); 441] = Dice::all_441();

/// Contains all 21 possibilities of dice and a number of how often they appear in 36 rolls.
///
/// So doubles appear 1 time, mixed rolls appear 2 times.
pub const ALL_21: [(Dice, usize); 21] = Dice::all_21();

impl Dice {
    #[inline]
    pub const fn new(die1: usize, die2: usize) -> Self {
        debug_assert!(die1 > 0);
        debug_assert!(die1 < 7);
        debug_assert!(die2 > 0);
        debug_assert!(die2 < 7);
        if die1 == die2 {
            Dice::Double(die1)
        } else {
            Dice::Mixed(MixedDice::new(die1, die2))
        }
    }
    const fn all_441() -> [([Dice; 2], usize); 441] {
        let mut dice_441 = [([Dice::Double(1), Dice::Double(1)], 0_usize); 441]; // Dummy values, will be replaced
        let dice_21 = ALL_21;

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

    /// All 6 double rolls. Used in tests.
    pub const fn all_6_double() -> [Dice; 6] {
        let mut dice = [Dice::Double(1); 6];
        let mut i = 0_usize;
        while i < 6 {
            dice[i] = Dice::new(i + 1, i + 1);
            i += 1;
        }
        dice
    }

    /// All 15 rolls with mixed dice (no doubles). Used in tests.
    pub const fn all_15_mixed() -> [Dice; 15] {
        let mut dice = [Dice::Double(1); 15]; // Dummy values, will be replaced

        // for loops don't work with `const fn`
        let mut i = 0_usize;
        let mut dice_index = 0_usize;
        while i < 6 {
            let mut j = i + 1;
            while j < 6 {
                dice[dice_index] = Dice::new(i + 1, j + 1);
                j += 1;
                dice_index += 1;
            }
            i += 1;
        }
        dice
    }

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

impl MixedDice {
    #[inline]
    pub fn small(&self) -> usize {
        self.small
    }

    #[inline]
    pub fn big(&self) -> usize {
        self.big
    }
    #[inline]
    pub const fn new(die1: usize, die2: usize) -> Self {
        let (big, small) = if die1 > die2 {
            (die1, die2)
        } else {
            (die2, die1)
        };
        Self { big, small }
    }
}

#[cfg(test)]
mod dice_tests {
    use crate::dice::Dice::{Double, Mixed};
    use crate::dice::{Dice, ALL_441};
    use std::collections::HashSet;

    #[test]
    fn all_441() {
        let all_441 = ALL_441;
        let sum_of_numbers: usize = all_441.iter().map(|element| element.1).sum();
        assert_eq!(sum_of_numbers, 1296);
        for (dice, number) in all_441 {
            match (dice[0], dice[1]) {
                (Mixed(_), Mixed(_)) => assert_eq!(number, 4),
                (Double(_), Mixed(_)) => assert_eq!(number, 2),
                (Mixed(_), Double(_)) => assert_eq!(number, 2),
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
                Mixed(dice) => assert!(dice.small > 0 && dice.big < 7),
                Double(die) => assert!(die > 0 && die < 7),
            }
            dice_set.insert(dice);
        }
        assert_eq!(dice_set.len(), 441);
    }
}
