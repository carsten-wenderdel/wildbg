pub struct DiceGen {
    generator: fastrand::Rng,
}

impl DiceGen {
    #[allow(dead_code)]
    pub fn new() -> DiceGen {
        DiceGen {
            generator: fastrand::Rng::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_seed(seed: u64) -> DiceGen {
        DiceGen {
            generator: fastrand::Rng::with_seed(seed),
        }
    }

    /// Returns two random dice with values each between 1 and 6. Not sorted by size.
    #[allow(dead_code)]
    pub fn roll(&mut self) -> (usize, usize) {
        let random = self.generator.usize(0..36);
        let die1 = random / 6 + 1;
        let die2 = random % 6 + 1;
        (die1, die2)
    }
}

#[cfg(test)]
mod tests {
    use crate::dice_gen::DiceGen;

    #[test]
    fn all_numbers_are_occurring() {
        // Given
        let mut gen = DiceGen::with_seed(123);
        let mut count1 = [0, 0, 0, 0, 0, 0];
        let mut count2 = count1.clone();
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
