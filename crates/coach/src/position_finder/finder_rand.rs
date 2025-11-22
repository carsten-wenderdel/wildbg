pub(super) trait FinderRandomizer {
    /// Returns a number at least zero but smaller than the length of array `choices`.
    ///
    /// The chance that `i` is returned is proportional to the number in `choices[i]`.
    /// Example: When choices is [1.0, 2.0, 2.0], then in 20% of all cases '0` is returned.
    fn sample(&mut self, chances: &[f32]) -> usize;
}

pub(super) struct FinderRand {
    rng: fastrand::Rng,
}

impl FinderRand {
    pub(super) fn with_seed(seed: u64) -> Self {
        FinderRand {
            rng: fastrand::Rng::with_seed(seed),
        }
    }
}

impl FinderRandomizer for FinderRand {
    fn sample(&mut self, chances: &[f32]) -> usize {
        // This is all no very exact because of rounding errors, but good enough for our use case
        let mut threshold = chances.iter().sum::<f32>() * self.rng.f32();
        for (index, &value) in chances.iter().enumerate() {
            if value >= threshold {
                return index;
            }
            threshold -= value;
        }
        chances.len() - 1
    }
}
