use engine::position::OngoingPhase::Race;
use engine::position::{GamePhase, Position};
use engine::probabilities::Probabilities;

pub(super) trait FinderRandomizer {
    /// Returns a number at least zero but smaller than the length of array `choices`.
    ///
    /// The chance that `i` is returned is proportional to the number in `choices[i]`.
    /// Example: When choices is [1.0, 2.0, 2.0], then in 20% of all cases '0` is returned.
    fn sample(&mut self, chances: &[f32]) -> usize;

    fn next_position(
        &mut self,
        positions_and_probabilities: &[(Position, Probabilities)],
    ) -> Position {
        let best = positions_and_probabilities
            .first()
            .expect("move generator must always return a move");
        // In some cases we definitely want to return the best move
        match best.0.game_phase() {
            GamePhase::GameOver(_) => return best.0,
            GamePhase::Ongoing(ongoing_state) => {
                if ongoing_state == Race {
                    // For races we always want to take the best move - it's not worth it to explore
                    // different moves here; for example we can't reach a backgame from a race.
                    // Instead we would only roll out strange positions later on.
                    return best.0;
                }
            }
        }
        // Ok, now we are in contact game; sometimes we want return another move than the best one.
        let best_equity = best.1.equity();
        let chances: Vec<f32> = positions_and_probabilities
            .iter()
            .filter_map(|(_, probability)| {
                let equity_loss = best_equity - probability.equity();
                // Let's ignore really bad moves, but if the equity loss is small we sometimes
                // want to use them.
                if equity_loss < 0.05 {
                    // So for the best move the chance is 1.0.
                    // For a move with an equity loss of 0.03 the chance is 0.55
                    Some(1.0 - 15.0 * equity_loss)
                } else {
                    // `positions_and_probabilities` is sorted by equity.
                    // So we give a chance for the first entries in there and filter everything
                    // after that away.
                    None
                }
            })
            .collect();
        let choice = self.sample(&chances);
        positions_and_probabilities
            .get(choice)
            .expect("choose_index must return index smaller than the number of moves")
            .0
    }
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

#[cfg(test)]
mod tests {
    use crate::position_finder::finder_rand::FinderRandomizer;
    use engine::pos;
    use engine::probabilities::{Probabilities, ResultCounter};

    struct RandMock {}

    impl FinderRandomizer for RandMock {
        /// A bit quick and dirty - the values and asserts are hard coded for two tests at once: "contact()" and "race()"
        fn sample(&mut self, chances: &[f32]) -> usize {
            // This method should only be called in the test "contact". In race, this method should
            // not be called as the best position should be returned.
            assert_eq!(chances.len(), 2);
            assert_eq!(chances[0], 1.0);
            assert!((chances[1] - 0.85).abs() < 0.000001);
            1
        }
    }

    #[test]
    fn contact() {
        let pos_1 = pos!(x 20:1; o 1:1);
        let pos_2 = pos!(x 20:1; o 2:1);
        let pos_3 = pos!(x 20:1; o 3:1);

        let prob_1 = Probabilities::from(&ResultCounter::new(80, 20, 0, 0, 0, 0));
        let prob_2 = Probabilities::from(&ResultCounter::new(81, 19, 0, 0, 0, 0));
        let prob_3 = Probabilities::from(&ResultCounter::new(86, 14, 0, 0, 0, 0));

        // Not part of the actual test, just to make sure that we test later is done properly
        assert!((prob_1.equity() - prob_2.equity() - 0.01).abs() < 0.0000001);
        assert!((prob_1.equity() - prob_3.equity() - 0.06).abs() < 0.0000001);

        let mut rand = RandMock {};

        // Given
        let input = vec![(pos_1, prob_1), (pos_2, prob_2), (pos_3, prob_3)];
        // When
        let found = rand.next_position(&input);
        // Then
        assert_eq!(
            found, pos_2,
            "Second best move should be returned as specified in RandMock."
        );
    }

    #[test]
    fn race() {
        let pos_1 = pos!(x 1:1; o 10:1);
        let pos_2 = pos!(x 2:1; o 10:1);
        let pos_3 = pos!(x 3:1; o 10:1);

        let prob_1 = Probabilities::from(&ResultCounter::new(80, 20, 0, 0, 0, 0));
        let prob_2 = Probabilities::from(&ResultCounter::new(81, 19, 0, 0, 0, 0));
        let prob_3 = Probabilities::from(&ResultCounter::new(86, 14, 0, 0, 0, 0));

        // Not part of the actual test, just to make sure that we test later is done properly
        assert!((prob_1.equity() - prob_2.equity() - 0.01).abs() < 0.0000001);
        assert!((prob_1.equity() - prob_3.equity() - 0.06).abs() < 0.0000001);

        let mut rand = RandMock {};

        // Given
        let input = vec![(pos_1, prob_1), (pos_2, prob_2), (pos_3, prob_3)];
        // When
        let found = rand.next_position(&input);
        // Then
        assert_eq!(found, pos_1, "Best move should be returned");
    }
}
