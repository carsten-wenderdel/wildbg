use engine::dice_gen::DiceGen;
use engine::evaluator::Evaluator;
use engine::position::GameState::{GameOver, Ongoing};
use engine::position::STARTING;
use engine::probabilities::ResultCounter;

pub struct Duel<T: Evaluator, U: Evaluator> {
    evaluator1: T,
    evaluator2: U,
}

/// Let two `Evaluator`s duel each other. A bit quick and dirty.
impl<T: Evaluator, U: Evaluator> Duel<T, U> {
    #[allow(clippy::new_without_default)]
    pub fn new(evaluator1: T, evaluator2: U) -> Self {
        Self {
            evaluator1,
            evaluator2,
        }
    }

    /// The two `Evaluator`s will play twice each against each other.
    /// Either `Evaluator` will start once and play with the same dice as vice versa.
    pub fn duel<V: DiceGen>(&self, dice_gen: &mut V) -> ResultCounter {
        let mut pos1 = STARTING;
        let mut pos2 = STARTING;
        let mut iteration = 0;
        let mut pos1_finished = false;
        let mut pos2_finished = false;
        let mut counter = ResultCounter::default();
        while !(pos1_finished && pos2_finished) {
            let dice = dice_gen.roll();

            match pos1.game_state() {
                Ongoing => {
                    pos1 = if iteration % 2 == 0 {
                        self.evaluator1.best_position_by_equity(&pos1, &dice)
                    } else {
                        self.evaluator2.best_position_by_equity(&pos1, &dice)
                    };
                }
                GameOver(result) => {
                    if !pos1_finished {
                        pos1_finished = true;
                        let result = if iteration % 2 == 0 {
                            result
                        } else {
                            result.reverse()
                        };
                        counter.add(result);
                    }
                }
            }
            match pos2.game_state() {
                Ongoing => {
                    pos2 = if iteration % 2 == 0 {
                        self.evaluator2.best_position_by_equity(&pos2, &dice)
                    } else {
                        self.evaluator1.best_position_by_equity(&pos2, &dice)
                    };
                }
                GameOver(result) => {
                    if !pos2_finished {
                        pos2_finished = true;
                        let result = if iteration % 2 == 0 {
                            result.reverse()
                        } else {
                            result
                        };
                        counter.add(result);
                    }
                }
            }
            iteration += 1;
        }
        debug_assert!(counter.sum() == 2, "Each duel should have two game results");
        counter
    }
}
