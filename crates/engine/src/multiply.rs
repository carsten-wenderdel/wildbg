use crate::dice::ALL_21;
use crate::evaluator::Evaluator;
use crate::position::Position;
use crate::probabilities::Probabilities;

/// Looks one more ply ahead
///
/// It calculates for all 21 possible dice rolls the best response of the opponent and then takes
/// the average of the resulting evaluations (reversed of course to see it from 'x' side again).
///
/// Currently all possible moves are evaluated.
/// In the future we could speed things up by not evaluating really bad moves multi-ply.
pub struct MultiPlyEvaluator<T: Evaluator> {
    pub evaluator: T,
}

impl<T: Evaluator> Evaluator for MultiPlyEvaluator<T> {
    fn eval(&self, position: &Position) -> Probabilities {
        let mut win_normal = 0f32;
        let mut win_gammon = 0f32;
        let mut win_bg = 0f32;
        let mut lose_normal = 0f32;
        let mut lose_gammon = 0f32;
        let mut lose_bg = 0f32;
        ALL_21
            .iter()
            .map(|(dice, number)| {
                let probabilities = self
                    .evaluator
                    .positions_and_probabilities_by_equity(position, dice)
                    .first()
                    .unwrap()
                    .1
                    .clone();
                (probabilities, *number as f32)
            })
            .for_each(|(probabilities, number)| {
                win_normal += probabilities.win_normal * number;
                win_gammon += probabilities.win_gammon * number;
                win_bg += probabilities.win_bg * number;
                lose_normal += probabilities.lose_normal * number;
                lose_gammon += probabilities.lose_gammon * number;
                lose_bg += probabilities.lose_bg * number;
            });
        Probabilities {
            win_normal: win_normal / 36f32,
            win_gammon: win_gammon / 36f32,
            win_bg: win_bg / 36f32,
            lose_normal: lose_normal / 36f32,
            lose_gammon: lose_gammon / 36f32,
            lose_bg: lose_bg / 36f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::composite::CompositeEvaluator;
    use crate::evaluator::Evaluator;
    use crate::multiply::MultiPlyEvaluator;
    use crate::pos;

    #[test]
    fn equity_is_average_of_1ply_ahead_equities() {
        let evaluator = CompositeEvaluator::default_tests();

        // From this position `x` will always win, unless the roll is 2-1, 3-1 or 1-1.
        let position = pos!(x 5:1; o 24:1);

        // That's the position after the roll 2-1. We switch sides to see it from the perspective of `o`.
        let loser_position_1 = pos!(x 2:1; o 24:1).sides_switched();
        let loser_equity_1 = evaluator.eval(&loser_position_1).equity();

        // That's the position after the roll 3-1 or 1-1. We switch sides to see it from the perspective of `o`.
        let loser_position_2 = pos!(x 1:1; o 24:1).sides_switched();
        let loser_equity_2 = evaluator.eval(&loser_position_2).equity();

        let multi = MultiPlyEvaluator { evaluator };

        let probabilities_multi = multi.eval(&position);
        let multi_equity = probabilities_multi.equity();
        // In 31 of 36 cases the equity is 1.0 because we win. In the other cases the reversed loser_equity is taken.
        let expected_equity = ((-2.0 * loser_equity_1 - 3.0 * loser_equity_2) + 31.0) / 36.0;

        assert!((multi_equity - expected_equity).abs() < 0.0000001);
    }
}
