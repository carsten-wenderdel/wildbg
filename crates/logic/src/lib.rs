use crate::bg_move::BgMove;
use engine::complex::ComplexEvaluator;
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::position::Position;

pub mod bg_move;
pub mod cube;

type Error = &'static str;

pub fn best_move_1ptr(pips: [i8; 26], die1: u8, die2: u8) -> Result<BgMove, Error> {
    match ComplexEvaluator::try_default() {
        None => Err("Could not find neural networks. We expect 'contact.onnx' and 'race.onnx' in the folder 'neural-nets'."),
        Some(evaluator) => best_move_1ptr_with_evaluator(pips, die1, die2, &evaluator),
    }
}

fn best_move_1ptr_with_evaluator<T: Evaluator>(
    pips: [i8; 26],
    die1: u8,
    die2: u8,
    evaluator: &T,
) -> Result<BgMove, Error> {
    let position = Position::try_from(pips)?;
    let dice = Dice::try_from((die1 as usize, die2 as usize))?;
    // gammon and backgammon counts the same as normal wins, so we use p.win()
    let new_position = evaluator.best_position(&position, &dice, |p| p.win());
    let bg_move = BgMove::new(&position, &new_position.switch_sides(), &dice);
    Ok(bg_move)
}

#[cfg(test)]
mod tests {
    use crate::bg_move::{BgMove, MoveDetail};
    use engine::evaluator::Evaluator;
    use engine::pos;
    use engine::position::Position;
    use engine::probabilities::Probabilities;
    use std::collections::HashMap;

    fn position_with_lowest_equity() -> Position {
        pos!(x 5:1, 3:1; o 20:2).switch_sides()
    }

    /// Test double. Returns not so good probabilities for `expected_pos`, better for everything else.
    struct EvaluatorFake {}
    impl Evaluator for EvaluatorFake {
        fn eval(&self, pos: &Position) -> Probabilities {
            if pos == &position_with_lowest_equity() {
                // This would be position for money game.
                // Remember that this equity is already from the point of the opponent.
                Probabilities {
                    win_normal: 0.5,
                    win_gammon: 0.1,
                    win_bg: 0.1,
                    lose_normal: 0.1,
                    lose_gammon: 0.1,
                    lose_bg: 0.1,
                }
            } else {
                // This would be position for 1 ptrs.
                Probabilities {
                    win_normal: 0.38,
                    win_gammon: 0.2,
                    win_bg: 0.1,
                    lose_normal: 0.12,
                    lose_gammon: 0.1,
                    lose_bg: 0.1,
                }
            }
        }
    }

    #[test]
    fn best_move_1ptr_with_evaluator() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = EvaluatorFake {};
        // When
        let bg_move =
            crate::best_move_1ptr_with_evaluator(given_pos.clone().into(), 4, 2, &evaluator)
                .unwrap();
        // Then
        let expected_move = BgMove {
            details: vec![MoveDetail { from: 7, to: 5 }, MoveDetail { from: 5, to: 1 }],
        };
        assert_eq!(bg_move, expected_move);
    }

    #[test]
    fn best_move_1ptr_error() {
        let given_pos = pos!(x 7:2; o 20:2);
        assert_eq!(
            crate::best_move_1ptr(given_pos.into(), 4, 2).expect_err(
                "During tests folders are handled differently than when using a binary crate."
            ),
            "Could not find neural networks. We expect 'contact.onnx' and 'race.onnx' in the folder 'neural-nets'."
        );
    }
}
