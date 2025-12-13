use crate::bg_move::BgMove;
use crate::cube::CubeInfo;
use engine::composite::CompositeEvaluator;
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::position::Position;
use engine::probabilities::Probabilities;

pub enum ScoreConfig {
    MoneyGame,
    OnePointer,
}

impl TryFrom<(u32, u32)> for ScoreConfig {
    type Error = &'static str;

    #[inline]
    fn try_from((x_away, o_away): (u32, u32)) -> Result<Self, Self::Error> {
        match (x_away, o_away) {
            (0, 0) => Ok(ScoreConfig::MoneyGame),
            (1, 1) => Ok(ScoreConfig::OnePointer),
            (0, 1) => Err("If x_away is 0, then o_away must also be 0."),
            (1, 0) => Err("If o_away is 0, then x_away must also be 0."),
            (_, _) => Err("Currently only 1 pointers and money games are supported."),
        }
    }
}

impl ScoreConfig {
    #[inline]
    pub fn value(&self) -> impl Fn(&Probabilities) -> f32 {
        match self {
            ScoreConfig::OnePointer => |p: &Probabilities| p.win(),
            ScoreConfig::MoneyGame => |p: &Probabilities| p.equity(),
        }
    }
}

pub struct WildbgApi<T: Evaluator> {
    evaluator: T,
}

impl WildbgApi<CompositeEvaluator> {
    pub fn try_default() -> Result<Self, String> {
        CompositeEvaluator::try_default().map(|evaluator| Self { evaluator })
    }
}

impl<T: Evaluator> WildbgApi<T> {
    #[inline]
    pub fn with_evaluator(evaluator: T) -> Self {
        Self { evaluator }
    }

    #[inline]
    pub fn probabilities(&self, position: &Position) -> Probabilities {
        self.evaluator.eval(position)
    }

    #[inline]
    pub fn all_moves(
        &self,
        position: &Position,
        dice: &Dice,
        config: &ScoreConfig,
    ) -> Vec<(Position, Probabilities)> {
        self.evaluator
            .positions_and_probabilities(position, dice, config.value())
    }

    #[inline]
    pub fn best_move(&self, position: &Position, dice: &Dice, config: &ScoreConfig) -> BgMove {
        let new_position = self.evaluator.best_position(position, dice, config.value());
        BgMove::new(position, &new_position.sides_switched(), dice)
    }

    pub fn cube_info(&self, position: &Position) -> CubeInfo {
        CubeInfo::from(&self.evaluator.eval(position))
    }
}

#[cfg(test)]
mod tests {
    use crate::bg_move::{BgMove, MoveDetail};
    use crate::wildbg_api::{ScoreConfig, WildbgApi};
    use engine::dice::Dice;
    use engine::evaluator::EvaluatorFake;
    use engine::pos;
    use engine::position::Position;

    fn position_with_lowest_equity() -> Position {
        pos!(x 5:1, 3:1; o 20:2).sides_switched()
    }

    /// Test double. Returns not so good probabilities for `expected_pos`, better for everything else.
    fn evaluator_fake() -> EvaluatorFake {
        let mut fake = EvaluatorFake::with_default([0.38, 0.2, 0.1, 0.12, 0.1, 0.1].into());
        fake.insert(
            position_with_lowest_equity(),
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1].into(),
        );
        fake
    }

    #[test]
    fn best_move_1ptr() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = evaluator_fake();
        let api = WildbgApi { evaluator };
        // When
        let config = ScoreConfig::OnePointer;
        let bg_move = api.best_move(&given_pos, &Dice::new(4, 2), &config);
        // Then
        let expected_move = BgMove {
            details: vec![MoveDetail { from: 7, to: 5 }, MoveDetail { from: 5, to: 1 }],
        };
        assert_eq!(bg_move, expected_move);
    }

    #[test]
    fn best_move_money_game() {
        // Given
        let given_pos = pos!(x 7:2; o 20:2);
        let evaluator = evaluator_fake();
        let api = WildbgApi { evaluator };
        // When
        let config = ScoreConfig::MoneyGame;
        let bg_move = api.best_move(&given_pos, &Dice::new(4, 2), &config);
        // Then
        let expected_move = BgMove {
            details: vec![MoveDetail { from: 7, to: 3 }, MoveDetail { from: 7, to: 5 }],
        };
        assert_eq!(bg_move, expected_move);
    }
}
