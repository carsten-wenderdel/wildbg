use crate::composite::GameOverEvaluator;
use crate::evaluator::{Evaluator, PartialEvaluator};
use crate::inputs::{ContactInputsGen, RaceInputsGen};
use crate::onnx::OnnxEvaluator;
use crate::position::{GamePhase, OngoingPhase, Position};
use crate::probabilities::Probabilities;

pub struct ComplexEvaluator {
    contact_evaluator: OnnxEvaluator<ContactInputsGen>,
    race_evaluator: OnnxEvaluator<RaceInputsGen>,
    game_over_evaluator: GameOverEvaluator,
}

impl ComplexEvaluator {}

impl Evaluator for ComplexEvaluator {
    fn eval(&self, pos: &Position) -> Probabilities {
        match pos.game_phase() {
            GamePhase::Ongoing(ongoing) => match ongoing {
                OngoingPhase::Contact => self.contact_evaluator.eval(pos),
                OngoingPhase::Race => self.race_evaluator.eval(pos),
            },
            GamePhase::GameOver(_) => self
                .game_over_evaluator
                .try_eval(pos)
                .expect("GameOver must be handled by this."),
        }
    }
}

impl ComplexEvaluator {
    pub fn try_default() -> Option<Self> {
        let contact_evaluator = OnnxEvaluator::contact_default()?;
        let race_evaluator = OnnxEvaluator::race_default()?;
        Some(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }

    pub fn default_tests() -> Self {
        let contact_evaluator = OnnxEvaluator::contact_default_tests();
        let race_evaluator = OnnxEvaluator::race_default_tests();
        Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        }
    }

    pub fn from_file_paths(contact_path: &str, race_path: &str) -> Option<Self> {
        let contact_evaluator = OnnxEvaluator::from_file_path(contact_path, ContactInputsGen {})?;
        let race_evaluator = OnnxEvaluator::from_file_path(race_path, RaceInputsGen {})?;
        Some(Self {
            contact_evaluator,
            race_evaluator,
            game_over_evaluator: GameOverEvaluator {},
        })
    }
}
