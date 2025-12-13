use crate::position_finder::MoveSelector;
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::multiply::MultiPlyEvaluator;
use engine::position::GamePhase::Ongoing;
use engine::position::{OngoingPhase, Position};
use engine::probabilities::Probabilities;

pub(super) trait DiscrepantEvaluators {
    type Strong: Evaluator;
    type Weak: Evaluator;
    fn strong(&self) -> &Self::Strong;
    fn weak(&self) -> &Self::Weak;
}

pub(super) struct MultiPlyDiscrepancy<T: Evaluator> {
    pub(super) multiply: MultiPlyEvaluator<T>,
}

impl<T: Evaluator> DiscrepantEvaluators for MultiPlyDiscrepancy<T> {
    type Strong = MultiPlyEvaluator<T>;
    type Weak = T;

    fn strong(&self) -> &Self::Strong {
        &self.multiply
    }

    fn weak(&self) -> &Self::Weak {
        &self.multiply.evaluator
    }
}

/// Finds positions where two different evaluators return different equities.
pub(super) struct DiscrepancySelector<T: DiscrepantEvaluators> {
    pub(super) evaluators: T,
    pub(super) threshold: f32,
}

impl<T: DiscrepantEvaluators> MoveSelector for DiscrepancySelector<T> {
    fn next_and_found(
        &mut self,
        position: Position,
        dice: Dice,
        phase: OngoingPhase,
    ) -> (Position, Vec<Position>) {
        assert_eq!(
            phase,
            OngoingPhase::Race,
            "For now, we only deal with race positions."
        );

        let positions = position.all_positions_after_moving(&dice);

        // Only one legal move, nothing to compare.
        if positions.len() == 1 {
            return (*positions.first().unwrap(), vec![]);
        }
        // Game ending, nothing to compare.
        if let Some(end_of_game) = positions.iter().find(|p| p.has_lost()) {
            return (*end_of_game, vec![]);
        }

        let weak_positions_and_probabilities = self
            .evaluators
            .weak()
            .positions_and_probabilities_by_equity(&position, &dice);
        let weak_pos = weak_positions_and_probabilities
            .first()
            .unwrap()
            .0
            .sides_switched();

        if weak_pos.game_phase() != Ongoing(phase) {
            // We are only interested in positions of type `phase`, so let's not waste time using the strong evaluator.
            return (weak_pos, vec![]);
        }

        let strong_positions_and_probabilities = self
            .evaluators
            .strong()
            .positions_and_probabilities_by_equity(&position, &dice);
        let strong_pos = strong_positions_and_probabilities
            .first()
            .unwrap()
            .0
            .sides_switched();

        if strong_pos.game_phase() != Ongoing(phase) {
            // weak_pos and _strong_pos are in different phases. Let's ignore that for now.
            return (strong_pos, vec![]);
        }

        if weak_pos == strong_pos {
            (strong_pos, vec![])
        } else {
            let discrepancy = self.discrepancy_between_best_positions(
                &weak_positions_and_probabilities,
                &strong_positions_and_probabilities,
            );
            if discrepancy < self.threshold {
                (strong_pos, vec![])
            } else {
                (strong_pos, vec![weak_pos, strong_pos])
            }
        }
    }
}

impl<T: DiscrepantEvaluators> DiscrepancySelector<T> {
    fn discrepancy_between_best_positions(
        &mut self,
        weak_pos_and_probs: &[(Position, Probabilities)],
        strong_pos_and_probs: &[(Position, Probabilities)],
    ) -> f32 {
        assert_eq!(weak_pos_and_probs.len(), strong_pos_and_probs.len());
        let (weak_pos, weak_probs) = weak_pos_and_probs.first().unwrap().clone();
        let (strong_pos, strong_probs) = strong_pos_and_probs.first().unwrap().clone();
        let weak_equity = weak_probs.equity();
        let strong_equity = strong_probs.equity();

        let weak_equity_for_strong_pos = weak_pos_and_probs
            .iter()
            .find(|(pos, _)| pos == &strong_pos)
            .unwrap()
            .1
            .equity();
        let strong_equity_for_weak_pos = strong_pos_and_probs
            .iter()
            .find(|(pos, _)| pos == &weak_pos)
            .unwrap()
            .1
            .equity();

        // Positions and equities are already reverted.
        assert!(weak_equity >= weak_equity_for_strong_pos);
        assert!(strong_equity >= strong_equity_for_weak_pos);
        (weak_equity - weak_equity_for_strong_pos) + (strong_equity - strong_equity_for_weak_pos)
    }
}

#[cfg(test)]
mod tests {
    use crate::position_finder::MoveSelector;
    use crate::position_finder::discrepancy_selector::{DiscrepancySelector, DiscrepantEvaluators};
    use engine::dice::Dice;
    use engine::evaluator::{Evaluator, EvaluatorFake};
    use engine::pos;
    use engine::position::OngoingPhase;
    use engine::probabilities::Probabilities;

    struct DiscrepancyFake {
        strong: EvaluatorFake,
        weak: EvaluatorFake,
    }

    impl DiscrepantEvaluators for DiscrepancyFake {
        type Strong = EvaluatorFake;
        type Weak = EvaluatorFake;

        fn strong(&self) -> &Self::Strong {
            &self.strong
        }

        fn weak(&self) -> &Self::Weak {
            &self.weak
        }
    }

    /// When one of those evaluator fakes is used, it will crash and the test will fail.
    fn unused_evaluator_fake() -> DiscrepancyFake {
        let strong = EvaluatorFake::with_no_default();
        let weak = EvaluatorFake::with_no_default();
        DiscrepancyFake { strong, weak }
    }

    #[test]
    fn winning_move_returns_no_positions() {
        // Given
        let mut selector = DiscrepancySelector {
            evaluators: unused_evaluator_fake(),
            threshold: 0.03,
        };

        // When
        let pos = pos![x 1:1, 3:1; o 24:1];
        let (pos, found) = selector.next_and_found(pos, Dice::new(3, 1), OngoingPhase::Race);

        // Then
        assert_eq!(pos, pos![x 1:1; o]);
        assert!(found.is_empty());
    }

    #[test]
    fn only_one_legal_move_returns_no_positions() {
        // Given
        let mut selector = DiscrepancySelector {
            evaluators: unused_evaluator_fake(),
            threshold: 0.03,
        };

        // When
        let pos = pos![x 10:1 ; o 24:1];
        let (_, found) = selector.next_and_found(pos, Dice::new(3, 1), OngoingPhase::Race);

        // Then
        assert!(found.is_empty());
    }

    #[test]
    fn both_evaluators_agree_about_best_move_returns_no_positions() {
        // Given
        let best_position = pos![x 1:10; o 10:5, 15:5];
        // best probabilities from the player's perspective means worst from the opponent's perspective.
        let best_probabilities: Probabilities = [0.6, 0.0, 0.1, 0.1, 0.1, 0.1].into();
        let default_probabilities: Probabilities = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1].into();

        let mut strong = EvaluatorFake::with_default(default_probabilities.clone());
        let mut weak = EvaluatorFake::with_default(default_probabilities);
        strong.insert(best_position, best_probabilities.clone());
        weak.insert(best_position, best_probabilities);

        let evaluators = DiscrepancyFake { strong, weak };
        let mut selector = DiscrepancySelector {
            evaluators,
            threshold: 0.03,
        };

        // When
        let pos = pos![x 15:6, 10:4 ; o 24:10];
        let (pos, found) = selector.next_and_found(pos, Dice::new(3, 2), OngoingPhase::Race);

        // Then
        assert_eq!(pos, best_position);
        assert!(found.is_empty());
    }

    #[test]
    fn possible_moves_contains_both_race_and_contact_positions_hence_nothing_is_returned() {
        // Given
        let best_position = pos![x 13:10; o 12:2, 6:1, 1:1];
        // best probabilities from the player's perspective means worst from the opponent's perspective.
        let best_probabilities: Probabilities = [0.6, 0.0, 0.1, 0.1, 0.1, 0.1].into();
        let default_probabilities: Probabilities = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1].into();

        let strong = EvaluatorFake::with_default(default_probabilities.clone());
        let mut weak = EvaluatorFake::with_default(default_probabilities);
        weak.insert(best_position, best_probabilities);

        let evaluators = DiscrepancyFake { strong, weak };
        let mut selector = DiscrepancySelector {
            evaluators,
            threshold: 0.02,
        };

        // When
        let pos = pos![x 24:2, 13:2; o 12:10];
        let (pos, found) = selector.next_and_found(pos, Dice::new(3, 2), OngoingPhase::Race);

        // Then
        assert_eq!(pos, best_position);
        assert!(found.is_empty());
    }

    #[test]
    fn both_evaluators_disagree_about_best_move_returns_two_positions() {
        // Given
        // best probabilities from the player's perspective means worst from the opponent's perspective.
        let best_probabilities: Probabilities = [0.6, 0.0, 0.1, 0.1, 0.1, 0.1].into();
        let default_probabilities: Probabilities = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1].into();

        let mut strong = EvaluatorFake::with_default(default_probabilities.clone());
        let mut weak = EvaluatorFake::with_default(default_probabilities);
        let best_strong_pos = pos![x 1:10; o 10:5, 15:5];
        let best_weak_pos = pos![x 1:10; o 10:4, 12:1, 13:1, 15:4];
        strong.insert(best_strong_pos, best_probabilities.clone());
        weak.insert(best_weak_pos, best_probabilities);

        let evaluators = DiscrepancyFake { strong, weak };
        let mut selector = DiscrepancySelector {
            evaluators,
            threshold: 0.03,
        };

        // When
        let pos = pos![x 15:6, 10:4 ; o 24:10];
        let (pos, found) = selector.next_and_found(pos, Dice::new(3, 2), OngoingPhase::Race);

        // Then
        assert_eq!(pos, best_strong_pos);
        assert_eq!(found, vec![best_weak_pos, best_strong_pos]);
    }

    #[test]
    fn both_evaluators_disagree_but_discrepancy_is_under_threshold() {
        // Given
        // best probabilities from the player's perspective means worst from the opponent's perspective.
        let best_probabilities: Probabilities = [0.6, 0.0, 0.1, 0.1, 0.1, 0.1].into();
        let default_probabilities: Probabilities = [0.59, 0.01, 0.1, 0.1, 0.1, 0.1].into();

        let mut strong = EvaluatorFake::with_default(default_probabilities.clone());
        let mut weak = EvaluatorFake::with_default(default_probabilities);
        let best_strong_pos = pos![x 1:10; o 10:5, 15:5];
        let best_weak_pos = pos![x 1:10; o 10:4, 12:1, 13:1, 15:4];
        strong.insert(best_strong_pos, best_probabilities.clone());
        weak.insert(best_weak_pos, best_probabilities);

        let evaluators = DiscrepancyFake { strong, weak };
        let mut selector = DiscrepancySelector {
            evaluators,
            threshold: 0.03,
        };

        // When
        let pos = pos![x 15:6, 10:4 ; o 24:10];
        let (pos, found) = selector.next_and_found(pos, Dice::new(3, 2), OngoingPhase::Race);

        // Then
        assert_eq!(pos, best_strong_pos);
        assert_eq!(found, vec![]);
    }

    #[test]
    fn discrepancy_between_best_positions_calculates_correctly() {
        // Given
        let strong_default: Probabilities = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1].into();
        let strong_best: Probabilities = [0.55, 0.05, 0.1, 0.1, 0.1, 0.1].into();
        let weak_default: Probabilities = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1].into();
        let weak_best: Probabilities = [0.41, 0.19, 0.1, 0.1, 0.1, 0.1].into();

        let best_strong_pos = pos![x 1:10; o 10:5, 15:5];
        let best_weak_pos = pos![x 1:10; o 10:4, 12:1, 13:1, 15:4];

        let mut strong = EvaluatorFake::with_default(strong_default);
        let mut weak = EvaluatorFake::with_default(weak_default);
        strong.insert(best_strong_pos, strong_best);
        weak.insert(best_weak_pos, weak_best);

        let evaluators = DiscrepancyFake { strong, weak };
        let mut selector = DiscrepancySelector {
            evaluators,
            threshold: 0.03,
        };

        // When
        let pos = pos![x 15:6, 10:4 ; o 24:10];
        let weak_pos_and_probs = selector
            .evaluators
            .weak()
            .positions_and_probabilities_by_equity(&pos, &Dice::new(3, 2));
        let strong_pos_and_probs = selector
            .evaluators
            .strong()
            .positions_and_probabilities_by_equity(&pos, &Dice::new(3, 2));

        // Then
        let actual_discrepancy =
            selector.discrepancy_between_best_positions(&weak_pos_and_probs, &strong_pos_and_probs);
        let expected = 0.06;
        // Floating points are not base 10 -> minor errors.
        assert!((actual_discrepancy - expected).abs() < 0.000001);
    }
}
