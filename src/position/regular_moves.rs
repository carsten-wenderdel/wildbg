use crate::position::{Position, O_BAR, X_BAR};
use std::cmp::max;

impl Position {
    /// Returns a vector of all possible moves when rolling a double.
    /// The return value both contains the moves and the resulting positions.
    /// The move is encoded in an array of 2 numbers, representing the pip from where to move.
    /// Element '0' in that array is the pip from where to move with the first die,
    /// element '1' is the pip from where to move with the second die.
    #[allow(dead_code)]
    fn all_regular_moves(&self, die1: usize, die2: usize) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        match self.pips[X_BAR] {
            0 => self.moves_with_0_checkers_on_bar(die1, die2),
            1 => self.moves_with_1_checker_on_bar(die1, die2),
            _ => self.moves_with_2_checkers_on_bar(die1, die2),
        }
    }

    /// Regular moves with no checkers on the bar.
    fn moves_with_1_checker_on_bar(
        &self,
        die1: usize,
        die2: usize,
    ) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] == 1);

        if self.can_enter(die1) {
            let position = self.clone_and_enter_single_checker(die1);
            let mut moves1 = position.one_checker_moves(die2, 1, Some(X_BAR));
            if self.can_enter(die2) {
                let position = self.clone_and_enter_single_checker(die2);
                let mut moves2 = position.one_checker_moves(die1, 0, Some(X_BAR));
                if moves2.first().unwrap().0[0] != None {
                    if moves1.first().unwrap().0[1] != None {
                        // Both move vectors contain moves with both checkers
                        if self.pips[X_BAR - die1] != -1 && self.pips[X_BAR - die2] != -1 {
                            // Moving the checker from X_BAR to (X_BAR - die1 - die2) is in both vectors
                            // Nothing is hit, so it's a duplication that must be removed
                            moves2.retain(|m| m.0[0] != Some(X_BAR - die2));
                        }
                        moves1.append(&mut moves2);
                        return moves1;
                    } else {
                        // Only moves2 contains moves with both checkers
                        return moves2;
                    }
                } else {
                    // moves2 does not contain moves with both checkers
                    // We return moves1 - if it contains moves with both checkers, it's perfect.
                    // If moves1 only contains a single move with a single checker, the bigger die wins.
                    return moves1;
                }
            } else {
                // die2 can't enter
                return moves1;
            }
        } else {
            // die1 can't enter
            if self.can_enter(die2) {
                let position = self.clone_and_enter_single_checker(die2);
                return position.one_checker_moves(die1, 0, Some(X_BAR));
            } else {
                // Neither die1 nor die2 allow entering - return the identity move.
                return Vec::from([([None, None], self.clone())]);
            }
        }
    }

    /// Regular moves with no checkers on the bar.
    fn moves_with_0_checkers_on_bar(
        &self,
        die1: usize,
        die2: usize,
    ) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] == 0);

        match self.move_possibilities(die1, die2) {
            MovePossibilities::None => Vec::from([([None, None], self.clone())]),
            MovePossibilities::One { die } => {
                let index = if die == die1 { 0 } else { 1 };
                self.one_checker_moves(die, index, None)
            }
            MovePossibilities::Two => self.two_checker_moves(die1, die2),
        }
    }

    /// All moves where one die/pip is already fixed.
    /// `index` is the position in the array from where the new `die` should be moved.
    fn one_checker_moves(
        &self,
        die: usize,
        index: usize,
        other_value: Option<usize>,
    ) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(self.pips[X_BAR] == 0);

        let mut moves: Vec<([Option<usize>; 2], Position)> = Vec::new();
        for i in (1..X_BAR).rev() {
            if self.can_move(i, die) {
                let position = self.clone_and_move_single_checker(i, die);
                let mut the_move = [other_value, other_value];
                the_move[index] = Some(i);
                moves.push((the_move, position));
            }
        }
        if moves.is_empty() {
            let mut the_move = [other_value, other_value];
            the_move[index] = None;
            moves.push((the_move, self.clone()));
        }
        moves
    }

    // All moves with no checkers on the bar where two checkers can be moved.
    fn two_checker_moves(&self, die1: usize, die2: usize) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] == 0);

        let mut moves: Vec<([Option<usize>; 2], Position)> = Vec::new();
        for i in (1..X_BAR).rev() {
            // Looking at moves where die1 *can* be used first
            if self.can_move(i, die1) {
                let position = self.clone_and_move_single_checker(i, die1);
                // We have to look at all pips in the home board, in case bearing off just became possible. This is why the 7 appears in the max function.
                for j in (1..max(7, i + 1)).rev() {
                    if position.can_move(j, die2) {
                        let final_position = position.clone_and_move_single_checker(j, die2);
                        moves.push(([Some(i), Some(j)], final_position));
                    }
                }
            }
            // Looking at moves where die2 *must* be moved first
            // This can be because of two reasons:
            // 1. We make two movements with the same checker, but for die1 it's initially blocked.
            // 2. We make two movements with different checkers and hit something with the first move, either die1 or die2
            // 3. After moving die2, we now can bear off with die1, which was illegal before.
            if self.can_move(i, die2) {
                let position = self.clone_and_move_single_checker(i, die2);
                // We have to look at all pips in the home board, in case bearing off just became possible. This is why the 7 appears in the max function.
                for j in (1..max(7, i + 1)).rev() {
                    if position.can_move(j, die1) {
                        // This describes cases 1 and 2:
                        let two_movements_with_same_checker_and_different_outcome =
                            j == i - die2 && (self.pips[i - die1] < 0 || self.pips[i - die2] < 0);
                        // This describes case 3:
                        let bear_off_was_illegal_but_not_anymore = i > 6 && j <= die1;
                        if two_movements_with_same_checker_and_different_outcome
                            || bear_off_was_illegal_but_not_anymore
                        {
                            let final_position = position.clone_and_move_single_checker(j, die1);
                            moves.push(([Some(j), Some(i)], final_position));
                        }
                    }
                }
            }
        }
        debug_assert!(!moves.is_empty());
        moves
    }

    /// All moves (well, exactly one) when at least two checkers are on the bar.
    fn moves_with_2_checkers_on_bar(
        &self,
        die1: usize,
        die2: usize,
    ) -> Vec<([Option<usize>; 2], Position)> {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] > 1);

        let mut position = self.clone();
        let mut the_move = [None, None];

        if position.can_enter(die1) {
            position.enter_single_checker(die1);
            the_move[0] = Some(X_BAR);
        }
        if position.can_enter(die2) {
            position.enter_single_checker(die2);
            the_move[1] = Some(X_BAR);
        }
        Vec::from([(the_move, position)])
    }

    /// Will return 2 if 2 or more checkers can be moved.
    /// Will return 0 if no checker can be moved.
    fn move_possibilities(&self, die1: usize, die2: usize) -> MovePossibilities {
        debug_assert!(die1 > die2);
        debug_assert!(self.pips[X_BAR] == 0);

        let mut can_move_die1 = false;
        let mut can_move_die2 = false;

        // Move die1 first
        for i in (1..X_BAR).rev() {
            if self.can_move(i, die1) {
                can_move_die1 = true;
                let position = self.clone_and_move_single_checker(i, die1);
                // We have to look at all pips in the home board, in case bearing off just became possible. This is why the 7 appears in the max function.
                for j in (1..max(7, i + 1)).rev() {
                    if position.can_move(j, die2) {
                        return MovePossibilities::Two;
                    }
                }
            }
        }

        // Move die2 first, assuming die1 cannot be moved first
        for i in (1..X_BAR).rev() {
            if self.can_move(i, die2) {
                can_move_die2 = true;
                let position = self.clone_and_move_single_checker(i, die2);
                // If die1 and die2 could be used with different checkers without bearing off, then we would not get here.
                // So, we only need to check if die1 can be moved with the same checker as die2.
                if i > die2 && position.can_move(i - die2, die1) {
                    return MovePossibilities::Two;
                }
                // Now checking bearing off
                for j in (1..7).rev() {
                    if position.can_move(j, die1) {
                        return MovePossibilities::Two;
                    }
                }
            }
        }

        if can_move_die1 {
            MovePossibilities::One { die: die1 }
        } else if can_move_die2 {
            MovePossibilities::One { die: die2 }
        } else {
            MovePossibilities::None
        }
    }

    fn can_enter(&self, die: usize) -> bool {
        debug_assert!(
            self.pips[X_BAR] > 0,
            "only call this function if x has checkers on the bar"
        );
        self.pips[X_BAR - die] > -2
    }

    fn clone_and_enter_single_checker(&self, die: usize) -> Position {
        let mut position = self.clone();
        position.enter_single_checker(die);
        position
    }

    fn enter_single_checker(&mut self, die: usize) {
        debug_assert!(
            self.pips[X_BAR] > 0,
            "only call this function if x has checkers on the bar"
        );
        debug_assert!(
            self.pips[X_BAR - die] > -2,
            "only call this function if x can enter"
        );
        self.pips[X_BAR] -= 1;
        if self.pips[X_BAR - die] == -1 {
            // hit opponent
            self.pips[X_BAR - die] = 1;
            self.pips[O_BAR] -= 1;
        } else {
            // no hitting
            self.pips[X_BAR - die] += 1;
        }
    }
}

enum MovePossibilities {
    None,
    One { die: usize },
    Two,
}

#[cfg(test)]
mod tests {
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

    // Two checkers on bar

    #[test]
    fn cannot_enter_with_two_checkers_on_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([None, None], position)]));
    }

    #[test]
    fn can_enter_bigger_die_with_two_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (20, 1), (10, 2)]),
            &HashMap::from([(22, 2)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([Some(X_BAR), None], expected)]));
    }

    #[test]
    fn can_enter_smaller_die_with_two_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 2), (10, 2)]),
            &HashMap::from([(22, 1), (20, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (22, 1), (10, 2)]),
            &HashMap::from([(20, 2), (O_BAR, 1)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([None, Some(X_BAR)], expected)]));
    }

    #[test]
    fn can_enter_both_with_three_on_the_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 3), (10, 2)]),
            &HashMap::from([(20, 1)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(X_BAR, 1), (22, 1), (20, 1), (10, 2)]),
            &HashMap::from([(O_BAR, 1)]),
        );
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([Some(X_BAR), Some(X_BAR)], expected)]));
    }

    // One checker on bar

    #[test]
    fn cannot_enter_with_one_checker_on_bar() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 1), (10, 2)]),
            &HashMap::from([(22, 2), (20, 2)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        assert_eq!(moves, Vec::from([([None, None], position)]));
    }

    #[test]
    fn can_enter_with_bigger_die_but_no_other_movement() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 1), (10, 2)]),
            &HashMap::from([(22, 2), (20, 1), (17, 2), (7, 3)]),
        );
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected = Position::from(
            &HashMap::from([(20, 1), (10, 2)]),
            &HashMap::from([(22, 2), (17, 2), (7, 3), (O_BAR, 1)]),
        );
        assert_eq!(moves, Vec::from([([Some(X_BAR), None], expected)]));
    }

    #[test]
    fn can_enter_with_smaller_die_but_no_other_movement() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 1)]),
            &HashMap::from([(19, 2), (14, 2)]),
        );
        // When
        let moves = position.all_regular_moves(6, 5);
        // Then
        let expected = Position::from(
            &HashMap::from([(20, 1)]),
            &HashMap::from([(19, 2), (14, 2)]),
        );
        assert_eq!(moves, Vec::from([([None, Some(X_BAR)], expected)]));
    }

    #[test]
    fn could_enter_with_either_die_but_must_use_bigger_one() {
        // Given
        let position = Position::from(&HashMap::from([(X_BAR, 1)]), &HashMap::from([(20, 2)]));
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected = Position::from(&HashMap::from([(22, 1)]), &HashMap::from([(20, 2)]));
        assert_eq!(moves, Vec::from([([Some(X_BAR), None], expected)]));
    }

    #[test]
    fn only_entering_with_smaller_die_allows_two_checkers_to_move() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 1), (12, 1)]),
            &HashMap::from([(20, 2), (10, 2)]),
        );
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected = Position::from(
            &HashMap::from([(23, 1), (9, 1)]),
            &HashMap::from([(20, 2), (10, 2)]),
        );
        assert_eq!(moves, Vec::from([([Some(12), Some(X_BAR)], expected)]));
    }

    #[test]
    fn only_entering_with_bigger_die_allows_two_checkers_to_move() {
        // Given
        let position = Position::from(
            &HashMap::from([(X_BAR, 1), (12, 1)]),
            &HashMap::from([(20, 2), (9, 2)]),
        );
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected = Position::from(
            &HashMap::from([(22, 1), (10, 1)]),
            &HashMap::from([(20, 2), (9, 2)]),
        );
        assert_eq!(moves, Vec::from([([Some(X_BAR), Some(12)], expected)]));
    }

    #[test]
    fn entering_with_either_die_allowed_but_only_one_final_position() {
        // Given
        let position = Position::from(&HashMap::from([(X_BAR, 1)]), &HashMap::from([(9, 2)]));
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected = Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(9, 2)]));
        assert_eq!(moves, Vec::from([([Some(X_BAR), Some(22)], expected)]));
    }

    #[test]
    fn final_position_but_different_move_because_die1_hits_opponent() {
        // Given
        let position = Position::from(&HashMap::from([(X_BAR, 1)]), &HashMap::from([(22, 1)]));
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected1 = (
            [Some(X_BAR), Some(22)],
            Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(O_BAR, 1)])),
        );
        let expected2 = (
            [Some(23), Some(X_BAR)],
            Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(22, 1)])),
        );
        assert_eq!(moves, Vec::from([expected1, expected2]));
    }

    #[test]
    fn final_position_but_different_move_because_die2_hits_opponent() {
        // Given
        let position = Position::from(&HashMap::from([(X_BAR, 1)]), &HashMap::from([(23, 1)]));
        // When
        let moves = position.all_regular_moves(3, 2);
        // Then
        let expected1 = (
            [Some(X_BAR), Some(22)],
            Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(23, 1)])),
        );
        let expected2 = (
            [Some(23), Some(X_BAR)],
            Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(O_BAR, 1)])),
        );
        assert_eq!(moves, Vec::from([expected1, expected2]));
    }

    // No checkers on bar

    #[test]
    fn cannot_user_either_die() {
        // Given
        let position = Position::from(
            &HashMap::from([(10, 2), (2, 3)]),
            &HashMap::from([(8, 2), (6, 2)]),
        );
        // When
        let moves = position.all_regular_moves(4, 2);
        // Then
        assert_eq!(moves.len(), 1);
        assert_eq!(moves, Vec::from([([None, None], position)]));
    }

    #[test]
    fn forced_only_smaller_die() {
        // Given
        let position = Position::from(&HashMap::from([(7, 2)]), &HashMap::from([(2, 2)]));
        // When
        let moves = position.all_regular_moves(5, 2);
        // Then
        let expected = Position::from(&HashMap::from([(7, 1), (5, 1)]), &HashMap::from([(2, 2)]));
        assert_eq!(moves, Vec::from([([None, Some(7)], expected)]));
    }

    #[test]
    fn forced_smaller_die_first_then_bear_off() {
        // Given
        let position = Position::from(&HashMap::from([(8, 1), (4, 3)]), &HashMap::from([(1, 2)]));
        // When
        let moves = position.all_regular_moves(4, 3);
        // Then
        let expected = Position::from(&HashMap::from([(5, 1), (4, 2)]), &HashMap::from([(1, 2)]));
        assert_eq!(moves, Vec::from([([Some(4), Some(8)], expected)]));
    }

    #[test]
    fn bigger_die_cannot_move_initially() {
        // Given
        let position = Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(16, 3)]));
        // When
        let moves = position.all_regular_moves(4, 3);
        // Then
        let expected = Position::from(&HashMap::from([(13, 1)]), &HashMap::from([(16, 3)]));
        assert_eq!(moves, Vec::from([([Some(17), Some(20)], expected)]));
    }

    #[test]
    fn smaller_first_allows_bear_off() {
        // Given
        let position = Position::from(&HashMap::from([(9, 1), (5, 1)]), &HashMap::from([(20, 2)]));
        // When
        let moves = position.all_regular_moves(5, 3);
        // Then
        let expected1 = (
            [Some(9), Some(5)],
            Position::from(&HashMap::from([(4, 1), (2, 1)]), &HashMap::from([(20, 2)])),
        );
        let expected2 = (
            [Some(9), Some(4)],
            Position::from(&HashMap::from([(5, 1), (1, 1)]), &HashMap::from([(20, 2)])),
        );
        let expected3 = (
            [Some(5), Some(9)],
            Position::from(&HashMap::from([(6, 1)]), &HashMap::from([(20, 2)])),
        );
        assert_eq!(moves, Vec::from([expected1, expected2, expected3]));
    }

    #[test]
    fn could_bear_off_but_could_do_other_moves_as_well() {
        // Given
        let position = Position::from(
            &HashMap::from([(5, 2), (4, 3), (3, 1)]),
            &HashMap::from([(20, 1)]),
        );
        // When
        let moves = position.all_regular_moves(4, 3);
        // Then
        let expected1 = (
            [Some(5), Some(5)],
            Position::from(
                &HashMap::from([(4, 3), (3, 1), (2, 1), (1, 1)]),
                &HashMap::from([(20, 1)]),
            ),
        );
        let expected2 = (
            [Some(5), Some(4)],
            Position::from(
                &HashMap::from([(5, 1), (4, 2), (3, 1), (1, 2)]),
                &HashMap::from([(20, 1)]),
            ),
        );
        let expected3 = (
            [Some(5), Some(3)],
            Position::from(
                &HashMap::from([(5, 1), (4, 3), (1, 1)]),
                &HashMap::from([(20, 1)]),
            ),
        );
        let expected4 = (
            [Some(4), Some(5)],
            Position::from(
                &HashMap::from([(5, 1), (4, 2), (3, 1), (2, 1)]),
                &HashMap::from([(20, 1)]),
            ),
        );
        let expected5 = (
            [Some(4), Some(4)],
            Position::from(
                &HashMap::from([(5, 2), (4, 1), (3, 1), (1, 1)]),
                &HashMap::from([(20, 1)]),
            ),
        );
        let expected6 = (
            [Some(4), Some(3)],
            Position::from(&HashMap::from([(5, 2), (4, 2)]), &HashMap::from([(20, 1)])),
        );
        assert_eq!(
            moves,
            Vec::from([expected1, expected2, expected3, expected4, expected5, expected6])
        );
    }

    #[test]
    fn only_one_move_if_order_is_not_important() {
        // Given
        let position = Position::from(&HashMap::from([(20, 1)]), &HashMap::from([(22, 2)]));
        // When
        let moves = position.all_regular_moves(4, 3);
        // Then
        let expected = Position::from(&HashMap::from([(13, 1)]), &HashMap::from([(22, 2)]));
        assert_eq!(moves, Vec::from([([Some(20), Some(16)], expected)]),);
    }

    #[test]
    fn only_one_move_in_home_board_if_order_is_not_important() {
        // Given
        let position = Position::from(&HashMap::from([(5, 1)]), &HashMap::from([(22, 2)]));
        // When
        let moves = position.all_regular_moves(2, 1);
        // Then
        let expected = Position::from(&HashMap::from([(2, 1)]), &HashMap::from([(22, 2)]));
        assert_eq!(moves, Vec::from([([Some(5), Some(3)], expected)]),);
    }

    #[test]
    fn two_moves_if_bigger_die_hits_opponent() {
        // Given
        let position = Position::from(&HashMap::from([(10, 1)]), &HashMap::from([(6, 1)]));
        // When
        let moves = position.all_regular_moves(4, 2);
        // Then
        let expected1 = (
            [Some(10), Some(6)],
            Position::from(&HashMap::from([(4, 1)]), &HashMap::from([(O_BAR, 1)])),
        );
        let expected2 = (
            [Some(8), Some(10)],
            Position::from(&HashMap::from([(4, 1)]), &HashMap::from([(6, 1)])),
        );
        assert_eq!(moves, Vec::from([expected1, expected2]));
    }

    #[test]
    fn two_moves_if_smaller_die_hits_opponent() {
        // Given
        let position = Position::from(&HashMap::from([(5, 1)]), &HashMap::from([(4, 1)]));
        // When
        let moves = position.all_regular_moves(3, 1);
        // Then
        let expected1 = (
            [Some(5), Some(2)],
            Position::from(&HashMap::from([(1, 1)]), &HashMap::from([(4, 1)])),
        );
        let expected2 = (
            [Some(4), Some(5)],
            Position::from(&HashMap::from([(1, 1)]), &HashMap::from([(O_BAR, 1)])),
        );
        assert_eq!(moves, Vec::from([expected1, expected2]));
    }
}
