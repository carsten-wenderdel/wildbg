use crate::dice::RegularDice;
use crate::position::{Position, O_BAR, X_BAR};
use std::cmp::max;

impl Position {
    /// Returns all legal positions after rolling a double and then moving.
    /// The return values have not switched sides yet.
    pub(super) fn all_positions_after_regular_move(&self, dice: &RegularDice) -> Vec<Position> {
        debug_assert!(dice.big > dice.small);
        match self.pips[X_BAR] {
            0 => self.moves_with_0_checkers_on_bar(dice),
            1 => self.moves_with_1_checker_on_bar(dice),
            _ => self.moves_with_2_checkers_on_bar(dice),
        }
    }

    /// Regular moves with exactly 1 checker on the bar.
    fn moves_with_1_checker_on_bar(&self, dice: &RegularDice) -> Vec<Position> {
        debug_assert!(self.pips[X_BAR] == 1);

        if self.can_enter(dice.big) {
            let enter_big = self.clone_and_enter_single_checker(dice.big);
            let moves_after_entering_big = enter_big.one_checker_moves(dice.small);
            if self.can_enter(dice.small) {
                let enter_small = self.clone_and_enter_single_checker(dice.small);
                let moves_after_entering_small = enter_small.one_checker_moves(dice.big);
                match moves_after_entering_small {
                    None => moves_after_entering_big.unwrap_or(vec![enter_big]),
                    Some(small_moves) => match moves_after_entering_big {
                        None => small_moves,
                        Some(mut big_moves) => {
                            for position in small_moves {
                                if !big_moves.contains(&position) {
                                    // Lets say 21 is rolled: bar/24/22 and bar/23/22 will appear in both vectors.
                                    big_moves.push(position);
                                }
                            }
                            big_moves
                        }
                    },
                }
            } else {
                moves_after_entering_big.unwrap_or(vec![enter_big])
            }
        } else {
            // bigger die can't enter
            if self.can_enter(dice.small) {
                let position = self.clone_and_enter_single_checker(dice.small);
                let moved_twice = position.one_checker_moves(dice.big);
                moved_twice.unwrap_or(vec![position])
            } else {
                // Neither big nor small die allow entering - return the identity move.
                vec![self.clone()]
            }
        }
    }

    /// Regular moves with no checkers on the bar.
    fn moves_with_0_checkers_on_bar(&self, dice: &RegularDice) -> Vec<Position> {
        debug_assert!(self.pips[X_BAR] == 0);

        match self.move_possibilities(dice) {
            MovePossibilities::None => vec![self.clone()],
            MovePossibilities::One { die } => self
                .one_checker_moves(die)
                .expect("We already checked that moving one checker is possible"),
            MovePossibilities::Two => self.two_checker_moves(dice),
        }
    }

    /// All positions after moving a single checker once. If no move is possible it returns `None`.
    /// So if the return value is not `None`, the Vector is not empty.
    fn one_checker_moves(&self, die: usize) -> Option<Vec<Position>> {
        debug_assert!(self.pips[X_BAR] == 0);

        let mut moves: Vec<Position> = Vec::new();
        for i in (1..X_BAR).rev() {
            if self.can_move_in_board(i, die) {
                let position = self.clone_and_move_single_checker(i, die);
                moves.push(position);
            }
        }
        if moves.is_empty() {
            None
        } else {
            Some(moves)
        }
    }

    // All moves with no checkers on the bar where two checkers can be moved.
    fn two_checker_moves(&self, dice: &RegularDice) -> Vec<Position> {
        debug_assert!(self.pips[X_BAR] == 0);

        let mut moves: Vec<Position> = Vec::new();
        for i in (1..X_BAR).rev() {
            // Looking at moves where the big die *can* be used first
            if self.can_move_in_board(i, dice.big) {
                let position = self.clone_and_move_single_checker(i, dice.big);
                for j in (1..X_BAR).rev() {
                    if position.can_move_in_board(j, dice.small) {
                        let final_position = position.clone_and_move_single_checker(j, dice.small);
                        moves.push(final_position);
                    }
                }
            }
            // Looking at moves where the small die *must* be moved first
            // This can be because of two reasons:
            // 1. We make two movements with the same checker, but for the big die it's initially blocked.
            // 2. We make two movements with the same checker and hit something with the first movement, either with the big or the small die.
            // 3. After moving the small die, we now can bear off with the big die, which was illegal before.
            if self.can_move_in_board(i, dice.small) {
                let position = self.clone_and_move_single_checker(i, dice.small);
                // We have to look at all pips in the home board, in case bearing off just became possible. This is why the 7 appears in the max function.
                for j in (1..max(7, i + 1)).rev() {
                    if position.can_move_in_board(j, dice.big) {
                        // This describes cases 1 and 2:
                        let two_movements_with_same_checker_and_different_outcome = i
                            == j + dice.small
                            && i > dice.big
                            && i > dice.small
                            && (self.pips[i - dice.big] < 0 || self.pips[i - dice.small] < 0);
                        // This describes case 3:
                        let bear_off_was_illegal_but_not_anymore = i > 6 && j <= dice.big;
                        let could_not_bear_off_because_die_bigger_than_pip_and_checker_was_on_bigger_pip =
                            dice.big > j && i > j;
                        if two_movements_with_same_checker_and_different_outcome
                            || bear_off_was_illegal_but_not_anymore
                            || could_not_bear_off_because_die_bigger_than_pip_and_checker_was_on_bigger_pip
                        {
                            let final_position = position.clone_and_move_single_checker(j, dice.big);
                            moves.push( final_position);
                        }
                    }
                }
            }
        }
        debug_assert!(!moves.is_empty());
        moves
    }

    /// All moves (well, exactly one) when at least two checkers are on the bar.
    fn moves_with_2_checkers_on_bar(&self, dice: &RegularDice) -> Vec<Position> {
        debug_assert!(self.pips[X_BAR] > 1);

        let mut position = self.clone();
        if position.can_enter(dice.big) {
            position.enter_single_checker(dice.big);
        }
        if position.can_enter(dice.small) {
            position.enter_single_checker(dice.small);
        }
        vec![position]
    }

    /// Will return 2 if 2 or more checkers can be moved.
    /// Will return 0 if no checker can be moved.
    fn move_possibilities(&self, dice: &RegularDice) -> MovePossibilities {
        debug_assert!(self.pips[X_BAR] == 0);

        let mut can_move_big = false;
        let mut can_move_small = false;

        // Move die1 first
        for i in (1..X_BAR).rev() {
            if self.can_move_in_board(i, dice.big) {
                can_move_big = true;
                let position = self.clone_and_move_single_checker(i, dice.big);
                // We have to look at all pips in the home board, in case bearing off just became possible. This is why the 7 appears in the max function.
                for j in (1..max(7, i + 1)).rev() {
                    if position.can_move_in_board(j, dice.small) {
                        return MovePossibilities::Two;
                    }
                }
            }
        }

        // Move die2 first, assuming die1 cannot be moved first
        for i in (1..X_BAR).rev() {
            if self.can_move_in_board(i, dice.small) {
                can_move_small = true;
                let position = self.clone_and_move_single_checker(i, dice.small);
                // If die1 and die2 could be used with different checkers without bearing off, then we would not get here.
                // So, we only need to check if die1 can be moved with the same checker as die2.
                if i > dice.small && position.can_move_in_board(i - dice.small, dice.big) {
                    return MovePossibilities::Two;
                }
                // Now checking bearing off
                for j in (1..7).rev() {
                    if position.can_move_in_board(j, dice.big) {
                        return MovePossibilities::Two;
                    }
                }
            }
        }

        if can_move_big {
            MovePossibilities::One { die: dice.big }
        } else if can_move_small {
            MovePossibilities::One { die: dice.small }
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
    use crate::dice::RegularDice;
    use crate::pos;
    use crate::position::{Position, O_BAR, X_BAR};
    use std::collections::HashMap;

    // Two checkers on bar

    #[test]
    fn cannot_enter_with_two_checkers_on_bar() {
        // Given
        let position = pos!(x X_BAR:2, 10:2; o 22:2, 20:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice { big: 5, small: 3 });
        // Then
        assert_eq!(resulting_positions, vec![position]);
    }

    #[test]
    fn can_enter_bigger_die_with_two_on_the_bar() {
        // Given
        let position = pos!(x X_BAR:2, 10:2; o 22:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        let expected = pos!(x X_BAR:1, 20:1, 10:2; o 22:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn can_enter_smaller_die_with_two_on_the_bar() {
        // Given
        let position = pos!(x X_BAR:2, 10:2; o 22:1, 20:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        let expected = pos!(x X_BAR:1, 22:1, 10:2; o 20:2, O_BAR:1);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn can_enter_both_with_three_on_the_bar() {
        // Given
        let position = pos!(x X_BAR:3, 10:2; o 20:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        let expected = pos!(x X_BAR:1, 22:1, 20:1, 10:2; o O_BAR:1);
        assert_eq!(resulting_positions, vec![expected]);
    }

    // One checker on bar

    #[test]
    fn cannot_enter_with_one_checker_on_bar() {
        // Given
        let position = pos!(x X_BAR:1, 10:2; o 22:2, 20:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        assert_eq!(resulting_positions, vec![position]);
    }

    #[test]
    fn can_enter_with_bigger_die_but_no_other_movement() {
        // Given
        let position = pos!(x X_BAR:1, 10:2; o 22:2, 20:1, 17:2, 7:3);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        let expected = pos!(x 20:1, 10:2; o 22:2, 17:2, 7:3, O_BAR:1);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn can_enter_with_smaller_die_but_no_other_movement() {
        // Given
        let position = pos!(x X_BAR:1; o 19:2, 14:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(6, 5));
        // Then
        let expected = pos!(x 20:1; o 19:2, 14:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn could_enter_with_either_die_but_must_use_bigger_one() {
        // Given
        let position = pos!(x X_BAR:1; o 20:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected = pos!(x 22:1; o 20:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn only_entering_with_smaller_die_allows_two_checkers_to_move() {
        // Given
        let position = pos!(x X_BAR:1, 12:1; o 20:2, 10:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected = pos!(x 23:1, 9:1; o 20:2, 10:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn only_entering_with_bigger_die_allows_two_checkers_to_move() {
        // Given
        let position = pos!(x X_BAR:1, 12:1; o 20:2, 9:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected = pos!(x 22:1, 10:1; o 20:2, 9:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn entering_with_either_die_allowed_but_only_one_final_position() {
        // Given
        let position = pos!(x X_BAR:1; o 9:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected = pos!(x 20:1; o 9:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn final_position_but_different_move_because_die1_hits_opponent() {
        // Given
        let position = pos!(x X_BAR:1; o 22:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected1 = pos!(x 20:1; o O_BAR:1);
        let expected2 = pos!(x 20:1; o 22:1);
        assert_eq!(resulting_positions, vec![expected1, expected2]);
    }

    #[test]
    fn final_position_but_different_move_because_die2_hits_opponent() {
        // Given
        let position = pos!(x X_BAR:1; o 23:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 2));
        // Then
        let expected1 = pos!(x 20:1; o 23:1);
        let expected2 = pos!(x 20:1; o O_BAR:1);
        assert_eq!(resulting_positions, vec![expected1, expected2]);
    }

    // No checkers on bar

    #[test]
    fn cannot_user_either_die() {
        // Given
        let position = pos!(x 10:2, 2:3; o 8:2, 6:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 2));
        // Then
        assert_eq!(resulting_positions, vec![position]);
    }

    #[test]
    fn forced_only_smaller_die() {
        // Given
        let position = pos!(x 7:2; o 2:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 2));
        // Then
        let expected = pos!(x 7:1, 5:1; o 2:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn forced_smaller_die_first_then_bear_off() {
        // Given
        let position = pos!(x 8:1, 4:3; o 1:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 3));
        // Then
        let expected = pos!(x 5:1, 4:2; o 1:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn bigger_die_cannot_move_initially() {
        // Given
        let position = pos!(x 20:1; o 16:3);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 3));
        // Then
        let expected = pos!(x 13:1; o 16:3);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn smaller_first_allows_bear_off() {
        // Given
        let position = pos!(x 9:1, 5:1; o 20:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 3));
        // Then
        let expected1 = pos!(x 4:1, 2:1; o 20:2);
        let expected2 = pos!(x 5:1, 1:1; o 20:2);
        let expected3 = pos!(x 6:1; o 20:2);
        assert_eq!(resulting_positions, vec![expected1, expected2, expected3]);
    }

    #[test]
    fn could_bear_off_but_could_do_other_moves_as_well() {
        // Given
        let position = pos!(x 5:2, 4:3, 3:1; o 20:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 3));
        // Then
        let expected1 = pos!(x 4:3, 3:1, 2:1, 1:1; o 20:1);
        let expected2 = pos!(x 5:1, 4:2, 3:1, 1:2; o 20:1);
        let expected3 = pos!(x 5:1, 4:3, 1:1; o 20:1);
        let expected4 = pos!(x 5:1, 4:2, 3:1, 2:1; o 20:1);
        let expected5 = pos!(x 5:2, 4:1, 3:1, 1:1; o 20:1);
        let expected6 = pos!(x 5:2, 4:2; o 20:1);
        assert_eq!(
            resulting_positions,
            vec![expected1, expected2, expected3, expected4, expected5, expected6]
        );
    }

    #[test]
    fn only_one_move_if_order_is_not_important() {
        // Given
        let position = pos!(x 20:1; o 22:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 3));
        // Then
        let expected = pos!(x 13:1; o 22:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn only_one_move_in_home_board_if_order_is_not_important() {
        // Given
        let position = pos!(x 5:1; o 22:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(2, 1));
        // Then
        let expected = pos!(x 2:1; o 22:2);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn two_moves_if_bigger_die_hits_opponent() {
        // Given
        let position = pos!(x 10:1; o 6:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(4, 2));
        // Then
        let expected1 = pos!(x 4:1; o O_BAR:1);
        let expected2 = pos!(x 4:1; o 6:1);
        assert_eq!(resulting_positions, vec![expected1, expected2]);
    }

    #[test]
    fn two_moves_if_smaller_die_hits_opponent() {
        // Given
        let position = pos!(x 5:1; o 4:1);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(3, 1));
        // Then
        let expected1 = pos!(x 1:1; o 4:1);
        let expected2 = pos!(x 1:1; o O_BAR:1);
        assert_eq!(resulting_positions, vec![expected1, expected2]);
    }

    #[test]
    fn two_bear_offs_from_same_pip() {
        // Given
        let position = pos!(x 1:5; o 24:8);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(6, 4));
        // Then
        let expected = pos!(x 1:3; o 24:8);
        assert_eq!(resulting_positions, vec![expected]);
    }

    #[test]
    fn bear_off_from_same_pip_with_either_big_or_small_die() {
        // Given
        let position = pos!(x 2:1, 1:5; o);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(6, 1));
        // Then
        let expected1 = pos!(x 1:4; o);
        let expected2 = pos!(x 1:5; o);
        assert_eq!(resulting_positions, vec![expected1, expected2]);
    }

    #[test]
    fn use_smaller_die_from_bigger_pip() {
        // Given
        let position = pos!(x 7:1, 6:3; o 2:2);
        // When
        let resulting_positions =
            position.all_positions_after_regular_move(&RegularDice::new(5, 4));
        // Then
        let expected = pos!(x 6:2, 3:1, 1:1; o 2:2);
        assert_eq!(resulting_positions, vec![expected]);
    }
}
