use crate::position::Position;
use std::cmp::max;

mod regular;

#[allow(dead_code)]
struct BgMove {
    details: Vec<MoveDetail>,
}

#[derive(Debug, PartialEq)]
struct MoveDetail {
    from: usize,
    to: usize,
}

impl BgMove {
    /// Finds up to two pips where `more[pip] > less[pip]`.
    /// Only looks at positive pips, so it ignores the opponent.
    /// If `less[pip]` is negative (the opponent had a checker there), it will treat at 0, not as -1.
    fn more_checkers(more: &Position, less: &Position) -> [Option<usize>; 2] {
        let mut from: [Option<usize>; 2] = [None; 2];
        let mut from_index = 0_usize;
        for i in (1..26).rev() {
            match more.pip(i) - max(0, less.pip(i)) {
                2 => return [Some(i), Some(i)],
                1 => {
                    from[from_index] = Some(i);
                    from_index += 1;
                }
                _ => {}
            }
        }
        from
    }
}
