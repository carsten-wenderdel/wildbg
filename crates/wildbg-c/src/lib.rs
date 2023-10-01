use core::ffi::*;
use logic::bg_move::{BgMove, MoveDetail};

// When this file is changed, recreate the header file by executing this from the project's root:
// touch cbindgen.toml
// cbindgen --config cbindgen.toml --crate wildbg-c --output crates/wildbg-c/wildbg.h --lang c
// rm cbindgen.toml

/// When no move is possible, all member variables in all details will be `-1`.
///
/// If only one checker can be moved once, `detail1` will contain this information,
/// `detail2`, `detail3` and `detail4` will contain `-1` for both `from` and `to`.
///
/// If the same checker is moved twice, this is encoded in two details.
#[repr(C)]
#[derive(Default)]
pub struct CMove {
    detail1: CMoveDetail,
    detail2: CMoveDetail,
    detail3: CMoveDetail,
    detail4: CMoveDetail,
}

impl From<BgMove> for CMove {
    fn from(value: BgMove) -> Self {
        let details = value.into_details();
        let mut c_move = CMove::default();
        #[allow(clippy::len_zero)]
        if details.len() > 0 {
            c_move.detail1 = (&details[0]).into();
        }
        if details.len() > 1 {
            c_move.detail2 = (&details[1]).into();
        }
        if details.len() > 2 {
            c_move.detail3 = (&details[2]).into();
        }
        if details.len() > 3 {
            c_move.detail4 = (&details[3]).into();
        }
        c_move
    }
}

/// If the move is not possible, both `from` and `to` will contain `-1`.
///
/// If the move is possible, `from` is an integer between 25 and 1,
/// `to` is an integer between 24 and 0.
/// `from - to` is then at least 1 and at most 6.
#[repr(C)]
pub struct CMoveDetail {
    from: c_int,
    to: c_int,
}

impl Default for CMoveDetail {
    fn default() -> Self {
        Self { from: -1, to: -1 }
    }
}

impl From<&MoveDetail> for CMoveDetail {
    fn from(value: &MoveDetail) -> Self {
        CMoveDetail {
            from: value.from() as c_int,
            to: value.to() as c_int,
        }
    }
}

/// Returns the best move for the given position.
///
/// The player on turn always moves from pip 24 to pip 1.
/// The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
/// Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
#[no_mangle]
pub extern "C" fn best_move_1ptr(pips: &[c_int; 26], die1: c_uint, die2: c_uint) -> CMove {
    let pips = pips.map(|pip| pip as i8);
    match logic::best_move_1ptr(pips, die1 as u8, die2 as u8) {
        Ok(bg_move) => CMove::from(bg_move),
        Err(_) => CMove::default(),
    }
}
