use core::ffi::*;
use engine::composite::CompositeEvaluator;
use engine::dice::Dice;
use engine::position::Position;
use engine::probabilities::Probabilities;
use logic::bg_move::{BgMove, MoveDetail};
use logic::wildbg_api::{WildbgApi, WildbgConfig};

// When this file is changed, recreate the header file by executing this from the project's root:
// touch cbindgen.toml
// cbindgen --config cbindgen.toml --crate wildbg-c --output crates/wildbg-c/wildbg.h --lang c
// rm cbindgen.toml

// For more infos about Rust -> C see
// https://docs.rust-embedded.org/book/interoperability/rust-with-c.html
// http://jakegoulding.com/rust-ffi-omnibus/objects/

// Wrap the WildbgApi into a new struct, so that we don't have to expose the CompositeEvaluator
pub struct Wildbg {
    api: WildbgApi<CompositeEvaluator>,
}

/// Configuration needed for the evaluation of positions.
///
/// Currently only 1 pointers and money game are supported.
/// In the future `BgConfig` can also include information about Crawford, cube possession, strength of the engine and so on.
#[repr(C)]
pub struct BgConfig {
    /// Number of points the player on turn needs to finish the match. Zero indicates money game.
    pub x_away: c_uint,
    /// Number of points the opponent needs to finish the match. Zero indicates money game.
    pub o_away: c_uint,
}

impl From<&BgConfig> for WildbgConfig {
    fn from(value: &BgConfig) -> Self {
        if value.x_away == 0 && value.o_away == 0 {
            Self { away: None }
        } else {
            Self {
                away: Some((value.x_away, value.o_away)),
            }
        }
    }
}

#[no_mangle]
/// Loads the neural nets into memory and returns a pointer to the API.
/// Returns `NULL` if the neural nets cannot be found.
///
/// To free the memory after usage, call `wildbg_free`.
pub extern "C" fn wildbg_new() -> *mut Wildbg {
    match WildbgApi::try_default() {
        Ok(api) => Box::into_raw(Box::new(Wildbg { api })),
        Err(message) => {
            eprintln!("{message}");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
/// # Safety
///
/// Frees the memory of the argument.
/// Don't call it with a NULL pointer. Don't call it more than once for the same `Wildbg` pointer.
pub unsafe extern "C" fn wildbg_free(ptr: *mut Wildbg) {
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[repr(C)]
#[derive(Default)]
pub struct CProbabilities {
    /// Cubeless probability to win the game. This includes gammons and backgammons.
    win: c_float,
    /// Probability to win gammon or backgammon.
    win_g: c_float,
    /// Probability to win backgammon.
    win_bg: c_float,
    /// Probability to lose gammon or backgammon.
    lose_g: c_float,
    /// Probability to lose backgammon.
    lose_bg: c_float,
}

impl From<&Probabilities> for CProbabilities {
    fn from(value: &Probabilities) -> Self {
        Self {
            win: value.win_normal + value.win_gammon + value.win_bg,
            win_g: value.win_gammon + value.win_bg,
            win_bg: value.win_bg,
            lose_g: value.lose_gammon + value.lose_bg,
            lose_bg: value.lose_bg,
        }
    }
}

/// When no move is possible, detail_count will be 0.
///
/// If only a single move is possible, `details[0]` will contain this information.
/// `detail_count` will contain a value between 0 and 4.
///
/// If the same checker is moved twice, this is encoded in two details.
#[repr(C)]
#[derive(Default)]
pub struct CMove {
    details: [CMoveDetail; 4],
    detail_count: c_int,
}

impl From<BgMove> for CMove {
    fn from(value: BgMove) -> Self {
        let details = value.into_details();
        let mut c_move = CMove::default();
        details.iter().enumerate().for_each(|(i, detail)| {
            c_move.details[i] = detail.into();
        });
        c_move.detail_count = details.len() as c_int;
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

type Error = &'static str;

/// Returns the best move for the given position.
///
/// The player on turn always moves from pip 24 to pip 1.
/// The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
/// Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
#[no_mangle]
pub extern "C" fn best_move(
    wildbg: &Wildbg,
    pips: &[c_int; 26],
    die1: c_uint,
    die2: c_uint,
    config: &BgConfig,
) -> CMove {
    let pips = pips.map(|pip| pip as i8);
    let move_result = || -> Result<BgMove, Error> {
        let position = Position::try_from(pips)?;
        let dice = Dice::try_from((die1 as usize, die2 as usize))?;
        let bg_move = wildbg
            .api
            .best_move(&position, &dice, &WildbgConfig::from(config));
        Ok(bg_move)
    };
    match move_result() {
        Ok(bg_move) => CMove::from(bg_move),
        Err(error) => {
            eprintln!("{}", error);
            CMove::default()
        }
    }
}

/// Returns cubeless money game probabilities for a certain position.
/// If an illegal position is encountered, all probabilities will be zero.
///
/// The player on turn always moves from pip 24 to pip 1.
/// The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
/// Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
#[no_mangle]
pub extern "C" fn probabilities(wildbg: &Wildbg, pips: &[c_int; 26]) -> CProbabilities {
    let pips = pips.map(|pip| pip as i8);
    match Position::try_from(pips) {
        Ok(position) => (&wildbg.api.probabilities(&position)).into(),
        Err(error) => {
            eprintln!("{}", error);
            CProbabilities::default()
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::CProbabilities;
    use engine::position::X_BAR;
    use engine::{dice::Dice, pos};

    #[test]
    fn from_probabilities() {
        let model_probs = engine::probabilities::Probabilities {
            win_normal: 0.32,
            win_gammon: 0.26,
            win_bg: 0.12,
            lose_normal: 0.15,
            lose_gammon: 0.1,
            lose_bg: 0.05,
        };

        let c_probs: CProbabilities = (&model_probs).into();
        assert_eq!(c_probs.win, 0.7);
        assert_eq!(c_probs.win_g, 0.38);
        assert_eq!(c_probs.win_bg, 0.12);
        assert_eq!(c_probs.lose_g, 0.15);
        assert_eq!(c_probs.lose_bg, 0.05);
    }

    #[test]
    fn from_bgmove_mixed_dice() {
        let old = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        let new = pos!(x 23: 1, 22:1, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        let dice = Dice::new(2, 1);
        let bg_move = logic::bg_move::BgMove::new(&old, &new, &dice);
        let c_move = crate::CMove::from(bg_move);
        assert_eq!(c_move.detail_count, 2);
        assert_eq!(c_move.details[0].from, 24);
        assert_eq!(c_move.details[0].to, 22);
        assert_eq!(c_move.details[1].from, 24);
        assert_eq!(c_move.details[1].to, 23);
    }

    #[test]
    fn from_bgmove_double_dice() {
        let old = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        let new = pos!(x 18:2, 13:3, 7:2, 8:3, 6:5; o 19:5, 17:3, 12:5, 1:2);
        let dice = Dice::new(6, 6);
        let bg_move = logic::bg_move::BgMove::new(&old, &new, &dice);
        let c_move = crate::CMove::from(bg_move);
        assert_eq!(c_move.detail_count, 4);
        assert_eq!(c_move.details[0].from, 24);
        assert_eq!(c_move.details[0].to, 18);
        assert_eq!(c_move.details[1].from, 24);
        assert_eq!(c_move.details[1].to, 18);
        assert_eq!(c_move.details[2].from, 13);
        assert_eq!(c_move.details[2].to, 7);
        assert_eq!(c_move.details[3].from, 13);
        assert_eq!(c_move.details[3].to, 7);
    }

    #[test]
    fn from_bgmove_nomoves() {
        let pos = pos!(x X_BAR:15; o 24:3, 23:3, 22:3, 21:2, 20: 2, 19: 2);
        let dice = Dice::new(1, 1);
        let bg_move = logic::bg_move::BgMove::new(&pos, &pos, &dice);
        let c_move = crate::CMove::from(bg_move);
        assert_eq!(c_move.detail_count, 0);
    }
}
