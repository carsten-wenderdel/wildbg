use core::ffi::*;
use engine::complex::ComplexEvaluator;
use engine::dice::Dice;
use engine::position::Position;
use logic::bg_move::{BgMove, MoveDetail};
use logic::wildbg_api::{WildbgApi, WildbgConfig};

// When this file is changed, recreate the header file by executing this from the project's root:
// touch cbindgen.toml
// cbindgen --config cbindgen.toml --crate wildbg-c --output crates/wildbg-c/wildbg.h --lang c
// rm cbindgen.toml

// For more infos about Rust -> C see
// https://docs.rust-embedded.org/book/interoperability/rust-with-c.html
// http://jakegoulding.com/rust-ffi-omnibus/objects/

// Wrap the WildbgApi into a new struct, so that we don't have to expose the ComplexEvaluator
pub struct Wildbg {
    api: WildbgApi<ComplexEvaluator>,
}

/// Configuration needed for the evaluation of positions.
///
/// Currently only 1 pointers and money game are supported.
/// In the future `BgConfig` can also include information about Crawford, strength of the engine and so on.
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

type Error = &'static str;

#[no_mangle]
/// Loads the neural nets into memory and returns a pointer to the API.
///
/// To free the memory after usage, call `wildbg_free`.
pub extern "C" fn wildbg_new() -> *mut Wildbg {
    if let Some(api) = WildbgApi::try_default() {
        Box::into_raw(Box::new(Wildbg { api }))
    } else {
        std::ptr::null_mut()
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
