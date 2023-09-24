use crate::position::GameResult::*;
use std::fmt;
use std::fmt::Formatter;

/// Sum of all six fields will always be 1.0
#[derive(PartialEq)]
pub struct Probabilities {
    pub(crate) win_normal: f32,
    pub(crate) win_gammon: f32,
    pub(crate) win_bg: f32,
    pub(crate) lose_normal: f32,
    pub(crate) lose_gammon: f32,
    pub(crate) lose_bg: f32,
}

impl fmt::Debug for Probabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Probabilities: wn {:.2}%; wg {:.2}%; wb {:.2}%; ln {:.2}%; lg {:.2}%; lb {:.2}%",
            100.0 * self.win_normal,
            100.0 * self.win_gammon,
            100.0 * self.win_bg,
            100.0 * self.lose_normal,
            100.0 * self.lose_gammon,
            100.0 * self.lose_bg
        )
    }
}

/// Used when writing CSV data to a file
impl fmt::Display for Probabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{};{};{};{};{};{}",
            self.win_normal,
            self.win_gammon,
            self.win_bg,
            self.lose_normal,
            self.lose_gammon,
            self.lose_bg
        )
    }
}

impl Probabilities {
    pub fn csv_header() -> String {
        "win_normal;win_gammon;win_bg;lose_normal;lose_gammon;lose_bg".to_string()
    }

    /// Typically used from rollouts.
    /// The index within the array has to correspond to the discriminant of the `Probabilities` enum.
    /// Input integer values will be normalized so that the sum in the return value is 1.0
    pub fn new(results: &[u32; 6]) -> Self {
        let sum = results.iter().sum::<u32>() as f32;
        Probabilities {
            win_normal: results[WinNormal as usize] as f32 / sum,
            win_gammon: results[WinGammon as usize] as f32 / sum,
            win_bg: results[WinBg as usize] as f32 / sum,
            lose_normal: results[LoseNormal as usize] as f32 / sum,
            lose_gammon: results[LoseGammon as usize] as f32 / sum,
            lose_bg: results[LoseBg as usize] as f32 / sum,
        }
    }

    pub(crate) fn switch_sides(&self) -> Self {
        Self {
            win_normal: self.lose_normal,
            win_gammon: self.lose_gammon,
            win_bg: self.lose_bg,
            lose_normal: self.win_normal,
            lose_gammon: self.win_gammon,
            lose_bg: self.win_bg,
        }
    }

    /// Cubeless equity
    pub fn equity(&self) -> f32 {
        self.win_normal - self.lose_normal
            + 2.0 * (self.win_gammon - self.lose_gammon)
            + 3.0 * (self.win_bg - self.lose_bg)
    }
}

#[cfg(test)]
mod tests {
    use crate::probabilities::Probabilities;

    #[test]
    fn new() {
        // sum of `results is 32, a power of 2. Makes fractions easier to handle.
        let results = [0_u32, 1, 3, 4, 8, 16];
        let probabilities = Probabilities::new(&results);
        assert_eq!(probabilities.win_normal, 0.0);
        assert_eq!(probabilities.win_gammon, 0.03125);
        assert_eq!(probabilities.win_bg, 0.09375);
        assert_eq!(probabilities.lose_normal, 0.125);
        assert_eq!(probabilities.lose_gammon, 0.25);
        assert_eq!(probabilities.lose_bg, 0.5);
    }

    #[test]
    fn to_string() {
        let probabilities = Probabilities {
            win_normal: 1.0 / 21.0,
            win_gammon: 2.0 / 21.0,
            win_bg: 3.0 / 21.0,
            lose_normal: 4.0 / 21.0,
            lose_gammon: 5.0 / 21.0,
            lose_bg: 6.0 / 21.0,
        };
        assert_eq!(
            probabilities.to_string(),
            "0.04761905;0.0952381;0.14285715;0.1904762;0.23809524;0.2857143"
        );
    }

    #[test]
    fn equity_win_normal() {
        let probabilities = Probabilities {
            win_normal: 1.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 1.0);
    }

    #[test]
    fn equity_win_gammon() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 1.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 2.0);
    }

    #[test]
    fn equity_win_bg() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 1.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), 3.0);
    }

    #[test]
    fn equity_lose_normal() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 1.0,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), -1.0);
    }

    #[test]
    fn equity_lose_gammon() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 1.0,
            lose_bg: 0.0,
        };
        assert_eq!(probabilities.equity(), -2.0);
    }

    #[test]
    fn equity_lose_bg() {
        let probabilities = Probabilities {
            win_normal: 0.0,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 0.0,
            lose_gammon: 0.0,
            lose_bg: 1.0,
        };
        assert_eq!(probabilities.equity(), -3.0);
    }

    #[test]
    fn equity_balanced() {
        let probabilities = Probabilities {
            win_normal: 0.3,
            win_gammon: 0.1,
            win_bg: 0.1,
            lose_normal: 0.3,
            lose_gammon: 0.1,
            lose_bg: 0.1,
        };
        assert_eq!(probabilities.equity(), 0.0);
    }
}
