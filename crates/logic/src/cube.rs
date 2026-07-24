use engine::probabilities::Probabilities;
#[cfg(feature = "web")]
use serde::Serialize;
#[cfg(feature = "web")]
use utoipa::ToSchema;

/// Cube efficiency (also called "cube-life index") as suggested by Janowski.
/// `0.0` would model a completely dead cube, `1.0` a fully live (continuous) cube.
/// `2/3` is Janowski's recommended value for a typical money game.
const CUBE_EFFICIENCY: f32 = 2.0 / 3.0;

#[cfg_attr(feature = "web", derive(Serialize, ToSchema))]
#[cfg_attr(feature = "web", serde(rename_all = "camelCase"))]
/// Money game cube decisions based on Janowski's cube formulae.
///
/// See <https://bkgm.com/articles/Janowski/cubeformulae.pdf>.
///
/// As the API currently carries no information about cube ownership or match
/// score, all values assume a money game with a centered cube (an initial
/// double decision). Match play and cube ownership are future extensions.
pub struct CubeInfo {
    /// `true` if the player `x` should double, `false` if no double yet or too good.
    double: bool,
    /// `true` if the opponent should take the cube, `false` if they should reject.
    accept: bool,
    /// Cubeless money game equity of the position, from player `x`'s perspective.
    equity_cubeless: f32,
    /// Cubeful equity if player `x` does not double (centered cube), from `x`'s perspective.
    equity_no_double: f32,
    /// Cubeful equity if player `x` doubles and the opponent takes, from `x`'s perspective.
    /// The stake is already doubled, so this is comparable to the other equities.
    equity_double_take: f32,
}

impl From<&Probabilities> for CubeInfo {
    fn from(value: &Probabilities) -> Self {
        let x = CUBE_EFFICIENCY;

        // Probability of winning and losing (cubeless).
        let p = value.win();
        let q = 1.0 - p;

        // Average points won given a win (`w`) and lost given a loss (`l`).
        // Both are positive and lie in `[1, 3]`. Guard against division by zero
        // for certain wins or losses, in which case that side never happens.
        let w = if p > 0.0 {
            (value.win_normal + 2.0 * value.win_gammon + 3.0 * value.win_bg) / p
        } else {
            0.0
        };
        let l = if q > 0.0 {
            (value.lose_normal + 2.0 * value.lose_gammon + 3.0 * value.lose_bg) / q
        } else {
            0.0
        };

        // Janowski's cubeful equities from `x`'s perspective for a cube of value 1.
        let common = p * (w + l + 0.5 * x) - l;
        // Centered cube: neither player has doubled yet.
        let equity_no_double = (4.0 / (4.0 - x)) * (common - 0.25 * x);
        // Opponent owns the cube after `x` doubles and the opponent takes.
        let equity_opponent_owns = common - 0.5 * x;
        // After a double and take the stake is doubled.
        let equity_double_take = 2.0 * equity_opponent_owns;

        // The opponent takes when doing so is better for them than dropping.
        // Dropping costs them the current stake (`x`'s equity would be +1.0).
        let accept = equity_double_take < 1.0;
        // `x` doubles when doubling – after the opponent's best response of
        // take or drop – beats keeping the cube in the center. Positions that
        // are "too good" to double yield `equity_no_double > 1.0` and thus `false`.
        let double = equity_double_take.min(1.0) > equity_no_double;

        Self {
            double,
            accept,
            equity_cubeless: value.equity(),
            equity_no_double,
            equity_double_take,
        }
    }
}

impl CubeInfo {
    pub fn double(&self) -> bool {
        self.double
    }
    pub fn accept(&self) -> bool {
        self.accept
    }
    pub fn equity_cubeless(&self) -> f32 {
        self.equity_cubeless
    }
    pub fn equity_no_double(&self) -> f32 {
        self.equity_no_double
    }
    pub fn equity_double_take(&self) -> f32 {
        self.equity_double_take
    }
}

#[cfg(test)]
mod tests {
    use super::CubeInfo;
    use engine::probabilities::Probabilities;

    /// Helper for a position without gammons or backgammons and a given win probability.
    fn no_gammons(win: f32) -> Probabilities {
        Probabilities {
            win_normal: win,
            win_gammon: 0.0,
            win_bg: 0.0,
            lose_normal: 1.0 - win,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        }
    }

    #[test]
    fn symmetric_position_is_neutral() {
        // Given a completely symmetric position (50% wins, no gammons)
        let cube = CubeInfo::from(&no_gammons(0.5));
        // Then equities are zero, there is no double and a take is correct.
        assert!(cube.equity_cubeless().abs() < 1e-6);
        assert!(cube.equity_no_double().abs() < 1e-6);
        assert!(!cube.double());
        assert!(cube.accept());
    }

    #[test]
    fn double_take_window() {
        // Given a clear advantage that is still takeable (around 70% wins)
        let cube = CubeInfo::from(&no_gammons(0.70));
        // Then the player should double and the opponent should take.
        assert!(cube.double());
        assert!(cube.accept());
    }

    #[test]
    fn too_strong_to_be_taken() {
        // Given a huge racing lead (85% wins, no gammons)
        let cube = CubeInfo::from(&no_gammons(0.85));
        // Then the player should double but the opponent must drop.
        assert!(cube.double());
        assert!(!cube.accept());
    }

    #[test]
    fn no_double_when_barely_ahead() {
        // Given only a small advantage (55% wins)
        let cube = CubeInfo::from(&no_gammons(0.55));
        // Then it is too early to double, but a take would be trivial.
        assert!(!cube.double());
        assert!(cube.accept());
    }

    #[test]
    fn too_good_to_double() {
        // Given a position that wins a lot of gammons and backgammons,
        // playing on is worth more than a single point from doubling out.
        let probs = Probabilities {
            win_normal: 0.15,
            win_gammon: 0.55,
            win_bg: 0.2,
            lose_normal: 0.1,
            lose_gammon: 0.0,
            lose_bg: 0.0,
        };
        let cube = CubeInfo::from(&probs);
        // Then the player is too good to double (playing on beats cashing).
        assert!(cube.equity_no_double() > 1.0);
        assert!(!cube.double());
    }
}
