use engine::probabilities::Probabilities;
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
/// Information about proper cube decisions. Currently quick and dirty calculations.
pub struct CubeInfo {
    /// `true` if the player `x` should double, `false` if no double yet or too good.
    double: bool,
    /// `true` if the opponent should take the cube, `false` if they should reject.
    accept: bool,
}

impl From<&Probabilities> for CubeInfo {
    fn from(value: &Probabilities) -> Self {
        // This is just a very simple calculation so that we can implement the cube API.
        // Later we want better cube decisions, helpful could be the article:
        // https://bkgm.com/articles/Janowski/cubeformulae.pdf
        let equity = value.equity();
        let double = equity > 0.4 && equity < 0.6;
        let accept = equity < 0.5;
        Self { double, accept }
    }
}
