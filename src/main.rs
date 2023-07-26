use crate::position::Position;
use std::collections::HashMap;
mod dice_gen;
mod position;

fn main() {
    // Just some random calls to make sure everything is public that needs to be.
    let position = pos!(x 20:2; o 16:3);
    let moves = position.all_positions_after_moving(3, 3);
    println!("Number of moves: {}", moves.len());
}
