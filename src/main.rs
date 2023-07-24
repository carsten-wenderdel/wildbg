use crate::position::Position;
use std::collections::HashMap;
mod position;

fn main() {
    // Just some random calls to make sure everything is public that needs to be.
    let position = Position::from(&HashMap::from([(20, 2)]), &HashMap::from([(16, 3)]));
    let moves = position.all_positions_after_moving(3, 3);
    println!("Number of moves: {}", moves.len());
}
