use engine::dice::Dice;
use engine::position::Position;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn positions_from_file(file: File) -> Vec<Position> {
    BufReader::new(file)
        .lines()
        .map(|l| Position::from_id(&l.expect("Could not parse line")))
        .collect()
}

pub fn contact_positions() -> Vec<Position> {
    let file = File::open("resources/contact.csv").unwrap();
    let positions = positions_from_file(file);
    assert_eq!(positions.len(), 1_000);
    positions
}

#[allow(dead_code)]
/// Roughly 800 Vecs containing 2 to roughly 250 positions.
/// This simulates the number of positions we have to evaluate during rollout.
pub fn contact_positions_after_moving() -> Vec<Vec<Position>> {
    positions_after_moving(contact_positions())
}

#[allow(dead_code)]
pub fn race_positions() -> Vec<Position> {
    let file = File::open("resources/race.csv").unwrap();
    let positions = positions_from_file(file);
    assert_eq!(positions.len(), 1_000);
    positions
}

#[allow(dead_code)]
/// Roughly 800 Vecs containing 2 to roughly 250 positions.
/// This simulates the number of positions we have to evaluate during rollout.
pub fn race_positions_after_moving() -> Vec<Vec<Position>> {
    positions_after_moving(race_positions())
}

#[allow(dead_code)]
fn positions_after_moving(positions: Vec<Position>) -> Vec<Vec<Position>> {
    // We will never use evaluation when there is only one legal move - so let's filter that away.
    positions
        .into_iter()
        .zip(Dice::all_36().iter().cycle())
        .map(|(p, dice)| p.all_positions_after_moving(dice))
        .filter(|p| p.len() > 1)
        .collect()
}
