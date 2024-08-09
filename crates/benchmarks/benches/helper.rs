use engine::position::Position;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn positions_from_file(file: File) -> Vec<Position> {
    BufReader::new(file)
        .lines()
        .map(|l| Position::from_id(l.expect("Could not parse line")))
        .collect()
}
pub fn contact_positions() -> Vec<Position> {
    let file = File::open("resources/contact.csv").unwrap();
    let positions = positions_from_file(file);
    assert_eq!(positions.len(), 1_000);
    positions
}

#[allow(dead_code)]
pub fn race_positions() -> Vec<Position> {
    let file = File::open("resources/race.csv").unwrap();
    let positions = positions_from_file(file);
    assert_eq!(positions.len(), 1_000);
    positions
}
