use criterion::{criterion_group, criterion_main, Criterion};
use engine::dice::ALL_21;
use engine::pos;
use engine::position::GameState::Ongoing;
use engine::position::{Position, O_BAR, STARTING, X_BAR};
use std::collections::HashMap;
use std::hint::black_box;

/// A tuple is returned, `value.0` is the how often `all_positions_after_moving` has been called.
/// `value.1` is the number of positions returned by that. Not only leave nodes are taken into
/// account, but all. So for a depth of `0`, the number of positions is already `1`, representing
/// the initial position.
fn generations_and_positions(position: &Position, depth: usize) -> (usize, usize) {
    if depth == 0 || position.game_state() != Ongoing {
        (0, 1)
    } else {
        let result = ALL_21
            .iter()
            .flat_map(|(dice, _)| position.all_positions_after_moving(dice))
            .map(|new_pos: Position| generations_and_positions(&new_pos, depth - 1))
            .fold((21, 1), |acc, e| (acc.0 + e.0, acc.1 + e.1));
        result
    }
}

fn from_bar(c: &mut Criterion) {
    let position = pos!(x 1:1, 3:2, 4:3, 5:2, 10:1, 20:2, 23:1, X_BAR:3; o O_BAR: 3, 2:2, 6:1, 17:1, 18:2, 19:3, 21:2, 22:1);
    let max_depth = 3;

    println!("depth: (# positions, # move generation calls)");
    for depth in 0..=max_depth {
        println!(
            "{}: {:?}",
            depth,
            generations_and_positions(&position, depth)
        );
    }

    c.bench_function("number_of_generations_and_positions: from_bar", |b| {
        b.iter(|| generations_and_positions(black_box(&position), black_box(max_depth)))
    });
}

fn bearoff(c: &mut Criterion) {
    let position = pos!(x 1:4, 3:2, 4:1, 5:5; o 24:4, 23:1, 21:5, 19:3);
    let max_depth = 2;

    println!("depth: (# positions, # move generation calls)");
    for depth in 0..=max_depth {
        println!(
            "{}: {:?}",
            depth,
            generations_and_positions(&position, depth)
        );
    }

    c.bench_function("number_of_generations_and_positions: bearoff", |b| {
        b.iter(|| generations_and_positions(black_box(&position), black_box(max_depth)))
    });
}

fn starting(c: &mut Criterion) {
    let position = STARTING;
    let max_depth = 2;

    println!("depth: (# positions, # move generation calls)");
    for depth in 0..=max_depth {
        println!(
            "{}: {:?}",
            depth,
            generations_and_positions(&position, depth)
        );
    }

    c.bench_function("number_of_generations_and_positions: STARTING", |b| {
        b.iter(|| generations_and_positions(black_box(&position), black_box(max_depth)))
    });
}

criterion_group!(benches, starting, from_bar, bearoff);
criterion_main!(benches);
