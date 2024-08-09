use crate::helper::contact_positions;
use criterion::{criterion_group, criterion_main, Criterion};
use engine::dice::Dice;
use engine::position::Position;
use mimalloc::MiMalloc;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod helper;

// This file contains benchmarks for functions operating on a `Position`.

fn switch_sides(c: &mut Criterion) {
    let mut positions: Vec<Position> = Vec::with_capacity(200_000);

    for position in contact_positions() {
        for dice in &Dice::all_15_mixed() {
            positions.extend(position.all_positions_after_moving(dice));
        }
    }
    assert!(positions.len() > 150_000);

    c.bench_function("Switch sides of positions", |b| {
        b.iter(|| {
            black_box(&mut positions)
                .iter_mut()
                .for_each(|p| *p = p.sides_switched());
        })
    });
}

criterion_group!(benches, switch_sides);
criterion_main!(benches);
