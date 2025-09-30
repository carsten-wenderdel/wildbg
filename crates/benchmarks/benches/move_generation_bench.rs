use crate::helper::{contact_positions, race_positions};
use criterion::{Criterion, criterion_group, criterion_main};
use engine::dice::Dice;
use engine::position::Position;
use mimalloc::MiMalloc;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod helper;

// This file contains benchmarks for generating moves/positions for a given position and dice.

// Helper Methods

fn number_double_moves(positions: &[Position]) -> usize {
    number_moves(positions, &Dice::all_6_double())
}

fn number_mixed_moves(positions: &[Position]) -> usize {
    number_moves(positions, &Dice::all_15_mixed())
}

#[inline]
fn number_moves(positions: &[Position], dice: &[Dice]) -> usize {
    dice.iter()
        .map(|dice| {
            positions
                .iter()
                .map(|position| position.all_positions_after_moving(dice).len())
                .sum::<usize>()
        })
        .sum()
}

// Benchmark methods

#[allow(dead_code)]
fn contact_double(c: &mut Criterion) {
    let positions = contact_positions();

    let number = number_double_moves(&positions);
    assert_eq!(number, 218_356);
    println!("1000 contact positions, 6 double dice rolls -> {number} generated moves.");

    c.bench_function("generate moves: contact, double", |b| {
        b.iter(|| number_double_moves(black_box(&positions)))
    });
}

#[allow(dead_code)]
fn contact_mixed(c: &mut Criterion) {
    let positions = contact_positions();

    let number = number_mixed_moves(&positions);
    assert_eq!(number, 185_957);
    println!("1000 contact positions, 15 mixed dice rolls -> {number} generated moves.");

    c.bench_function("generate moves: contact, mixed", |b| {
        b.iter(|| number_mixed_moves(black_box(&positions)))
    });
}

#[allow(dead_code)]
fn race_double(c: &mut Criterion) {
    let positions = race_positions();

    let number = number_double_moves(&positions);
    assert_eq!(number, 140_720);
    println!("1000 race positions, 6 double dice rolls -> {number} generated moves.");

    c.bench_function("generate moves: race, double", |b| {
        b.iter(|| number_double_moves(black_box(&positions)))
    });
}

#[allow(dead_code)]
fn race_mixed(c: &mut Criterion) {
    let positions = race_positions();

    let number = number_mixed_moves(&positions);
    assert_eq!(number, 122_570);
    println!("1000 race positions, 15 mixed dice rolls -> {number} generated moves.");

    c.bench_function("generate moves: race, mixed", |b| {
        b.iter(|| number_mixed_moves(black_box(&positions)))
    });
}

criterion_group!(
    benches,
    contact_double,
    race_double,
    contact_mixed,
    race_mixed
);
criterion_main!(benches);
