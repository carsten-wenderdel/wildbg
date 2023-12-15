use crate::helper::{contact_positions, race_positions};
use criterion::{criterion_group, criterion_main, Criterion};
use engine::inputs::{ContactInputsGen, InputsGen, RaceInputsGen};
use engine::position::Position;
use mimalloc::MiMalloc;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod helper;

// This file contains benchmarks for generating neural net inputs for a given position.

// Helper Methods

fn sum_of_all_inputs<T: InputsGen>(positions: &[Position], inputs_gen: &T) {
    // Create 20 inputs at once. Should have roughly the same size as number of legal moves from a random position.
    let batch_size = 20;
    positions.chunks_exact(batch_size).for_each(|chunk| {
        inputs_gen.inputs_for_all(chunk);
    });
}

// Benchmark methods

#[allow(dead_code)]
fn contact_inputs(c: &mut Criterion) {
    let positions = contact_positions();
    let inputs_gen = ContactInputsGen {};

    c.bench_function("generate inputs for: contact", |b| {
        b.iter(|| sum_of_all_inputs(black_box(&positions), &inputs_gen))
    });
}

#[allow(dead_code)]
fn race_inputs(c: &mut Criterion) {
    let positions = race_positions();
    let inputs_gen = RaceInputsGen {};

    c.bench_function("generate inputs for: race", |b| {
        b.iter(|| sum_of_all_inputs(black_box(&positions), &inputs_gen))
    });
}

criterion_group!(benches, contact_inputs, race_inputs);
criterion_main!(benches);
