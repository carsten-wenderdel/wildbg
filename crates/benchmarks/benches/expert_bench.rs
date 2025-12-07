use crate::helper::contact_positions;
use criterion::{Criterion, criterion_group, criterion_main};
use engine::inputs::expert::ExpertInputs;
use engine::position::Position;
use mimalloc::MiMalloc;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod helper;

// This file contains benchmarks for expert features like pip_count.

// Helper Methods

fn sum_of_pip_counts(positions: &[Position]) -> f32 {
    positions.iter().map(|p| p.pip_count()).sum()
}

// Benchmark methods

fn pip_count_contact(c: &mut Criterion) {
    let positions = contact_positions();

    c.bench_function("pip_count for contact positions", |b| {
        b.iter(|| sum_of_pip_counts(black_box(&positions)))
    });
}

criterion_group!(benches, pip_count_contact);
criterion_main!(benches);
