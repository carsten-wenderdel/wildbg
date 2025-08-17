use crate::helper::{contact_positions_after_moving, race_positions_after_moving};
use criterion::{Criterion, criterion_group, criterion_main};
use engine::inputs::{ContactInputsGen, InputsGen, RaceInputsGen};
use engine::onnx::OnnxEvaluator;
use mimalloc::MiMalloc;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod helper;

// This file contains benchmarks for inference of the neural networks.
// These should best be run with proper networks, not the tiny ones committed to the main branch.

// Helper methods

fn infer_net<T: InputsGen>(net: &OnnxEvaluator<T>, inputs: Vec<Vec<f32>>) {
    for input in inputs {
        net.eval_inputs(input);
    }
}

// Benchmark methods

fn contact_net(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.sample_size(40);

    let inputs_gen = ContactInputsGen {};
    let positions = contact_positions_after_moving();
    let inputs: Vec<Vec<f32>> = positions
        .iter()
        .map(|positions| inputs_gen.inputs_for_all(positions))
        .collect();
    let onnx = OnnxEvaluator::contact_default().unwrap();

    println!(
        "{:?} positions in {:?} batches",
        positions.iter().map(|x| x.len()).sum::<usize>(),
        positions.len()
    );

    group.bench_function("Inference: contact", |b| {
        b.iter(|| infer_net(black_box(&onnx), black_box(inputs.clone())))
    });
}

fn race_net(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.sample_size(40);

    let inputs_gen = RaceInputsGen {};
    let positions = race_positions_after_moving();
    let inputs: Vec<Vec<f32>> = positions
        .iter()
        .map(|positions| inputs_gen.inputs_for_all(positions))
        .collect();
    let onnx = OnnxEvaluator::race_default().unwrap();

    println!(
        "{:?} positions in {:?} batches",
        positions.iter().map(|x| x.len()).sum::<usize>(),
        positions.len()
    );

    group.bench_function("Inference: race", |b| {
        b.iter(|| infer_net(black_box(&onnx), black_box(inputs.clone())))
    });
}

criterion_group!(benches, contact_net, race_net);
criterion_main!(benches);
