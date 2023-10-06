use coach::rollout::RolloutEvaluator;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::evaluator::Evaluator;
use engine::onnx::OnnxEvaluator;
use engine::pos;
use engine::position::Position;
use std::collections::HashMap;

fn rollout_close_to_race(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.sample_size(10);
    let rollout = RolloutEvaluator::with_evaluator(OnnxEvaluator::with_default_model_for_tests());
    // Some random position before there is a race.
    let position =
        pos!(x 13:2, 9:1, 8:1, 7:2, 6:3, 5:2, 5:4, 3:1; o 12:4, 15:2, 17:1, 18:2, 19:3, 20:2, 21:1);
    group.bench_function("rollout_close_to_race", |b| {
        b.iter(|| rollout.eval(black_box(&position)))
    });
    group.finish();
}

fn rollout_early_game(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.sample_size(10);
    let rollout = RolloutEvaluator::with_evaluator(OnnxEvaluator::with_default_model_for_tests());
    // Let's say the opponent has already moved 54 from the starting position:
    let position = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:4, 12:4, 5:1, 1:1);
    group.bench_function("rollout_early_game", |b| {
        b.iter(|| rollout.eval(black_box(&position)))
    });
    group.finish();
}

criterion_group!(benches, rollout_close_to_race, rollout_early_game);
criterion_main!(benches);
