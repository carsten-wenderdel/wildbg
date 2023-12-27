use coach::rollout::RolloutEvaluator;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator;
use engine::pos;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Once this function is called there is no going back to more threads for the other benchmarks.
// So better call it in every benchmark function.
fn use_only_one_thread() {
    _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
}

fn rollout_close_to_race(c: &mut Criterion) {
    use_only_one_thread();
    let mut group = c.benchmark_group("group");
    group.sample_size(10);
    let rollout =
        RolloutEvaluator::with_evaluator_and_seed(CompositeEvaluator::default_tests(), 123456);
    // Some random position before there is a race.
    let position =
        pos!(x 13:2, 9:1, 8:1, 7:2, 6:3, 5:2, 5:4, 3:1; o 12:4, 15:2, 17:1, 18:2, 19:3, 20:2, 21:1);
    group.bench_function("rollout_close_to_race", |b| {
        b.iter(|| rollout.eval(black_box(&position)))
    });
    group.finish();
}

fn rollout_early_game(c: &mut Criterion) {
    use_only_one_thread();
    let mut group = c.benchmark_group("group");
    group.sample_size(10);
    let rollout =
        RolloutEvaluator::with_evaluator_and_seed(CompositeEvaluator::default_tests(), 123456);
    // Let's say the opponent has already moved 54 from the starting position:
    let position = pos!(x 24:2, 13:5, 8:3, 6:5; o 19:5, 17:4, 12:4, 5:1, 1:1);
    group.bench_function("rollout_early_game", |b| {
        b.iter(|| rollout.eval(black_box(&position)))
    });
    group.finish();
}

criterion_group!(benches, rollout_close_to_race, rollout_early_game);
criterion_main!(benches);
