//! Criterion benchmark target.
//!
//! Intentionally empty: it compiles and runs with zero benchmarks so the
//! harness is wired up before there is engine code to measure. Add benchmark
//! functions here (perft, movegen, eval, search-to-depth) and register them in
//! `criterion_group!`.

use criterion::{criterion_group, criterion_main, Criterion};

fn benchmarks(_c: &mut Criterion) {
    // No benchmarks yet.
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
