//! Criterion benchmark target.
//!
//! What is measured here is what the search actually spends its time on at a
//! 512-wide network: the accumulator update per move, the output layer per
//! leaf, and a fixed-depth search that exercises both in proportion.

use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};

use nexus::board::Position;
use nexus::nnue::{self, Accumulator};
use nexus::search::{Evaluator, Search, SearchLimits};
use nexus::types::{Color, Piece, PieceType, Square};

/// Middlegame position with plenty of material, so accumulator work is typical.
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

fn benchmarks(c: &mut Criterion) {
    nexus::magic::init();
    let net = nnue::embedded().expect("networks/default.bin matches this build");
    let pos: Position = KIWIPETE.parse().expect("test fen is valid");
    let acc = Accumulator::refresh(&pos, net);

    c.bench_function("nnue evaluate", |b| {
        b.iter(|| black_box(net.evaluate(black_box(&acc), pos.side_to_move())))
    });

    c.bench_function("nnue accumulator add", |b| {
        b.iter(|| {
            let mut acc = acc;
            let piece = Piece::new(Color::White, PieceType::Knight);
            let e5 = Square::from_uci("e5").expect("e5 is a square");
            acc.add(net, black_box(piece), black_box(e5));
            black_box(acc)
        })
    });

    c.bench_function("nnue accumulator refresh", |b| {
        b.iter(|| black_box(Accumulator::refresh(black_box(&pos), net)))
    });

    c.bench_function("search depth 8", |b| {
        b.iter(|| {
            let mut search = Search::new(Arc::new(AtomicBool::new(false)));
            search.set_evaluator(Evaluator::Nnue(net));
            black_box(search.run(
                black_box(&pos),
                SearchLimits {
                    max_depth: Some(8),
                    ..Default::default()
                },
                &mut |_| {},
            ))
        })
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
