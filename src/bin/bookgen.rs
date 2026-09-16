//! Generate an opening book for SPRT testing.
//!
//! ```text
//! cargo build --release --features bookgen --bin bookgen
//! ./target/release/bookgen 2000 > books/random8.epd
//! ```
//!
//! Why a book at all: the engine is deterministic, so every game from the start
//! position would be byte-identical and a match would carry no information.
//!
//! Why our own rather than a published one: the standard test books are built
//! for engines strong enough to draw most balanced positions, and deliberately
//! skew the opening to force decisive results. At this strength games are
//! already decisive, so an unbiased spread of positions is the better sample —
//! and generating it needs nothing downloaded.
//!
//! The recipe is the one datagen will use in milestone 12: play a few random
//! plies, then throw the position away unless a shallow search says it is
//! roughly level. Purely random plies produce a fair number of positions that
//! are already lost, and those teach a match nothing.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use nexus::board::Position;
use nexus::movegen::generate_legal;
use nexus::rng::Rng;
use nexus::search::{Evaluator, Search, SearchLimits};

/// Random plies played out of the start position.
const OPENING_PLIES: usize = 8;
/// Reject anything a shallow search scores beyond this, in centipawns.
const BALANCE_MARGIN: i32 = 150;
/// Depth of the balance check. Enough to see a hanging piece, cheap enough to
/// run thousands of times.
const FILTER_DEPTH: u32 = 4;

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2000);

    let mut rng = Rng::new(0x00B0_0C00_0000_0001);
    let mut search = Search::new(Arc::new(AtomicBool::new(false)));
    // Handcrafted, so the book depends only on this file and not on whichever
    // network happens to be embedded when it is regenerated.
    search.set_evaluator(Evaluator::Handcrafted);
    let limits = SearchLimits {
        max_depth: Some(FILTER_DEPTH),
        ..Default::default()
    };

    let mut emitted = 0usize;
    let mut attempts = 0usize;
    while emitted < count {
        attempts += 1;
        let Some(pos) = random_opening(&mut rng) else {
            continue;
        };
        let score = search.run(&pos, limits, &mut |_| {}).score;
        if score.abs() > BALANCE_MARGIN {
            continue;
        }
        println!("{}", pos.to_fen());
        emitted += 1;
    }
    eprintln!("{emitted} positions from {attempts} attempts");
}

/// Play [`OPENING_PLIES`] uniformly random legal moves. `None` if the game
/// ended early or the result is already terminal.
fn random_opening(rng: &mut Rng) -> Option<Position> {
    let mut pos = Position::startpos();
    for _ in 0..OPENING_PLIES {
        let moves = generate_legal(&pos);
        let pick = rng.choose(moves.len())?;
        pos = pos.make_move(moves[pick]);
    }
    // A book position both sides can actually play from.
    if generate_legal(&pos).is_empty() {
        return None;
    }
    Some(pos)
}
