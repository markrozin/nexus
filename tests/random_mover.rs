//! Milestone 5 gate: the search must beat a random mover.
//!
//! Not a strength measurement — it is a smoke test that the whole stack lines
//! up. A sign error in evaluation, a negation dropped in negamax, or a mate
//! score without the ply term all still produce a legal game; they just produce
//! a bad one. Losing or drawing to a random mover catches all three.
//!
//! Real strength testing needs SPRT against a real opponent, which is milestone
//! 9 and needs `fastchess`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use newchessbot::board::Position;
use newchessbot::movegen::generate_legal;
use newchessbot::rng::Rng;
use newchessbot::search::{Search, SearchLimits};
use newchessbot::types::{Color, PieceType};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    EngineWins,
    RandomWins,
    Draw(&'static str),
}

/// Cap on game length. A won game the engine cannot finish is a failure, so
/// this is deliberately generous rather than forgiving.
const MAX_PLIES: usize = 400;

fn play_game(engine_plays: Color, seed: u64, depth: u32) -> Outcome {
    let mut pos = Position::startpos();
    let mut rng = Rng::new(seed);
    let mut search = Search::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits {
        max_depth: Some(depth),
        ..Default::default()
    };

    for _ in 0..MAX_PLIES {
        let moves = generate_legal(&pos);
        if moves.is_empty() {
            return if pos.in_check(pos.side_to_move()) {
                // The side to move is checkmated.
                if pos.side_to_move() == engine_plays {
                    Outcome::RandomWins
                } else {
                    Outcome::EngineWins
                }
            } else {
                Outcome::Draw("stalemate")
            };
        }
        if pos.halfmove_clock() >= 100 {
            return Outcome::Draw("fifty-move rule");
        }
        if is_bare_kings(&pos) {
            return Outcome::Draw("insufficient material");
        }

        let mv = if pos.side_to_move() == engine_plays {
            search.run(&pos, limits, &mut |_| {}).best_move
        } else {
            let pick = rng.choose(moves.len()).expect("move list is non-empty");
            moves[pick]
        };
        assert!(
            moves.contains(&mv),
            "played an illegal move {mv} in {}",
            pos.to_fen()
        );
        pos = pos.make_move(mv);
    }
    Outcome::Draw("move limit")
}

fn is_bare_kings(pos: &Position) -> bool {
    let minors = pos.by_type(PieceType::Knight) | pos.by_type(PieceType::Bishop);
    !pos.by_type(PieceType::Pawn).any()
        && !pos.by_type(PieceType::Rook).any()
        && !pos.by_type(PieceType::Queen).any()
        && minors.popcount() <= 1
}

/// Play `games` games, alternating colors, and report anything that was not a
/// win for the engine.
fn run_match(games: usize, depth: u32) {
    let mut losses = Vec::new();
    for game in 0..games {
        // Alternate colors so a sign error cannot hide behind always playing
        // White, and vary the seed so the games differ.
        let engine_plays = if game % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };
        let seed = 0x51D5_0000_0000_0001u64.wrapping_add(game as u64 * 0x9E37_79B9);
        let outcome = play_game(engine_plays, seed, depth);
        if outcome != Outcome::EngineWins {
            losses.push((game, engine_plays, seed, outcome));
        }
    }
    assert!(
        losses.is_empty(),
        "{} of {games} games were not won by the engine at depth {depth}: {:#?}",
        losses.len(),
        losses
    );
}

/// Quick version, kept in the default suite so a regression shows up straight
/// away.
#[test]
fn beats_a_random_mover_over_ten_games() {
    run_match(10, 3);
}

/// The milestone 5 gate proper. Ignored by default because a hundred
/// depth-4 games is slow in a debug build; run with:
/// `cargo test --release -- --ignored --nocapture`.
#[test]
#[ignore = "100 games at depth 4; run explicitly, in release"]
fn beats_a_random_mover_one_hundred_games() {
    run_match(100, 4);
}
