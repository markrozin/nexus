//! Self-play data generation for NNUE training.
//!
//! ```text
//! cargo build --release --features datagen --bin datagen
//! ./target/release/datagen --games 10000 --threads 8 --out data/selfplay.txt
//! ```
//!
//! Emits one line per recorded position:
//!
//! ```text
//! <FEN> | <score> | <result>
//! ```
//!
//! Score is centipawns and result is 1.0/0.5/0.0, both **White-relative**, which
//! is the text format `bullet` converts from. Writing text rather than packed
//! binary is deliberate: the conversion is bullet's problem, and a sign error in
//! a hand-rolled binary writer is invisible until a trained net plays like it is
//! trying to lose.
//!
//! # Why the labels look like this
//!
//! The training target blends two flawed signals. The search score is precise
//! but only teaches the network what this engine already knows; the game result
//! is unbiased but extremely noisy, since one blunder forty moves later flips
//! the label on a position that was genuinely fine. Blending takes precision
//! from one and grounding from the other. Both are written out and the trainer
//! does the blending.
//!
//! # Why these filters
//!
//! The network has no search. It cannot see a hanging queen, so training it on
//! tactical positions teaches it noise — the same reason quiescence search
//! exists. A position is kept only when it is quiet by all three of:
//!
//! - the side to move is not in check,
//! - `|static eval - quiescence eval| <= 60` centipawns,
//! - `|static eval - search eval| <= 70` centipawns.
//!
//! The two margins are from Tan and Watkinson Medina, *Study of the Proper NNUE
//! Dataset* (arXiv:2412.17948), who tuned them rather than guessing. The
//! original plan here was the cruder "skip checks and positions whose best move
//! is a capture".

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use nexus::board::Position;
use nexus::eval::evaluate;
use nexus::movegen::generate_legal;
use nexus::rng::Rng;
use nexus::search::{Evaluator, Search, SearchLimits, MATE_IN_MAX_PLY};
use nexus::types::Color;

/// Fixed node budget per move.
///
/// Nodes, not depth: node counts are hardware-independent, so data generated on
/// one machine matches another, and a later speed optimization does not
/// silently change the data distribution.
const NODES_PER_MOVE: u64 = 5_000;

/// Random plies out of the start position, for opening variety.
const RANDOM_PLIES: usize = 8;

/// Discard an opening this lopsided: a game already lost teaches nothing.
const OPENING_BALANCE_MARGIN: i32 = 200;

/// Quiet-position margins. See the module docs.
const QUIET_MARGIN_QSEARCH: i32 = 60;
const QUIET_MARGIN_SEARCH: i32 = 70;

/// Skip the first few plies of each game: they are near-identical across games
/// and would be wildly over-represented.
const SKIP_OPENING_PLIES: usize = 2;

/// Adjudicate once one side is this far ahead for this many consecutive plies.
const ADJUDICATE_SCORE: i32 = 1_500;
const ADJUDICATE_PLIES: usize = 6;

/// Hard cap on game length.
const MAX_GAME_PLIES: usize = 400;

struct Config {
    games: usize,
    threads: usize,
    out: PathBuf,
    seed: u64,
}

fn parse_args() -> Config {
    let mut config = Config {
        games: 1_000,
        threads: 1,
        out: PathBuf::from("data/selfplay.txt"),
        seed: 0x5EED_0000_0000_0001,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let value = args.get(i + 1);
        match args[i].as_str() {
            "--games" => config.games = value.and_then(|v| v.parse().ok()).unwrap_or(config.games),
            "--threads" => {
                config.threads = value.and_then(|v| v.parse().ok()).unwrap_or(config.threads)
            }
            "--out" => {
                if let Some(v) = value {
                    config.out = PathBuf::from(v);
                }
            }
            "--seed" => config.seed = value.and_then(|v| v.parse().ok()).unwrap_or(config.seed),
            other => eprintln!("ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    config
}

/// One position held back until the game result is known.
struct Sample {
    fen: String,
    /// Centipawns, White-relative.
    score: i32,
}

fn main() -> std::io::Result<()> {
    let config = parse_args();
    if let Some(parent) = config.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    nexus::magic::init();

    let writer = Arc::new(Mutex::new(BufWriter::new(File::create(&config.out)?)));
    let games_done = Arc::new(AtomicU64::new(0));
    let positions = Arc::new(AtomicU64::new(0));
    let start = std::time::Instant::now();

    let per_thread = config.games.div_ceil(config.threads.max(1));
    std::thread::scope(|scope| {
        for thread_index in 0..config.threads.max(1) {
            let writer = Arc::clone(&writer);
            let games_done = Arc::clone(&games_done);
            let positions = Arc::clone(&positions);
            let seed = config.seed.wrapping_add(thread_index as u64 * 0x9E37_79B9_7F4A_7C15);
            scope.spawn(move || {
                // One search and one table per thread; nothing is shared, so
                // threads never contend and results stay reproducible per seed.
                let mut worker = Worker::new(seed);
                // Buffer locally and flush in batches: taking the output lock
                // once per position would serialise every thread.
                let mut batch: Vec<String> = Vec::with_capacity(4096);
                for _ in 0..per_thread {
                    let written = worker.play_game(&mut batch);
                    positions.fetch_add(written as u64, Ordering::Relaxed);
                    let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                    if batch.len() >= 2048 {
                        flush(&writer, &mut batch);
                    }
                    if done % 100 == 0 {
                        let secs = start.elapsed().as_secs_f64();
                        eprintln!(
                            "{done} games, {} positions, {:.0} pos/s",
                            positions.load(Ordering::Relaxed),
                            positions.load(Ordering::Relaxed) as f64 / secs
                        );
                    }
                }
                flush(&writer, &mut batch);
            });
        }
    });

    writer.lock().expect("writer lock").flush()?;
    eprintln!(
        "done: {} games, {} positions in {:.1}s",
        games_done.load(Ordering::Relaxed),
        positions.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn flush(writer: &Mutex<BufWriter<File>>, batch: &mut Vec<String>) {
    if batch.is_empty() {
        return;
    }
    let mut guard = writer.lock().expect("writer lock");
    for line in batch.drain(..) {
        let _ = guard.write_all(line.as_bytes());
        let _ = guard.write_all(b"\n");
    }
}

struct Worker {
    rng: Rng,
    search: Search,
}

impl Worker {
    fn new(seed: u64) -> Self {
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        // The quiet filter compares the handcrafted static evaluation against
        // this search's scores, so both have to come from the same evaluator.
        // Moving datagen to the network means changing the filter with it.
        search.set_evaluator(Evaluator::Handcrafted);
        Self {
            rng: Rng::new(seed),
            search,
        }
    }

    fn limits(&self) -> SearchLimits {
        SearchLimits {
            max_nodes: Some(NODES_PER_MOVE),
            ..Default::default()
        }
    }

    /// Play one game, appending its recorded positions to `out`. Returns how
    /// many were written.
    fn play_game(&mut self, out: &mut Vec<String>) -> usize {
        let Some(mut pos) = self.random_opening() else {
            return 0;
        };

        let mut samples: Vec<Sample> = Vec::new();
        let mut keys: Vec<u64> = Vec::new();
        let mut decisive_streak = 0usize;
        let mut result = 0.5f32;

        for ply in 0..MAX_GAME_PLIES {
            let moves = generate_legal(&pos);
            if moves.is_empty() {
                result = if pos.in_check(pos.side_to_move()) {
                    // Side to move is mated.
                    match pos.side_to_move() {
                        Color::White => 0.0,
                        Color::Black => 1.0,
                    }
                } else {
                    0.5
                };
                break;
            }
            if pos.halfmove_clock() >= 100 {
                break;
            }
            // Bare kings, or a lone minor, cannot be won. Stop here rather than
            // shuffle toward the fifty-move rule recording dozens of dead
            // positions: they score a flat draw, sail through the quiet filter,
            // and teach the network nothing. bullet's validator flagged 112 of
            // them in a 16K-position sample before this check existed.
            if nexus::search::is_insufficient_material(&pos) {
                result = 0.5;
                break;
            }

            self.search.set_game_history(&keys);
            let outcome = self.search.run(&pos, self.limits(), &mut |_| {});
            if outcome.best_move.is_none() {
                break;
            }

            // Adjudicate a settled game rather than playing out a hopeless one.
            if outcome.score.abs() >= ADJUDICATE_SCORE {
                decisive_streak += 1;
                if decisive_streak >= ADJUDICATE_PLIES {
                    let winner_is_side_to_move = outcome.score > 0;
                    let winner = if winner_is_side_to_move {
                        pos.side_to_move()
                    } else {
                        pos.side_to_move().flip()
                    };
                    result = match winner {
                        Color::White => 1.0,
                        Color::Black => 0.0,
                    };
                    break;
                }
            } else {
                decisive_streak = 0;
            }

            if ply >= SKIP_OPENING_PLIES && self.is_quiet(&pos, outcome.score) {
                samples.push(Sample {
                    fen: pos.to_fen(),
                    score: white_relative(outcome.score, pos.side_to_move()),
                });
            }

            keys.push(pos.zobrist());
            pos = pos.make_move(outcome.best_move);
        }

        let written = samples.len();
        for sample in samples {
            out.push(format!("{} | {} | {:.1}", sample.fen, sample.score, result));
        }
        written
    }

    /// Is this position stable enough to be worth labelling?
    fn is_quiet(&mut self, pos: &Position, search_score: i32) -> bool {
        if pos.in_check(pos.side_to_move()) {
            return false;
        }
        // A mate score saturates the sigmoid the trainer applies and carries no
        // gradient, so it teaches nothing.
        if search_score.abs() >= MATE_IN_MAX_PLY {
            return false;
        }
        let static_score = evaluate(pos);
        if (static_score - search_score).abs() > QUIET_MARGIN_SEARCH {
            return false;
        }
        // The comparison the margin was tuned against is static eval versus
        // quiescence, so use quiescence itself -- not a depth-1 `run`, which
        // would also halve history and bump the table generation once per
        // position and quietly degrade every search after it.
        let quiet = self.search.quiescence_score(pos);
        (static_score - quiet).abs() <= QUIET_MARGIN_QSEARCH
    }

    /// Play [`RANDOM_PLIES`] random legal moves, rejecting an opening that is
    /// already decided.
    fn random_opening(&mut self) -> Option<Position> {
        for _ in 0..32 {
            let mut pos = Position::startpos();
            let mut ok = true;
            for _ in 0..RANDOM_PLIES {
                let moves = generate_legal(&pos);
                match self.rng.choose(moves.len()) {
                    Some(index) => pos = pos.make_move(moves[index]),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || generate_legal(&pos).is_empty() {
                continue;
            }
            let probe = self.search.run(&pos, self.limits(), &mut |_| {});
            if probe.score.abs() <= OPENING_BALANCE_MARGIN {
                return Some(pos);
            }
        }
        None
    }
}

/// Search scores are side-to-move relative; the text format wants them
/// White-relative. This is the sign convention that silently ruins a training
/// run if it is wrong, so it lives in one named function.
#[inline]
fn white_relative(score: i32, side_to_move: Color) -> i32 {
    match side_to_move {
        Color::White => score,
        Color::Black => -score,
    }
}
