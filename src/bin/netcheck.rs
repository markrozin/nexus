//! Check a trained network against datagen output before trusting it.
//!
//! ```text
//! cargo build --release --features datagen --bin netcheck
//! ./target/release/netcheck nets/net/quantised.bin data/pipeline.txt 200000
//! ./target/release/netcheck nets/net/quantised.bin data/heldout.txt 32000 --mappings
//! ```
//!
//! A network with a perspective bug trains to a clean loss curve and then plays
//! as though it is trying to lose. This catches that without playing a game.
//! A correct net's evaluations correlate with the search scores it was trained
//! on, for *both* sides to move. A sign error in the feature indexing or the
//! accumulator order shows up as a correlation near zero, or negative, on
//! exactly one side -- which is why the report splits by side to move.
//!
//! The handcrafted evaluation is reported alongside as a reference. It produced
//! those scores, to within the 70 centipawn quiet margin, so it is roughly the
//! ceiling a net trained on this data can reach.
//!
//! `--mappings` goes further. A *subtly* wrong feature mapping -- two piece
//! types swapped, a horizontal flip where a vertical one belongs -- still gives
//! sane material values and a respectable correlation, so the checks above
//! pass it. Scoring the net under each plausible alternative convention
//! settles it: the mapping the trainer really used fits best, by a clear
//! margin. Run it on positions the net was not trained on.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;

use newchessbot::board::Position;
use newchessbot::eval::evaluate;
use newchessbot::nnue::{self, Network};
use newchessbot::types::{Color, Piece, Square};

/// Below this on either side, the net is not merely weak but wrong.
const MIN_CORRELATION: f64 = 0.2;
/// A healthy net scores both sides about equally well; a large gap between
/// them points at a perspective error rather than undertraining.
const MAX_SIDE_GAP: f64 = 0.3;

/// Running Pearson correlation, so a large file never has to fit in memory.
#[derive(Default)]
struct Correlation {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Correlation {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }

    fn r(&self) -> f64 {
        if self.n < 2.0 {
            return 0.0;
        }
        let cov = self.sxy - self.sx * self.sy / self.n;
        let vx = self.sxx - self.sx * self.sx / self.n;
        let vy = self.syy - self.sy * self.sy / self.n;
        if vx <= 0.0 || vy <= 0.0 {
            0.0
        } else {
            cov / (vx * vy).sqrt()
        }
    }

    /// Least-squares fit `y = offset + slope * x`. For the net against the
    /// recorded scores, a slope far from 1 means the net's centipawns are on a
    /// different scale from the labels -- which a correlation cannot see.
    fn fit(&self) -> (f64, f64) {
        if self.n < 2.0 {
            return (0.0, 0.0);
        }
        let cov = self.sxy - self.sx * self.sy / self.n;
        let vx = self.sxx - self.sx * self.sx / self.n;
        let slope = if vx > 0.0 { cov / vx } else { 0.0 };
        let offset = (self.sy - slope * self.sx) / self.n;
        (slope, offset)
    }
}

#[derive(Default)]
struct Side {
    net: Correlation,
    handcrafted: Correlation,
    lines: u64,
}

fn white_relative(score: i32, side_to_move: Color) -> i32 {
    match side_to_move {
        Color::White => score,
        Color::Black => -score,
    }
}

/// Assemble a Chess768-shaped index from its three parts.
fn index(own: bool, piece_type: usize, square: usize) -> usize {
    (if own { 0 } else { 384 }) + piece_type * 64 + square
}

fn vertical(perspective: Color, square: usize) -> usize {
    match perspective {
        Color::White => square,
        Color::Black => square ^ 56,
    }
}

fn own(perspective: Color, piece: Piece) -> bool {
    piece.color() == perspective
}

/// A named feature convention, and whether the accumulator halves are swapped.
struct Mapping {
    name: &'static str,
    map: fn(Color, Piece, Square) -> usize,
    swap_halves: bool,
}

/// The engine's mapping first; every other entry changes one convention.
const MAPPINGS: &[Mapping] = &[
    Mapping {
        name: "engine (current)",
        map: nnue::feature,
        swap_halves: false,
    },
    Mapping {
        name: "knight/bishop swapped",
        map: |p, pc, sq| {
            let pt = match pc.piece_type().index() {
                1 => 2,
                2 => 1,
                other => other,
            };
            index(own(p, pc), pt, vertical(p, sq.index()))
        },
        swap_halves: false,
    },
    Mapping {
        name: "piece order reversed",
        map: |p, pc, sq| index(own(p, pc), 5 - pc.piece_type().index(), vertical(p, sq.index())),
        swap_halves: false,
    },
    Mapping {
        name: "own/opponent swapped",
        map: |p, pc, sq| index(!own(p, pc), pc.piece_type().index(), vertical(p, sq.index())),
        swap_halves: false,
    },
    Mapping {
        name: "no flip for Black",
        map: |p, pc, sq| index(own(p, pc), pc.piece_type().index(), sq.index()),
        swap_halves: false,
    },
    Mapping {
        name: "horizontal flip for Black",
        map: |p, pc, sq| {
            let square = match p {
                Color::White => sq.index(),
                Color::Black => sq.index() ^ 7,
            };
            index(own(p, pc), pc.piece_type().index(), square)
        },
        swap_halves: false,
    },
    Mapping {
        name: "rotation for Black",
        map: |p, pc, sq| {
            let square = match p {
                Color::White => sq.index(),
                Color::Black => sq.index() ^ 63,
            };
            index(own(p, pc), pc.piece_type().index(), square)
        },
        swap_halves: false,
    },
    Mapping {
        name: "accumulator halves swapped",
        map: nnue::feature,
        swap_halves: true,
    },
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(net_path), Some(data_path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: netcheck <quantised.bin> <data.txt> [max_lines] [--mappings]");
        return ExitCode::from(2);
    };
    let limit: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    let compare_mappings = args.iter().any(|a| a == "--mappings");

    let net = match Network::load(Path::new(net_path)) {
        Ok(net) => net,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let file = match File::open(data_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {data_path}: {e}");
            return ExitCode::from(2);
        }
    };

    let mut white = Side::default();
    let mut black = Side::default();
    // Per mapping: [White to move, Black to move].
    let mut mapped: Vec<[Correlation; 2]> = MAPPINGS.iter().map(|_| Default::default()).collect();
    let mut skipped = 0u64;

    for line in BufReader::new(file).lines().take(limit as usize) {
        let Ok(line) = line else {
            skipped += 1;
            continue;
        };
        let fields: Vec<&str> = line.split(" | ").collect();
        if fields.len() != 3 {
            skipped += 1;
            continue;
        }
        let (Ok(pos), Ok(score)) = (fields[0].parse::<Position>(), fields[1].trim().parse::<i32>())
        else {
            skipped += 1;
            continue;
        };

        let stm = pos.side_to_move();
        let net_white = white_relative(net.evaluate_position(&pos), stm) as f64;
        let hce_white = white_relative(evaluate(&pos), stm) as f64;
        let recorded = score as f64;

        let side = match stm {
            Color::White => &mut white,
            Color::Black => &mut black,
        };
        side.lines += 1;
        side.net.add(net_white, recorded);
        side.handcrafted.add(hce_white, recorded);

        if compare_mappings {
            for (mapping, sides) in MAPPINGS.iter().zip(mapped.iter_mut()) {
                let eval = net.evaluate_mapped(&pos, &mapping.map, mapping.swap_halves);
                sides[stm.index()].add(white_relative(eval, stm) as f64, recorded);
            }
        }
    }

    println!("skipped {skipped} unreadable lines\n");
    println!(
        "{:<15} {:>9} {:>12} {:>14} {:>10} {:>10}",
        "side to move", "lines", "net r", "handcrafted r", "slope", "offset"
    );
    for (name, side) in [("White", &white), ("Black", &black)] {
        let (slope, offset) = side.net.fit();
        println!(
            "{:<15} {:>9} {:>12.3} {:>14.3} {:>10.3} {:>10.1}",
            name,
            side.lines,
            side.net.r(),
            side.handcrafted.r(),
            slope,
            offset
        );
    }
    println!("\n(slope and offset fit recorded = offset + slope * net, in centipawns)");

    if compare_mappings {
        println!("\n{:<28} {:>10} {:>10}", "feature mapping", "White r", "Black r");
        for (mapping, sides) in MAPPINGS.iter().zip(&mapped) {
            println!(
                "{:<28} {:>10.3} {:>10.3}",
                mapping.name,
                sides[0].r(),
                sides[1].r()
            );
        }
        let best = mapped
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let score = |s: &[Correlation; 2]| s[0].r() + s[1].r();
                score(a.1).total_cmp(&score(b.1))
            })
            .map_or(0, |(i, _)| i);
        println!(
            "\nbest fit: {}{}",
            MAPPINGS[best].name,
            if best == 0 {
                " -- the engine agrees with the trainer"
            } else {
                " -- NOT the engine's mapping: suspect a feature mismatch"
            }
        );
    }

    let (rw, rb) = (white.net.r(), black.net.r());
    println!();
    if rw < MIN_CORRELATION || rb < MIN_CORRELATION {
        println!(
            "FAIL: net correlation below {MIN_CORRELATION} on at least one side. \
             If one side is fine and the other is not, suspect the perspective \
             flip or accumulator order before suspecting training."
        );
        ExitCode::from(1)
    } else if (rw - rb).abs() > MAX_SIDE_GAP {
        println!(
            "FAIL: the two sides differ by more than {MAX_SIDE_GAP}. A correct net \
             treats both colours alike; this gap points at a perspective error."
        );
        ExitCode::from(1)
    } else {
        println!("PASS: net tracks the recorded scores for both sides to move");
        ExitCode::SUCCESS
    }
}
