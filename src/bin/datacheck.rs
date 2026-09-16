//! Verify a datagen text file before spending money training on it.
//!
//! ```text
//! cargo build --release --features datagen --bin datacheck
//! ./target/release/datacheck data/selfplay.txt
//! ```
//!
//! The failures this guards against are the silent ones. A perspective sign
//! error or a flipped result does not break training: it produces a clean loss
//! curve and a network that plays as though it is trying to lose. You only find
//! out after paying for the GPU time, so find out here instead, for free.
//!
//! # The decisive check
//!
//! Datagen keeps a position only when `|static eval - search score| <= 70`,
//! with both sides of that comparison side-to-move relative, and then writes
//! the search score converted to White-relative. So for every line,
//! re-evaluating the FEN and converting the same way must land within 70 of the
//! recorded score. That is a guarantee, not a tendency.
//!
//! If Black-to-move scores were written with the wrong sign, the recorded score
//! would sit near `-static` while the recomputation gives `+static`, and the
//! bound would break by twice the evaluation on exactly the Black-to-move lines.
//! The report splits violations by side to move for that reason.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;

use nexus::board::Position;
use nexus::eval::evaluate;
use nexus::types::Color;

/// Must match `QUIET_MARGIN_SEARCH` in datagen. Allow one centipawn of slack:
/// nothing here is rounded, but a check that fails on its own boundary is worse
/// than useless.
const SEARCH_MARGIN: i32 = 70 + 1;

/// A score this large, one way or the other, should mostly agree with the
/// result. Used only as a sanity signal, since single games are noisy.
const DECISIVE_SCORE: i32 = 300;

#[derive(Default)]
struct SideStats {
    lines: u64,
    margin_violations: u64,
    /// Sum of `recorded - recomputed`, to expose a systematic offset.
    drift: i64,
}

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: datacheck <file>");
        return ExitCode::from(2);
    };
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return ExitCode::from(2);
        }
    };

    let mut malformed = 0u64;
    let mut bad_fen = 0u64;
    let mut bad_result = 0u64;
    let mut in_check = 0u64;
    let mut white = SideStats::default();
    let mut black = SideStats::default();
    let mut boards = HashSet::new();
    let mut duplicates = 0u64;
    // Among positions the score calls decisive: how many results agree.
    let mut decisive = 0u64;
    let mut decisive_agree = 0u64;
    let mut first_violation: Option<String> = None;

    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            malformed += 1;
            continue;
        };
        let fields: Vec<&str> = line.split(" | ").collect();
        if fields.len() != 3 {
            malformed += 1;
            continue;
        }
        let (fen, score_text, result_text) = (fields[0], fields[1], fields[2]);

        let Ok(pos) = fen.parse::<Position>() else {
            bad_fen += 1;
            continue;
        };
        let Ok(score) = score_text.trim().parse::<i32>() else {
            malformed += 1;
            continue;
        };
        let result = match result_text.trim() {
            "1.0" => 1.0f32,
            "0.5" => 0.5,
            "0.0" => 0.0,
            _ => {
                bad_result += 1;
                continue;
            }
        };

        // Only the placement, side and castling identify a board for training;
        // the move counters do not change what the network sees.
        let key: String = fen.split(' ').take(4).collect::<Vec<_>>().join(" ");
        if !boards.insert(key) {
            duplicates += 1;
        }

        if pos.in_check(pos.side_to_move()) {
            in_check += 1;
        }

        let static_stm = evaluate(&pos);
        let static_white = match pos.side_to_move() {
            Color::White => static_stm,
            Color::Black => -static_stm,
        };
        let stats = match pos.side_to_move() {
            Color::White => &mut white,
            Color::Black => &mut black,
        };
        stats.lines += 1;
        stats.drift += (score - static_white) as i64;
        if (score - static_white).abs() > SEARCH_MARGIN {
            stats.margin_violations += 1;
            if first_violation.is_none() {
                first_violation = Some(format!(
                    "{line}\n    recomputed static (White-relative) = {static_white}"
                ));
            }
        }

        if score.abs() >= DECISIVE_SCORE {
            decisive += 1;
            let favours_white = score > 0;
            if (favours_white && result == 1.0) || (!favours_white && result == 0.0) {
                decisive_agree += 1;
            }
        }
    }

    let total = white.lines + black.lines;
    println!("lines checked      {total}");
    println!("malformed          {malformed}");
    println!("unparseable FEN    {bad_fen}");
    println!("bad result field   {bad_result}");
    println!("in check           {in_check}   (datagen should emit none)");
    println!(
        "duplicate boards   {duplicates}   ({:.1}%)",
        pct(duplicates, total)
    );
    println!();
    for (name, s) in [("White to move", &white), ("Black to move", &black)] {
        let mean = if s.lines == 0 { 0.0 } else { s.drift as f64 / s.lines as f64 };
        println!(
            "{name}: {} lines, {} margin violations, mean recorded-minus-static {:+.1}",
            s.lines, s.margin_violations, mean
        );
    }
    println!();
    println!(
        "decisive scores (|score| >= {DECISIVE_SCORE}): {decisive}, result agrees {:.1}%",
        pct(decisive_agree, decisive)
    );
    if let Some(v) = &first_violation {
        println!();
        println!("first margin violation:\n    {v}");
    }

    let violations = white.margin_violations + black.margin_violations;
    let broken = malformed + bad_fen + bad_result + in_check + violations;
    println!();
    if broken == 0 {
        println!("PASS: format, sign convention and quiet filter all consistent");
        ExitCode::SUCCESS
    } else {
        println!("FAIL: {broken} problems. Do not train on this file.");
        ExitCode::from(1)
    }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}
