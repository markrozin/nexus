//! Remove duplicate boards from a datagen text file.
//!
//! ```text
//! cargo build --release --features datagen --bin datadedup
//! ./target/release/datadedup data/selfplay.txt data/selfplay.dedup.txt
//! ```
//!
//! Random openings recur: eight random plies from the start position reach the
//! same handful of boards far more often than chance would suggest, and each
//! game that passes through one records it again. About 10% of a sample run
//! was duplicates. Left in, they weight those openings far above their share
//! and the network overfits them.
//!
//! A board is identified by the first four FEN fields -- placement, side to
//! move, castling, en passant -- since the move counters change nothing the
//! network sees. The first occurrence is kept.
//!
//! Only a 64-bit hash of each key is stored, so memory is roughly 16 bytes per
//! unique board plus hash-set overhead: a few gigabytes at 100M positions.
//! At that scale the chance of two distinct boards colliding is on the order of
//! one in several thousand runs, which costs at most one dropped position.

use std::collections::HashSet;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: datadedup <input> <output>");
        return ExitCode::from(2);
    };
    if input == output {
        eprintln!("input and output must differ: the input is read while output is written");
        return ExitCode::from(2);
    }

    let reader = match File::open(input) {
        Ok(f) => BufReader::with_capacity(1 << 20, f),
        Err(e) => {
            eprintln!("cannot open {input}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut writer = match File::create(output) {
        Ok(f) => BufWriter::with_capacity(1 << 20, f),
        Err(e) => {
            eprintln!("cannot create {output}: {e}");
            return ExitCode::from(2);
        }
    };

    let mut seen: HashSet<u64> = HashSet::new();
    let mut kept = 0u64;
    let mut dropped = 0u64;
    let mut skipped = 0u64;

    for line in reader.lines() {
        let Ok(line) = line else {
            skipped += 1;
            continue;
        };
        let Some(fen) = line.split(" | ").next() else {
            skipped += 1;
            continue;
        };

        // Hash the four identifying fields without allocating a joined string.
        // `DefaultHasher::new` uses fixed keys, so a rerun keeps the same lines.
        let mut hasher = DefaultHasher::new();
        let mut fields = 0;
        for field in fen.split(' ').take(4) {
            field.hash(&mut hasher);
            fields += 1;
        }
        if fields < 4 {
            skipped += 1;
            continue;
        }

        if seen.insert(hasher.finish()) {
            if writeln!(writer, "{line}").is_err() {
                eprintln!("write failed");
                return ExitCode::from(1);
            }
            kept += 1;
        } else {
            dropped += 1;
        }
    }

    if writer.flush().is_err() {
        eprintln!("flush failed");
        return ExitCode::from(1);
    }

    let total = kept + dropped;
    let pct = if total == 0 { 0.0 } else { dropped as f64 * 100.0 / total as f64 };
    println!("kept {kept}, dropped {dropped} duplicates ({pct:.1}%), skipped {skipped} unreadable");
    ExitCode::SUCCESS
}
