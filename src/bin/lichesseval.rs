//! Convert Lichess evaluation data into bullet's text format.
//!
//! ```text
//! cargo build --release --features datagen --bin lichesseval
//! ./target/release/lichesseval lichess_db_eval.jsonl.zst data/lichess.txt [max_positions]
//! python parquet2tsv.py train-00000.parquet | ./target/release/lichesseval --tsv - data/shard0.txt
//! ```
//!
//! Two inputs, the same data:
//!
//! - the zstd-compressed JSON-lines file from database.lichess.org (CC0), with
//!   every evaluation and principal variation per position. `-` reads it from
//!   stdin. That server is too slow to feed a rental (~170 KB/s), so this mode
//!   is kept for local use.
//! - `--tsv`: `fen \t depth \t cp \t mate` lines, one position each, as
//!   `trainer/vast/parquet2tsv.py` streams them from the deduplicated Hugging
//!   Face mirror (`mateuszgrzyb/lichess-stockfish-normalized`, CC BY 4.0,
//!   derived from the Lichess CC0 database), which already keeps only the
//!   deepest evaluation per position.
//!
//! Each output line is `<FEN> | <centipawns> | 0.5`, score White-relative --
//! the same text format datagen writes. There are no game results, so the
//! result column is a placeholder and training must run at WDL 0.0.
//!
//! # What is kept
//!
//! The network has no search, so a position whose value depends on an
//! immediate tactic teaches it noise. Dropped:
//!
//! - mate scores, which saturate the training sigmoid and carry no gradient,
//! - shallow searches, below [`MIN_DEPTH`],
//! - impossible material, which the analysis board allows and bullet cannot store,
//! - the side to move in check,
//! - positions that are not quiet: in JSON mode, a best move that captures or
//!   promotes; in TSV mode, which has no principal variation, datagen's own
//!   test -- quiescence moves the static evaluation by more than
//!   [`QUIET_MARGIN`] centipawns,
//! - scores beyond [`MAX_CP`], already decided,
//! - insufficient material, a dead draw bullet's validator rejects.
//!
//! # The sign check
//!
//! Lichess documents its scores as White-relative. That is exactly the kind
//! of convention that silently ruins a training run if it is wrong, so it is
//! checked rather than trusted: the scores must correlate positively with the
//! handcrafted evaluation for both sides to move, or the run fails.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ruzstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};
use ruzstd::decoding::StreamingDecoder;

use newchessbot::board::Position;
use newchessbot::eval::evaluate;
use newchessbot::search::{is_insufficient_material, Evaluator, Search};
use newchessbot::types::{Color, PieceType, Square};

/// Shallower evaluations are too noisy to be worth a label.
const MIN_DEPTH: i64 = 18;
/// Beyond this the game is decided and the label carries almost no gradient.
const MAX_CP: i64 = 3_000;
/// TSV mode's quiet test, the same margin datagen uses between static
/// evaluation and quiescence.
const QUIET_MARGIN: i32 = 60;
/// Below this correlation with the handcrafted eval on either side, the score
/// convention is wrong rather than the evaluations merely disagreeing.
const MIN_SIGN_CORRELATION: f64 = 0.3;

/// Just enough JSON for this file: objects, arrays, strings and integers.
/// The dependency rules rule out serde, and the format is simple and fixed.
enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    Int(i64),
    Other,
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Json::Int(v) => Some(*v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Parser<'a> {
    fn parse(text: &'a str) -> Option<Json> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = parser.value()?;
        parser.whitespace();
        (parser.at == parser.bytes.len()).then_some(value)
    }

    fn whitespace(&mut self) {
        while self.bytes.get(self.at).is_some_and(|b| b.is_ascii_whitespace()) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Option<()> {
        self.whitespace();
        (self.bytes.get(self.at) == Some(&byte)).then(|| self.at += 1)
    }

    fn value(&mut self) -> Option<Json> {
        self.whitespace();
        match *self.bytes.get(self.at)? {
            b'{' => {
                self.at += 1;
                let mut fields = Vec::new();
                if self.eat(b'}').is_some() {
                    return Some(Json::Object(fields));
                }
                loop {
                    self.whitespace();
                    let key = self.string()?;
                    self.eat(b':')?;
                    fields.push((key, self.value()?));
                    if self.eat(b',').is_none() {
                        self.eat(b'}')?;
                        return Some(Json::Object(fields));
                    }
                }
            }
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                if self.eat(b']').is_some() {
                    return Some(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(b',').is_none() {
                        self.eat(b']')?;
                        return Some(Json::Array(items));
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            b'-' | b'0'..=b'9' => {
                let start = self.at;
                while self
                    .bytes
                    .get(self.at)
                    .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
                {
                    self.at += 1;
                }
                let text = std::str::from_utf8(&self.bytes[start..self.at]).ok()?;
                Some(text.parse().map_or(Json::Other, Json::Int))
            }
            b't' | b'f' | b'n' => {
                while self.bytes.get(self.at).is_some_and(u8::is_ascii_alphabetic) {
                    self.at += 1;
                }
                Some(Json::Other)
            }
            _ => None,
        }
    }

    fn string(&mut self) -> Option<String> {
        if self.bytes.get(self.at) != Some(&b'"') {
            return None;
        }
        self.at += 1;
        let start = self.at;
        loop {
            match *self.bytes.get(self.at)? {
                b'"' => break,
                // Nothing in this file is escaped, but step over an escape
                // rather than end the string early if something ever is.
                b'\\' => self.at += 2,
                _ => self.at += 1,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).ok()?.to_owned();
        self.at += 1;
        Some(text)
    }
}

/// Running Pearson correlation.
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
}

#[derive(Default)]
struct Stats {
    lines: u64,
    unreadable: u64,
    mate: u64,
    shallow: u64,
    bad_fen: u64,
    impossible: u64,
    in_check: u64,
    not_quiet: u64,
    extreme: u64,
    insufficient: u64,
    kept: u64,
}

/// Decode a zstd stream line by line, across however many frames it holds.
///
/// The Lichess file comes from a parallel compressor: many frames, each
/// preceded by a skippable frame recording its size. ruzstd's
/// `StreamingDecoder` reads exactly one frame and reports a skippable one as an
/// error, so the frames are walked here. Frames split at arbitrary bytes, so a
/// line cut by a frame boundary is carried over to the next.
///
/// `handle` returns whether to keep going. The result says whether the stream
/// ended mid-frame -- a cut-off download -- in which case everything before
/// the cut has still been handled.
fn for_each_zstd_line(input: impl Read, mut handle: impl FnMut(&str) -> bool) -> io::Result<bool> {
    let mut source = BufReader::with_capacity(1 << 20, input);
    let mut pending: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];

    loop {
        if source.fill_buf()?.is_empty() {
            break;
        }
        let mut frame = match StreamingDecoder::new(&mut source) {
            Ok(frame) => frame,
            Err(FrameDecoderError::ReadFrameHeaderError(ReadFrameHeaderError::SkipFrame {
                length,
                ..
            })) => {
                io::copy(&mut (&mut source).take(u64::from(length)), &mut io::sink())?;
                continue;
            }
            Err(_) => return Ok(true),
        };
        loop {
            let n = match frame.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => return Ok(true),
            };
            pending.extend_from_slice(&chunk[..n]);
            if let Some(last) = pending.iter().rposition(|&b| b == b'\n') {
                for line in pending[..last].split(|&b| b == b'\n') {
                    if let Ok(text) = std::str::from_utf8(line) {
                        if !handle(text.trim_end_matches('\r')) {
                            return Ok(false);
                        }
                    }
                }
                pending.drain(..=last);
            }
        }
    }
    // A complete stream whose last line has no newline.
    if !pending.is_empty() {
        if let Ok(text) = std::str::from_utf8(&pending) {
            handle(text.trim_end_matches('\r'));
        }
    }
    Ok(false)
}

/// Plain text, line by line. Never truncated in the zstd sense.
fn for_each_text_line(input: impl Read, mut handle: impl FnMut(&str) -> bool) -> io::Result<bool> {
    for line in BufReader::with_capacity(1 << 20, input).lines() {
        if !handle(line?.trim_end_matches('\r')) {
            break;
        }
    }
    Ok(false)
}

/// Could this material have arisen in a real game?
///
/// The analysis board lets users set up anything, and some did: a side with
/// sixteen pawns, or eighteen pieces. Our FEN parser accepts those, but they
/// teach nothing and bullet's format cannot hold them -- its converter drops
/// them with only a printed warning. So they are rejected here, visibly.
///
/// Promotions are the only way to gain a piece, and each one costs a pawn: so
/// pieces beyond the starting set can never outnumber the missing pawns.
fn is_plausible(pos: &Position) -> bool {
    for color in Color::ALL {
        let count = |kind: PieceType| pos.pieces(color, kind).popcount() as i32;
        let pawns = count(PieceType::Pawn);
        let promoted = (count(PieceType::Knight) - 2).max(0)
            + (count(PieceType::Bishop) - 2).max(0)
            + (count(PieceType::Rook) - 2).max(0)
            + (count(PieceType::Queen) - 1).max(0);
        if pawns > 8 || count(PieceType::King) != 1 || promoted > 8 - pawns {
            return false;
        }
        // A pawn can never stand on the first or last rank. A plain loop:
        // `Bitboard`'s own `any()` shadows the iterator method of that name.
        for sq in pos.pieces(color, PieceType::Pawn) {
            if matches!(sq.index() / 8, 0 | 7) {
                return false;
            }
        }
    }
    true
}

/// Does the first move of a PV capture or promote?
///
/// Lines are in UCI_Chess960 notation, where castling is written as the king
/// taking its own rook -- so a "capture" of a friendly piece is a castle, not
/// a capture.
fn is_forcing(pos: &Position, uci: &str) -> Option<bool> {
    if uci.len() == 5 {
        return Some(true);
    }
    let from = Square::from_uci(uci.get(0..2)?)?;
    let to = Square::from_uci(uci.get(2..4)?)?;
    let mover = pos.piece_at(from)?;
    Some(match pos.piece_at(to) {
        Some(target) => target.color() != mover.color(),
        // En passant: a pawn changing file onto an empty square.
        None => mover.piece_type() == PieceType::Pawn && from.index() % 8 != to.index() % 8,
    })
}

/// Checks shared by both formats once the FEN and score are known.
/// `quiet` is the format's own test, applied after the cheaper ones.
fn finish(
    fen: &str,
    cp: i64,
    stats: &mut Stats,
    quiet: impl FnOnce(&Position) -> Option<bool>,
) -> Option<(Position, String, i64)> {
    // Lichess FENs stop after the en passant field.
    let full_fen = if fen.split_whitespace().count() == 4 {
        format!("{fen} 0 1")
    } else {
        fen.to_owned()
    };
    let Ok(pos) = full_fen.parse::<Position>() else {
        stats.bad_fen += 1;
        return None;
    };
    if !is_plausible(&pos) {
        stats.impossible += 1;
        return None;
    }
    if pos.in_check(pos.side_to_move()) {
        stats.in_check += 1;
        return None;
    }
    if cp.abs() > MAX_CP {
        stats.extreme += 1;
        return None;
    }
    if is_insufficient_material(&pos) {
        stats.insufficient += 1;
        return None;
    }
    match quiet(&pos) {
        Some(true) => Some((pos, full_fen, cp)),
        Some(false) => {
            stats.not_quiet += 1;
            None
        }
        None => {
            stats.unreadable += 1;
            None
        }
    }
}

/// One JSON line: the deepest evaluation's first principal variation.
fn convert_json(line: &str, stats: &mut Stats) -> Option<(Position, String, i64)> {
    let Some(json) = Parser::parse(line) else {
        stats.unreadable += 1;
        return None;
    };
    let (Some(fen), Some(evals)) = (
        json.get("fen").and_then(Json::as_str),
        json.get("evals").and_then(Json::as_array),
    ) else {
        stats.unreadable += 1;
        return None;
    };

    let Some(deepest) = evals
        .iter()
        .max_by_key(|e| e.get("depth").and_then(Json::as_int).unwrap_or(-1))
    else {
        stats.unreadable += 1;
        return None;
    };
    let depth = deepest.get("depth").and_then(Json::as_int).unwrap_or(0);
    let Some(pv) = deepest
        .get("pvs")
        .and_then(Json::as_array)
        .and_then(|pvs| pvs.first())
    else {
        stats.unreadable += 1;
        return None;
    };

    if pv.get("mate").is_some() {
        stats.mate += 1;
        return None;
    }
    let Some(cp) = pv.get("cp").and_then(Json::as_int) else {
        stats.unreadable += 1;
        return None;
    };
    if depth < MIN_DEPTH {
        stats.shallow += 1;
        return None;
    }

    let first_move = pv
        .get("line")
        .and_then(Json::as_str)
        .and_then(|l| l.split_whitespace().next())
        .map(str::to_owned);
    finish(fen, cp, stats, |pos| {
        first_move.and_then(|mv| is_forcing(pos, &mv)).map(|forcing| !forcing)
    })
}

/// One TSV line: `fen \t depth \t cp \t mate`, empty fields for nulls.
fn convert_tsv(line: &str, stats: &mut Stats, search: &mut Search) -> Option<(Position, String, i64)> {
    let fields: Vec<&str> = line.split('\t').collect();
    let [fen, depth, cp, mate] = fields[..] else {
        stats.unreadable += 1;
        return None;
    };
    if !mate.trim().is_empty() {
        stats.mate += 1;
        return None;
    }
    let (Ok(depth), Ok(cp)) = (depth.trim().parse::<i64>(), cp.trim().parse::<i64>()) else {
        stats.unreadable += 1;
        return None;
    };
    if depth < MIN_DEPTH {
        stats.shallow += 1;
        return None;
    }
    finish(fen, cp, stats, |pos| {
        let stand_pat = evaluate(pos);
        Some((search.quiescence_score(pos) - stand_pat).abs() <= QUIET_MARGIN)
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tsv = args.iter().any(|a| a == "--tsv");
    let positional: Vec<&String> = args.iter().filter(|a| *a != "--tsv").collect();
    let (Some(input), Some(output)) = (positional.first(), positional.get(1)) else {
        eprintln!(
            "usage: lichesseval [--tsv] <lichess_db_eval.jsonl.zst | shard.tsv | -> <out.txt> [max_positions]"
        );
        return ExitCode::from(2);
    };
    let limit: u64 = positional.get(2).and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);

    let source: Box<dyn Read> = if input.as_str() == "-" {
        Box::new(io::stdin().lock())
    } else {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::with_capacity(1 << 20, f)),
            Err(e) => {
                eprintln!("cannot open {input}: {e}");
                return ExitCode::from(2);
            }
        }
    };
    let out = match File::create(output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot create {output}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut out = BufWriter::with_capacity(1 << 20, out);

    newchessbot::magic::init();
    // The quiet test in TSV mode compares against the handcrafted evaluation,
    // so quiescence must use it too.
    let mut search = Search::new(Arc::new(AtomicBool::new(false)));
    search.set_evaluator(Evaluator::Handcrafted);

    let mut stats = Stats::default();
    let mut sign = [Correlation::default(), Correlation::default()];
    let mut write_failed = false;

    let handle = |line: &str| {
        stats.lines += 1;
        if stats.lines % 10_000_000 == 0 {
            eprintln!("{} lines, {} kept", stats.lines, stats.kept);
        }
        let converted = if tsv {
            convert_tsv(line, &mut stats, &mut search)
        } else {
            convert_json(line, &mut stats)
        };
        let Some((pos, fen, cp)) = converted else {
            return true;
        };
        let stm = pos.side_to_move();
        let hce = match stm {
            Color::White => evaluate(&pos),
            Color::Black => -evaluate(&pos),
        };
        sign[stm.index()].add(hce as f64, cp as f64);
        if writeln!(out, "{fen} | {cp} | 0.5").is_err() {
            write_failed = true;
            return false;
        }
        stats.kept += 1;
        stats.kept < limit
    };
    let truncated = if tsv {
        for_each_text_line(source, handle)
    } else {
        for_each_zstd_line(source, handle)
    };
    let truncated = match truncated {
        Ok(truncated) => truncated,
        Err(e) => {
            eprintln!("reading {input} failed: {e}");
            return ExitCode::from(1);
        }
    };
    if write_failed || out.flush().is_err() {
        eprintln!("write to {output} failed");
        return ExitCode::from(1);
    }

    let s = &stats;
    eprintln!("lines read         {}{}", s.lines, if truncated { " (stream ended mid-frame)" } else { "" });
    eprintln!("unreadable         {}", s.unreadable);
    eprintln!("mate score         {}", s.mate);
    eprintln!("depth < {MIN_DEPTH}         {}", s.shallow);
    eprintln!("bad FEN            {}", s.bad_fen);
    eprintln!("impossible pos.    {}", s.impossible);
    eprintln!("in check           {}", s.in_check);
    eprintln!("|cp| > {MAX_CP}       {}", s.extreme);
    eprintln!("insufficient mat.  {}", s.insufficient);
    eprintln!("not quiet          {}", s.not_quiet);
    eprintln!("kept               {}", s.kept);
    eprintln!();
    eprintln!("score vs handcrafted eval, White to move r = {:.3}", sign[0].r());
    eprintln!("score vs handcrafted eval, Black to move r = {:.3}", sign[1].r());

    if s.kept == 0 {
        eprintln!("FAIL: nothing kept");
        return ExitCode::from(1);
    }
    if sign[0].r() < MIN_SIGN_CORRELATION || sign[1].r() < MIN_SIGN_CORRELATION {
        eprintln!(
            "FAIL: scores do not track the handcrafted evaluation on both sides; \
             the White-relative assumption is wrong"
        );
        return ExitCode::from(1);
    }
    eprintln!("PASS: White-relative scores confirmed for both sides to move");
    ExitCode::SUCCESS
}
