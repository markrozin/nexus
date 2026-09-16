//! UCI protocol handler.
//!
//! Commands are read on the caller's thread (the main thread in `main.rs`) and
//! `go` hands the position to a worker thread, so `stop`, `isready`, and `quit`
//! stay responsive while a search runs. The two threads share an
//! `Arc<AtomicBool>` stop flag: the worker polls it with [`Ordering::Relaxed`],
//! which is sufficient because the flag carries no data — it only needs to
//! become visible eventually, and every real handoff (`join`) already
//! synchronizes.
//!
//! Output goes through a [`Sink`], which both threads hold a clone of. That is
//! what lets tests capture a whole session into a buffer.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::Position;
use crate::movegen::generate_legal;
use crate::nnue::{self, Network};
use crate::search::{score_to_uci, Evaluator, IterationInfo, Search, SearchLimits};
use crate::tt::TranspositionTable;
use crate::types::Color;

pub const ENGINE_NAME: &str = "Nexus";
pub const ENGINE_AUTHOR: &str = "Mark Rozin";

// ---------------------------------------------------------------------------
// Output sinks
// ---------------------------------------------------------------------------

/// Somewhere the engine can write protocol lines. Both the command loop and the
/// search worker hold a clone, so it must be cheap to clone and `Send`.
pub trait Sink: Write + Clone + Send + 'static {}
impl<T: Write + Clone + Send + 'static> Sink for T {}

/// Writes to the process's standard output.
///
/// A thin wrapper rather than [`io::Stdout`] directly, because the handler
/// needs a `Clone` sink and `Stdout` is not `Clone`.
#[derive(Clone, Copy, Default, Debug)]
pub struct StdoutSink;

impl Write for StdoutSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::stdout().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

/// An in-memory sink shared by every clone. Used by the tests to capture a
/// scripted session.
#[derive(Clone, Default, Debug)]
pub struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contents(&self) -> String {
        let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8(guard.clone()).expect("engine output is ASCII")
    }

    pub fn lines(&self) -> Vec<String> {
        self.contents().lines().map(str::to_owned).collect()
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Engine options. Parsed and stored, but nothing reads them yet: `Hash` waits
/// on the transposition table and `Threads` on parallel search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub hash_mb: usize,
    pub threads: usize,
}

impl Options {
    pub const HASH_DEFAULT: usize = 16;
    pub const HASH_MIN: usize = 1;
    /// Advertised maximum. `setoption` allocates immediately, so this has to
    /// be a size a real machine can supply; there is no point offering 64 GB
    /// to a single-threaded engine that cannot use it.
    pub const HASH_MAX: usize = crate::tt::MAX_MEGABYTES;
    pub const THREADS_DEFAULT: usize = 1;
    pub const THREADS_MIN: usize = 1;
    pub const THREADS_MAX: usize = 1_024;
}

impl Default for Options {
    fn default() -> Self {
        Self {
            hash_mb: Self::HASH_DEFAULT,
            threads: Self::THREADS_DEFAULT,
        }
    }
}

// ---------------------------------------------------------------------------
// Search limits
// ---------------------------------------------------------------------------

/// Everything a `go` command can ask for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    pub depth: Option<u32>,
    pub movetime: Option<u64>,
    pub nodes: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub infinite: bool,
}

fn arg<T: std::str::FromStr>(tokens: &[&str], i: usize) -> Option<T> {
    tokens.get(i + 1).and_then(|v| v.parse().ok())
}

impl Limits {
    pub fn parse(args: &str) -> Self {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        let mut limits = Limits::default();
        let mut i = 0;
        while i < tokens.len() {
            let mut consumed = 2;
            match tokens[i] {
                "depth" => limits.depth = arg(&tokens, i),
                "movetime" => limits.movetime = arg(&tokens, i),
                "nodes" => limits.nodes = arg(&tokens, i),
                "wtime" => limits.wtime = arg(&tokens, i),
                "btime" => limits.btime = arg(&tokens, i),
                "winc" => limits.winc = arg(&tokens, i),
                "binc" => limits.binc = arg(&tokens, i),
                "movestogo" => limits.movestogo = arg(&tokens, i),
                "infinite" => {
                    limits.infinite = true;
                    consumed = 1;
                }
                // `searchmoves`, `ponder`, and `mate` are accepted and ignored.
                _ => consumed = 1,
            }
            i += consumed;
        }
        limits
    }

    /// Wall-clock budget for this move, or `None` to return immediately.
    ///
    /// The clock-based branch is a placeholder allocation, not a real time
    /// manager: it will be replaced when iterative deepening lands and there is
    /// something to spend the time on.
    pub fn budget(&self, stm: Color) -> Option<Duration> {
        if self.infinite {
            return None;
        }
        if let Some(ms) = self.movetime {
            return Some(Duration::from_millis(ms));
        }
        let (remaining, inc) = match stm {
            Color::White => (self.wtime?, self.winc.unwrap_or(0)),
            Color::Black => (self.btime?, self.binc.unwrap_or(0)),
        };
        let moves_left = self.movestogo.unwrap_or(30).max(1) as u64;
        let alloc = remaining / moves_left + inc * 3 / 4;
        // Keep a reserve so we never flag while waiting on the poll interval.
        let capped = alloc.min(remaining.saturating_sub(50)).max(1);
        Some(Duration::from_millis(capped))
    }

}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Apply a move written in UCI coordinate notation.
///
/// Matching against generated legal moves rather than decoding the string means
/// castling, en passant, and promotions all fall out for free, and an illegal
/// or malformed move is rejected rather than corrupting the position.
pub fn apply_uci_move(pos: &Position, text: &str) -> Option<Position> {
    generate_legal(pos)
        .iter()
        .find(|mv| mv.to_string().eq_ignore_ascii_case(text))
        .map(|&mv| pos.make_move(mv))
}

pub struct Uci<W: Sink> {
    position: Position,
    /// Zobrist keys of every position before `position`, oldest first.
    history: Vec<u64>,
    /// Shared with the search worker and reused across moves; that reuse is
    /// the entire point of the table.
    tt: Arc<TranspositionTable>,
    out: W,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    options: Options,
    /// Leaf evaluation for every search, chosen by `EvalFile`.
    evaluator: Evaluator,
}

/// `EvalFile` values that name an evaluator rather than a file: the
/// handcrafted evaluation (the default, until a network passes an SPRT), and
/// the network compiled into the binary.
const HANDCRAFTED_EVAL_FILE: &str = "<handcrafted>";
const EMBEDDED_EVAL_FILE: &str = "<embedded>";

impl Uci<StdoutSink> {
    /// A handler wired to real stdout with an entropy-seeded PRNG.
    pub fn stdout() -> Self {
        Self::new(StdoutSink)
    }
}

impl<W: Sink> Uci<W> {
    pub fn new(out: W) -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            tt: Arc::new(TranspositionTable::new(Options::HASH_DEFAULT)),
            out,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            options: Options::default(),
            evaluator: Evaluator::Handcrafted,
        }
    }

    pub fn options(&self) -> Options {
        self.options
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Read commands until `quit` or end of input.
    pub fn run(&mut self, input: impl BufRead) -> io::Result<()> {
        for line in input.lines() {
            if !self.handle(&line?)? {
                break;
            }
        }
        self.stop_search();
        Ok(())
    }

    /// Handle one command. Returns `false` when the engine should exit.
    ///
    /// Unrecognized input is reported as an `info string` and otherwise
    /// ignored, as the protocol requires — a GUI may send commands this engine
    /// does not implement, and the session must survive them.
    pub fn handle(&mut self, line: &str) -> io::Result<bool> {
        // Strip a UTF-8 BOM. Some Windows tooling prefixes one to the first
        // line it writes to a pipe, which would otherwise turn `uci` into an
        // unknown command and hang the handshake with no clue why.
        let line = line.trim().trim_start_matches('\u{feff}').trim_start();
        let (cmd, args) = match line.split_once(char::is_whitespace) {
            Some((cmd, rest)) => (cmd, rest.trim()),
            None => (line, ""),
        };

        match cmd {
            "" => {}
            "uci" => self.cmd_uci()?,
            "isready" => self.send("readyok")?,
            "ucinewgame" => {
                self.stop_search();
                self.position = Position::startpos();
                self.history.clear();
                // Entries from the previous game are noise at best.
                self.tt.clear();
            }
            "setoption" => self.cmd_setoption(args)?,
            "position" => self.cmd_position(args)?,
            "go" => self.cmd_go(args)?,
            "stop" => self.stop_search(),
            "ponderhit" => {}
            "d" => {
                let board = format!("{:?}", self.position);
                for line in board.lines().filter(|l| !l.is_empty()) {
                    self.send(line)?;
                }
            }
            "quit" => {
                self.stop_search();
                return Ok(false);
            }
            other => self.send(&format!("info string unknown command {other:?}"))?,
        }
        Ok(true)
    }

    fn send(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.out, "{line}")?;
        self.out.flush()
    }

    fn cmd_uci(&mut self) -> io::Result<()> {
        self.send(&format!(
            "id name {ENGINE_NAME} {}",
            env!("CARGO_PKG_VERSION")
        ))?;
        self.send(&format!("id author {ENGINE_AUTHOR}"))?;
        self.send(&format!(
            "option name Hash type spin default {} min {} max {}",
            Options::HASH_DEFAULT,
            Options::HASH_MIN,
            Options::HASH_MAX
        ))?;
        self.send(&format!(
            "option name Threads type spin default {} min {} max {}",
            Options::THREADS_DEFAULT,
            Options::THREADS_MIN,
            Options::THREADS_MAX
        ))?;
        self.send(&format!(
            "option name EvalFile type string default {HANDCRAFTED_EVAL_FILE}"
        ))?;
        self.send("uciok")
    }

    fn cmd_setoption(&mut self, args: &str) -> io::Result<()> {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        let Some(name_at) = tokens.iter().position(|t| t.eq_ignore_ascii_case("name")) else {
            return self.send("info string setoption: missing 'name'");
        };
        let value_at = tokens.iter().position(|t| t.eq_ignore_ascii_case("value"));
        let name_end = value_at.unwrap_or(tokens.len());
        if name_end <= name_at + 1 {
            return self.send("info string setoption: missing option name");
        }
        // Option names may contain spaces, so take everything up to `value`.
        let name = tokens[name_at + 1..name_end].join(" ");
        let value = value_at.map(|i| tokens[i + 1..].join(" ")).unwrap_or_default();

        match name.to_ascii_lowercase().as_str() {
            "hash" => match value.parse::<usize>() {
                Ok(v) => {
                    self.options.hash_mb = v.clamp(Options::HASH_MIN, Options::HASH_MAX);
                    // Resizing means reallocating, so the old contents go.
                    self.tt = Arc::new(TranspositionTable::new(self.options.hash_mb));
                }
                Err(_) => return self.send(&format!("info string bad Hash value {value:?}")),
            },
            "threads" => match value.parse::<usize>() {
                Ok(v) => self.options.threads = v.clamp(Options::THREADS_MIN, Options::THREADS_MAX),
                Err(_) => return self.send(&format!("info string bad Threads value {value:?}")),
            },
            "evalfile" => {
                if value.is_empty() || value == HANDCRAFTED_EVAL_FILE {
                    self.evaluator = Evaluator::Handcrafted;
                } else if value == EMBEDDED_EVAL_FILE {
                    self.evaluator = Evaluator::Nnue(nnue::embedded());
                } else {
                    match Network::load(std::path::Path::new(&value)) {
                        // Leaked on purpose: the search holds `&'static`, and a
                        // GUI sets this once per session, not once per move.
                        Ok(net) => self.evaluator = Evaluator::Nnue(Box::leak(Box::new(net))),
                        // Keep the previous network rather than play without one.
                        Err(e) => return self.send(&format!("info string EvalFile {value:?}: {e}")),
                    }
                }
            }
            _ => return self.send(&format!("info string unknown option {name:?}")),
        }
        Ok(())
    }

    fn cmd_position(&mut self, args: &str) -> io::Result<()> {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        let (spec, moves) = match tokens.iter().position(|t| t.eq_ignore_ascii_case("moves")) {
            Some(i) => (&tokens[..i], &tokens[i + 1..]),
            None => (&tokens[..], &tokens[tokens.len()..]),
        };

        let mut pos = match spec.first().copied() {
            Some("startpos") => Position::startpos(),
            Some("fen") => match Position::from_fen(&spec[1..].join(" ")) {
                Ok(pos) => pos,
                Err(e) => return self.send(&format!("info string {e}")),
            },
            _ => return self.send("info string position: expected 'startpos' or 'fen'"),
        };

        // Apply to a scratch copy so a bad move leaves the old position intact.
        let mut history = Vec::with_capacity(moves.len());
        for text in moves {
            match apply_uci_move(&pos, text) {
                Some(next) => {
                    history.push(pos.zobrist());
                    pos = next;
                }
                None => {
                    return self.send(&format!(
                        "info string illegal move {text:?}; position unchanged"
                    ))
                }
            }
        }
        self.position = pos;
        self.history = history;
        Ok(())
    }

    fn cmd_go(&mut self, args: &str) -> io::Result<()> {
        // A `go` while already searching is a protocol violation, but finish
        // the old search rather than leaking the thread.
        self.stop_search();

        let limits = Limits::parse(args);
        let position = self.position;
        let history = self.history.clone();
        let tt = Arc::clone(&self.tt);
        let stop = Arc::clone(&self.stop);
        let out = self.out.clone();
        let evaluator = self.evaluator;

        stop.store(false, Ordering::Relaxed);
        self.worker = Some(thread::spawn(move || {
            run_search(position, &history, limits, stop, tt, evaluator, out);
        }));
        Ok(())
    }

    /// Signal the worker to stop and wait for it, so `bestmove` has been
    /// written by the time this returns.
    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.worker.take() {
            // A panic in the worker has already been reported by the default
            // hook; there is nothing useful to add here.
            let _ = handle.join();
        }
    }
}

impl<W: Sink> Drop for Uci<W> {
    fn drop(&mut self) {
        self.stop_search();
    }
}

// ---------------------------------------------------------------------------
// Search worker
// ---------------------------------------------------------------------------

/// Body of the search thread: run iterative deepening, stream an `info` line
/// per completed iteration, then answer with `bestmove`.
///
/// The search watches the stop flag itself, so there is no polling loop here;
/// `go infinite` simply has no depth or time limit to hit.
fn run_search<W: Sink>(
    pos: Position,
    history: &[u64],
    limits: Limits,
    stop: Arc<AtomicBool>,
    tt: Arc<TranspositionTable>,
    evaluator: Evaluator,
    mut out: W,
) {
    let search_limits = SearchLimits {
        max_depth: limits.depth,
        max_nodes: limits.nodes,
        budget: limits.budget(pos.side_to_move()),
    };

    let hashfull = Arc::clone(&tt);
    let mut search = Search::with_table(stop, tt);
    search.set_evaluator(evaluator);
    search.set_game_history(history);
    let result = search.run(&pos, search_limits, &mut |info| {
        report_iteration(&mut out, info, hashfull.permille_full());
    });

    let _ = writeln!(&mut out, "bestmove {}", result.best_move);
    let _ = out.flush();
}

/// One `info` line. Written field by field rather than through a collected
/// string, so reporting never allocates.
fn report_iteration<W: Sink>(out: &mut W, info: IterationInfo, hashfull: u32) {
    // Clamp to 1ms so a sub-millisecond iteration does not divide by zero.
    let ms = (info.elapsed.as_millis() as u64).max(1);
    let _ = write!(
        out,
        "info depth {} score {} nodes {} nps {} time {} hashfull {}",
        info.depth,
        score_to_uci(info.score),
        info.nodes,
        info.nodes * 1000 / ms,
        info.elapsed.as_millis(),
        hashfull,
    );
    if !info.pv.is_empty() {
        let _ = write!(out, " pv");
        for mv in info.pv {
            let _ = write!(out, " {mv}");
        }
    }
    let _ = writeln!(out);
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;
    use std::io::Cursor;

    /// Pipe a scripted session through the handler and collect its output.
    ///
    /// The handler is dropped before the buffer is read, so every worker thread
    /// has been joined and the transcript is complete.
    fn session(script: &str) -> Vec<String> {
        let buf = SharedBuffer::new();
        {
            let mut uci = Uci::new(buf.clone());
            uci.run(Cursor::new(script)).expect("session runs cleanly");
        }
        buf.lines()
    }

    fn id_line() -> String {
        format!("id name {ENGINE_NAME} {}", env!("CARGO_PKG_VERSION"))
    }

    #[test]
    fn handshake_is_exact() {
        assert_eq!(
            session("uci\nisready\nquit\n"),
            vec![
                id_line(),
                format!("id author {ENGINE_AUTHOR}"),
                "option name Hash type spin default 16 min 1 max 1024".to_string(),
                "option name Threads type spin default 1 min 1 max 1024".to_string(),
                "option name EvalFile type string default <handcrafted>".to_string(),
                "uciok".to_string(),
                "readyok".to_string(),
            ]
        );
    }

    #[test]
    fn scripted_session_produces_a_well_formed_transcript() {
        // Each `go` is followed by `stop`, which joins the worker. That makes
        // the interleaving of main-thread and worker output deterministic.
        let lines = session(concat!(
            "uci\n",
            "isready\n",
            "ucinewgame\n",
            "setoption name Hash value 64\n",
            "setoption name Threads value 4\n",
            "position startpos moves e2e4 e7e5\n",
            "isready\n",
            "go depth 3\n",
            "stop\n",
            "go movetime 20\n",
            "stop\n",
            "go infinite\n",
            "stop\n",
            "quit\n",
        ));

        assert_eq!(lines[0], id_line());
        assert_eq!(lines[5], "uciok");
        assert_eq!(lines.iter().filter(|l| *l == "readyok").count(), 2);
        assert!(
            !lines.iter().any(|l| l.contains("unknown option")),
            "{lines:#?}"
        );

        // The position after 1.e4 e5 has 29 legal moves; every answer must be
        // one of them, which also proves the right position was searched.
        let mut pos = Position::startpos();
        for text in ["e2e4", "e7e5"] {
            pos = apply_uci_move(&pos, text).expect("scripted moves are legal");
        }
        let legal: Vec<String> = generate_legal(&pos).iter().map(|mv| mv.to_string()).collect();
        assert_eq!(legal.len(), 29);

        let bestmoves: Vec<&String> = lines.iter().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves.len(), 3, "one bestmove per go: {lines:#?}");
        for line in &bestmoves {
            let mv = line.strip_prefix("bestmove ").expect("prefix");
            assert!(legal.contains(&mv.to_string()), "{mv:?} is not legal here");
        }

        // Every info line carries the mandatory fields, and the move a search
        // finally answers with has to be the one heading its last reported pv.
        let mut pv_head: Option<String> = None;
        let mut matched = 0;
        for line in &lines {
            if line.starts_with("info depth ") {
                for field in ["score ", "nodes ", "nps ", "time "] {
                    assert!(line.contains(field), "{line:?} is missing {field:?}");
                }
                if let Some(at) = line.find(" pv ") {
                    pv_head = line[at + 4..].split_whitespace().next().map(str::to_owned);
                }
            } else if let Some(mv) = line.strip_prefix("bestmove ") {
                if let Some(head) = pv_head.take() {
                    assert_eq!(head, mv, "bestmove must head the last pv: {lines:#?}");
                    matched += 1;
                }
            }
        }
        assert!(matched >= 2, "expected at least two searches to report a pv");
    }

    #[test]
    fn search_is_deterministic() {
        // No RNG in the engine any more: the same script must produce the same
        // moves. Only `bestmove` lines are compared, since node counts and
        // timings legitimately vary with how far a timed search gets.
        let script = "position startpos\ngo depth 3\nstop\ngo depth 3\nstop\nquit\n";
        let best = |lines: Vec<String>| -> Vec<String> {
            lines
                .into_iter()
                .filter(|l| l.starts_with("bestmove "))
                .collect()
        };
        let first = best(session(script));
        assert_eq!(first.len(), 2);
        assert_eq!(first, best(session(script)));
    }


    #[test]
    fn leading_byte_order_mark_is_ignored() {
        // A BOM on the first line must not swallow the handshake.
        let lines = session("\u{feff}uci\nisready\nquit\n");
        assert_eq!(lines[0], id_line());
        assert_eq!(lines.last().unwrap(), "readyok");
    }

    #[test]
    fn setoption_stores_and_clamps() {
        let mut uci = Uci::new(SharedBuffer::new());
        uci.handle("setoption name Hash value 256").unwrap();
        uci.handle("setoption name Threads value 8").unwrap();
        assert_eq!(
            uci.options(),
            Options {
                hash_mb: 256,
                threads: 8
            }
        );

        // Clamping is checked at small sizes on purpose: `setoption` allocates,
        // and asking for the advertised maximum here would reserve a gigabyte
        // for no benefit. `TranspositionTable::new` covers the huge case.
        uci.handle("setoption name Hash value 0").unwrap();
        assert_eq!(uci.options().hash_mb, Options::HASH_MIN);
        uci.handle("setoption name Threads value 0").unwrap();
        assert_eq!(uci.options().threads, Options::THREADS_MIN);
    }

    #[test]
    fn a_bad_eval_file_is_reported_and_the_engine_still_plays() {
        let buf = SharedBuffer::new();
        {
            let mut uci = Uci::new(buf.clone());
            uci.handle("setoption name EvalFile value no/such/network.bin").unwrap();
            uci.handle("setoption name EvalFile value <embedded>").unwrap();
            uci.handle("go depth 2").unwrap();
            uci.handle("stop").unwrap();
        }
        let out = buf.contents();
        assert!(out.contains("info string EvalFile \"no/such/network.bin\""), "{out}");
        assert!(out.contains("bestmove "), "{out}");
    }

    #[test]
    fn position_accepts_startpos_and_fen() {
        let mut uci = Uci::new(SharedBuffer::new());

        uci.handle("position startpos moves e2e4").unwrap();
        assert_eq!(uci.position().ep_square(), Square::from_uci("e3"));

        let kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        uci.handle(&format!("position fen {kiwipete}")).unwrap();
        assert_eq!(uci.position().to_fen(), kiwipete);

        uci.handle("position fen 8/8/8/8/8/8/8/K6k w - - 0 1 moves a1a2")
            .unwrap();
        assert_eq!(uci.position().to_fen(), "8/8/8/8/8/8/K7/7k b - - 1 1");
    }

    #[test]
    fn bad_input_is_reported_and_does_not_disturb_the_position() {
        let buf = SharedBuffer::new();
        let mut uci = Uci::new(buf.clone());
        uci.handle("position startpos moves e2e4").unwrap();
        let before = uci.position().to_fen();

        uci.handle("position startpos moves e2e4 e2e4").unwrap();
        uci.handle("position fen not-a-fen").unwrap();
        uci.handle("frobnicate").unwrap();
        uci.handle("setoption name Nonsense value 3").unwrap();

        assert_eq!(uci.position().to_fen(), before, "position must be untouched");
        let out = buf.contents();
        assert!(out.contains("illegal move \"e2e4\""), "{out}");
        assert!(out.contains("invalid FEN"), "{out}");
        assert!(out.contains("unknown command \"frobnicate\""), "{out}");
        assert!(out.contains("unknown option \"Nonsense\""), "{out}");
    }

    #[test]
    fn go_infinite_answers_only_after_stop() {
        let buf = SharedBuffer::new();
        let mut uci = Uci::new(buf.clone());
        uci.handle("go infinite").unwrap();

        // The worker is parked on the stop flag and cannot reach `bestmove`
        // until `stop` sets it, so this is not racy.
        assert!(!buf.contents().contains("bestmove"));

        uci.handle("stop").unwrap();
        assert!(buf.contents().contains("bestmove"));
    }

    #[test]
    fn quit_returns_false_and_joins_the_worker() {
        let buf = SharedBuffer::new();
        let mut uci = Uci::new(buf.clone());
        uci.handle("go infinite").unwrap();
        assert!(!uci.handle("quit").unwrap());
        assert!(buf.contents().contains("bestmove"));
    }

    #[test]
    fn checkmate_reports_a_null_bestmove() {
        let buf = SharedBuffer::new();
        let mut uci = Uci::new(buf.clone());
        // Fool's mate: White is mated and has nothing to play.
        uci.handle("position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .unwrap();
        uci.handle("go depth 1").unwrap();
        uci.handle("stop").unwrap();
        assert_eq!(buf.lines(), vec!["bestmove 0000".to_string()]);
    }

    #[test]
    fn parses_go_arguments() {
        let l = Limits::parse("wtime 300000 btime 250000 winc 2000 binc 1000 movestogo 40");
        assert_eq!(l.wtime, Some(300_000));
        assert_eq!(l.btime, Some(250_000));
        assert_eq!(l.winc, Some(2_000));
        assert_eq!(l.movestogo, Some(40));
        assert!(!l.infinite);
        assert!(l.budget(Color::White).is_some());

        let l = Limits::parse("depth 12 nodes 1000000 infinite");
        assert_eq!(l.depth, Some(12));
        assert_eq!(l.nodes, Some(1_000_000));
        assert!(l.infinite);
        assert_eq!(l.budget(Color::White), None, "infinite has no budget");

        assert_eq!(
            Limits::parse("movetime 250").budget(Color::Black),
            Some(Duration::from_millis(250))
        );
        // A bare `go` has nothing to wait for.
        assert_eq!(Limits::parse("").budget(Color::White), None);
        assert!(Limits::parse("depth 4").budget(Color::White).is_none());
    }
}
