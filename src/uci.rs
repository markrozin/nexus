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
use std::time::{Duration, Instant};

use crate::board::Position;
use crate::movegen::generate_legal;
use crate::rng::Rng;
use crate::types::{Color, Move};

pub const ENGINE_NAME: &str = "newchessbot";
pub const ENGINE_AUTHOR: &str = "Mark Rozin";

/// How often a waiting worker re-reads the stop flag.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

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
    pub const HASH_MAX: usize = 65_536;
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

    /// Does this `go` need the worker to wait before answering?
    fn is_timed(&self) -> bool {
        self.infinite || self.movetime.is_some() || self.wtime.is_some() || self.btime.is_some()
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
    out: W,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    options: Options,
    rng: Rng,
}

impl Uci<StdoutSink> {
    /// A handler wired to real stdout with an entropy-seeded PRNG.
    pub fn stdout() -> Self {
        Self::new(StdoutSink)
    }
}

impl<W: Sink> Uci<W> {
    pub fn new(out: W) -> Self {
        Self::with_rng(out, Rng::from_entropy())
    }

    /// Fixed-seed constructor, so a scripted session is reproducible.
    pub fn with_seed(out: W, seed: u64) -> Self {
        Self::with_rng(out, Rng::new(seed))
    }

    fn with_rng(out: W, rng: Rng) -> Self {
        Self {
            position: Position::startpos(),
            out,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            options: Options::default(),
            rng,
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
            }
            "setoption" => self.cmd_setoption(args)?,
            "position" => self.cmd_position(args)?,
            "go" => self.cmd_go(args)?,
            "stop" => self.stop_search(),
            "ponderhit" => {}
            "d" => {
                let board = self.position.to_string();
                for line in board.lines() {
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
                Ok(v) => self.options.hash_mb = v.clamp(Options::HASH_MIN, Options::HASH_MAX),
                Err(_) => return self.send(&format!("info string bad Hash value {value:?}")),
            },
            "threads" => match value.parse::<usize>() {
                Ok(v) => self.options.threads = v.clamp(Options::THREADS_MIN, Options::THREADS_MAX),
                Err(_) => return self.send(&format!("info string bad Threads value {value:?}")),
            },
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
        for text in moves {
            match apply_uci_move(&pos, text) {
                Some(next) => pos = next,
                None => {
                    return self.send(&format!(
                        "info string illegal move {text:?}; position unchanged"
                    ))
                }
            }
        }
        self.position = pos;
        Ok(())
    }

    fn cmd_go(&mut self, args: &str) -> io::Result<()> {
        // A `go` while already searching is a protocol violation, but finish
        // the old search rather than leaking the thread.
        self.stop_search();

        let limits = Limits::parse(args);
        let position = self.position;
        let stop = Arc::clone(&self.stop);
        let out = self.out.clone();
        let rng = Rng::new(self.rng.next_u64());

        stop.store(false, Ordering::Relaxed);
        self.worker = Some(thread::spawn(move || {
            run_search(position, limits, stop, out, rng);
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
// Placeholder search
// ---------------------------------------------------------------------------

/// Stand-in for the real search: picks a uniformly random legal move.
///
/// It still respects the stop flag and the time budget, so the threading and
/// protocol plumbing is exercised for real before minimax replaces the move
/// choice. No score is reported because there is no evaluation yet.
fn run_search<W: Sink>(pos: Position, limits: Limits, stop: Arc<AtomicBool>, mut out: W, mut rng: Rng) {
    let moves = generate_legal(&pos);
    let best = match rng.choose(moves.len()) {
        Some(i) => moves[i],
        None => Move::NONE,
    };

    if !best.is_none() {
        let _ = writeln!(&mut out, "info depth 1 nodes {} pv {best}", moves.len());
    }

    if limits.is_timed() {
        wait_out_the_clock(&limits, &stop, pos.side_to_move());
    }

    let _ = writeln!(&mut out, "bestmove {best}");
    let _ = out.flush();
}

/// Block until the budget expires or `stop` is set, whichever comes first.
fn wait_out_the_clock(limits: &Limits, stop: &AtomicBool, stm: Color) {
    let deadline = limits.budget(stm).map(|budget| Instant::now() + budget);
    loop {
        // Relaxed is enough: this flag carries no data, and we only need to
        // observe the write eventually.
        if stop.load(Ordering::Relaxed) {
            return;
        }
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                return;
            }
        }
        thread::sleep(POLL_INTERVAL);
    }
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
            let mut uci = Uci::with_seed(buf.clone(), 0x00C0_FFEE);
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
                "option name Hash type spin default 16 min 1 max 65536".to_string(),
                "option name Threads type spin default 1 min 1 max 1024".to_string(),
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
            "go depth 4\n",
            "stop\n",
            "go movetime 5\n",
            "stop\n",
            "go infinite\n",
            "stop\n",
            "quit\n",
        ));

        assert_eq!(lines[0], id_line());
        assert_eq!(lines[4], "uciok");
        assert_eq!(lines.iter().filter(|l| *l == "readyok").count(), 2);
        // Every option we set is one we advertised, so nothing was rejected.
        assert!(
            !lines.iter().any(|l| l.contains("unknown option")),
            "{lines:#?}"
        );

        let search_output: Vec<&String> = lines
            .iter()
            .filter(|l| l.starts_with("info depth") || l.starts_with("bestmove"))
            .collect();
        assert_eq!(search_output.len(), 6, "three searches, two lines each");

        // The position after 1.e4 e5 has 29 legal moves; the `go` commands must
        // have searched that position and not, say, the start position.
        let mut pos = Position::startpos();
        for text in ["e2e4", "e7e5"] {
            pos = apply_uci_move(&pos, text).expect("scripted moves are legal");
        }
        let legal: Vec<String> = generate_legal(&pos).iter().map(|mv| mv.to_string()).collect();
        assert_eq!(legal.len(), 29);

        // Each search reports one info line naming its move, then that move.
        for pair in search_output.chunks(2) {
            let mv = pair[1]
                .strip_prefix("bestmove ")
                .expect("second line of each pair is the bestmove");
            assert!(legal.contains(&mv.to_string()), "{mv:?} is not legal here");
            assert_eq!(pair[0], &format!("info depth 1 nodes 29 pv {mv}"));
        }
    }

    #[test]
    fn leading_byte_order_mark_is_ignored() {
        // A BOM on the first line must not swallow the handshake.
        let lines = session("\u{feff}uci\nisready\nquit\n");
        assert_eq!(lines[0], id_line());
        assert_eq!(lines.last().unwrap(), "readyok");
    }

    #[test]
    fn same_seed_replays_identically() {
        let script = "position startpos\ngo depth 2\nstop\ngo depth 2\nstop\nquit\n";
        assert_eq!(session(script), session(script));
    }

    #[test]
    fn setoption_stores_and_clamps() {
        let mut uci = Uci::with_seed(SharedBuffer::new(), 1);
        uci.handle("setoption name Hash value 256").unwrap();
        uci.handle("setoption name Threads value 8").unwrap();
        assert_eq!(
            uci.options(),
            Options {
                hash_mb: 256,
                threads: 8
            }
        );

        uci.handle("setoption name Hash value 999999999").unwrap();
        assert_eq!(uci.options().hash_mb, Options::HASH_MAX);
        uci.handle("setoption name Threads value 0").unwrap();
        assert_eq!(uci.options().threads, Options::THREADS_MIN);
    }

    #[test]
    fn position_accepts_startpos_and_fen() {
        let mut uci = Uci::with_seed(SharedBuffer::new(), 1);

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
        let mut uci = Uci::with_seed(buf.clone(), 1);
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
        let mut uci = Uci::with_seed(buf.clone(), 99);
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
        let mut uci = Uci::with_seed(buf.clone(), 3);
        uci.handle("go infinite").unwrap();
        assert!(!uci.handle("quit").unwrap());
        assert!(buf.contents().contains("bestmove"));
    }

    #[test]
    fn checkmate_reports_a_null_bestmove() {
        let buf = SharedBuffer::new();
        let mut uci = Uci::with_seed(buf.clone(), 5);
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
        assert!(!Limits::parse("depth 4").is_timed());
    }
}
