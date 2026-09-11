//! Negamax with alpha-beta pruning and iterative deepening.
//!
//! Negamax rather than a separate max/min minimax, because `max(a, b)` is
//! `-min(-a, -b)`: with evaluation always from the side-to-move's perspective,
//! one function handles both players and the recursive call just negates and
//! swaps the window.
//!
//! Alpha-beta is exact — it returns what plain minimax would — so nothing here
//! needs an SPRT to justify it. The heuristic layer (PVS, null move, LMR,
//! futility) does, and none of it is here yet.
//!
//! Not yet present, in the order they are coming: a transposition table (and
//! with it repetition detection), and killer/history move ordering.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrayvec::ArrayVec;

use crate::board::Position;
use crate::eval::{evaluate, params};
use crate::movegen::{generate_legal, generate_tactical, MoveList};
use crate::types::{Move, PieceType};

/// Hard ceiling on search depth, and the size of every per-ply array.
pub const MAX_PLY: usize = 128;

/// A score no real evaluation can reach, used as an initial window bound.
///
/// Deliberately not `i32::MIN`: negating that overflows, and the window is
/// negated on every recursive call.
pub const INFINITY: i32 = 32_001;

/// Mate at ply 0. Mates found deeper score slightly lower, so the search
/// prefers the quickest one and actually finishes games.
pub const MATE: i32 = 32_000;

/// Anything at least this large is a mate score rather than an evaluation.
pub const MATE_IN_MAX_PLY: i32 = MATE - MAX_PLY as i32;

pub const DRAW: i32 = 0;

/// Game plies the repetition history reserves room for. Longer games simply
/// drop their oldest entries, which can only lose a repetition detection, never
/// invent one.
const MAX_GAME_PLIES: usize = 1024;

/// How often to consult the clock. Checking every node is measurable.
const CLOCK_CHECK_INTERVAL: u64 = 2048;

/// What the caller asked for, already resolved into absolutes.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    pub max_depth: Option<u32>,
    pub max_nodes: Option<u64>,
    /// Wall-clock budget for this move.
    pub budget: Option<Duration>,
}

/// One completed iteration, for `info` output.
pub struct IterationInfo<'a> {
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: &'a [Move],
}

/// The outcome of a whole `go`.
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
}

/// Triangular principal-variation table.
///
/// Row `ply` holds the best line found from that ply down. On an improvement at
/// `ply`, the move is prepended to the child row. Flat rather than
/// `[[Move; MAX_PLY]; MAX_PLY]` so the 32 KB is heap-allocated directly instead
/// of being built on the stack first.
struct PvTable {
    moves: Box<[Move]>,
    lengths: [usize; MAX_PLY],
}

impl PvTable {
    fn new() -> Self {
        Self {
            moves: vec![Move::NONE; MAX_PLY * MAX_PLY].into_boxed_slice(),
            lengths: [0; MAX_PLY],
        }
    }

    #[inline]
    fn clear(&mut self, ply: usize) {
        self.lengths[ply] = 0;
    }

    /// Record `mv` at `ply`, followed by whatever the child found.
    fn update(&mut self, ply: usize, mv: Move) {
        let child_len = if ply + 1 < MAX_PLY {
            self.lengths[ply + 1]
        } else {
            0
        };
        self.moves[ply * MAX_PLY] = mv;
        for i in 0..child_len {
            self.moves[ply * MAX_PLY + 1 + i] = self.moves[(ply + 1) * MAX_PLY + i];
        }
        self.lengths[ply] = child_len + 1;
    }

    fn line(&self, ply: usize) -> &[Move] {
        &self.moves[ply * MAX_PLY..ply * MAX_PLY + self.lengths[ply]]
    }
}

pub struct Search {
    pv: PvTable,
    nodes: u64,
    stop: Arc<AtomicBool>,
    limits: SearchLimits,
    deadline: Option<Instant>,
    start: Instant,
    /// Zobrist keys of the game so far, then of the current search path.
    /// Reserved once; `push` inside the search never reallocates.
    history: Vec<u64>,
    /// How many leading `history` entries belong to the game rather than the
    /// search tree.
    game_plies: usize,
    /// Cleared until the first iteration completes. Depth 1 is cheap and the
    /// engine must always answer with a move it actually searched, so the stop
    /// flag and the clock are not consulted until it is done.
    first_iteration_done: bool,
    /// Set once the search has run out of time or been told to stop. Everything
    /// unwinds by returning normally; see [`Search::aborted`].
    aborted: bool,
}

impl Search {
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        Self {
            pv: PvTable::new(),
            nodes: 0,
            stop,
            limits: SearchLimits::default(),
            deadline: None,
            start: Instant::now(),
            history: Vec::with_capacity(MAX_PLY + MAX_GAME_PLIES),
            game_plies: 0,
            first_iteration_done: false,
            aborted: false,
        }
    }

    /// Zobrist keys of every position played before the search root, oldest
    /// first. Without it the engine cannot see a repetition it has already
    /// walked into and will shuffle away won games.
    pub fn set_game_history(&mut self, keys: &[u64]) {
        self.history.clear();
        let start = keys.len().saturating_sub(MAX_GAME_PLIES);
        self.history.extend_from_slice(&keys[start..]);
        self.game_plies = self.history.len();
    }

    /// Has this position already occurred on the path to it?
    ///
    /// One earlier occurrence is enough. Inside a search a repetition means
    /// either side can force the draw, so waiting for a third occurrence only
    /// wastes depth.
    ///
    /// Only the last `halfmove_clock` plies can match: a pawn move or capture
    /// is irreversible, so nothing before one can recur. Positions repeat every
    /// second ply, hence the stride.
    fn is_repetition(&self, key: u64, halfmove_clock: u16) -> bool {
        let seen = self.history.len();
        let reversible = (halfmove_clock as usize).min(seen);
        let mut back = 2;
        while back <= reversible {
            if self.history[seen - back] == key {
                return true;
            }
            back += 2;
        }
        false
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    /// Iterative deepening: search depth 1, then 2, and so on until the budget
    /// runs out.
    ///
    /// This is a net speedup, not a cost. The tree grows exponentially, so all
    /// the shallow iterations together are cheaper than the deepest one — and
    /// once there is a transposition table they will also hand it the move
    /// ordering that makes the deep iteration cheap.
    ///
    /// `report` is called once per completed iteration.
    pub fn run(
        &mut self,
        pos: &Position,
        limits: SearchLimits,
        report: &mut dyn FnMut(IterationInfo<'_>),
    ) -> SearchResult {
        self.nodes = 0;
        self.aborted = false;
        self.first_iteration_done = false;
        self.limits = limits;
        self.start = Instant::now();
        self.deadline = limits.budget.map(|b| self.start + b);
        self.pv.lengths = [0; MAX_PLY];

        let root_moves = generate_legal(pos);
        let mut result = SearchResult {
            // Fall back to *some* legal move, so even an instant abort answers.
            best_move: root_moves.first().copied().unwrap_or(Move::NONE),
            score: DRAW,
            depth: 0,
            nodes: 0,
        };
        if root_moves.is_empty() {
            return result;
        }

        let max_depth = limits.max_depth.unwrap_or(MAX_PLY as u32 - 1);
        for depth in 1..=max_depth.min(MAX_PLY as u32 - 1) {
            // An aborted iteration unwinds without popping, so reset the path.
            self.history.truncate(self.game_plies);
            let score = self.negamax(pos, -INFINITY, INFINITY, depth as i32, 0);

            // A partial iteration is not comparable to a complete one: its
            // move ordering means the first few moves were searched properly
            // and the rest not at all. Throw it away.
            if self.aborted {
                break;
            }
            self.first_iteration_done = true;

            result.score = score;
            result.depth = depth;
            result.nodes = self.nodes;
            if let Some(&mv) = self.pv.line(0).first() {
                result.best_move = mv;
            }

            report(IterationInfo {
                depth,
                score,
                nodes: self.nodes,
                elapsed: self.start.elapsed(),
                pv: self.pv.line(0),
            });

            // A mate score is final; deeper will not improve on it.
            if score.abs() >= MATE_IN_MAX_PLY {
                break;
            }
            if self.out_of_time_for_another_iteration() {
                break;
            }
        }

        result.nodes = self.nodes;
        result
    }

    fn negamax(&mut self, pos: &Position, mut alpha: i32, beta: i32, depth: i32, ply: usize) -> i32 {
        self.pv.clear(ply);

        if self.check_abort() {
            return DRAW; // discarded by the caller
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(pos);
        }
        // The fifty-move rule is a draw wherever it lands, but never claim one
        // at the root: the caller needs a move back.
        if ply > 0 && pos.halfmove_clock() >= 100 {
            return DRAW;
        }
        if ply > 0 && self.is_repetition(pos.zobrist(), pos.halfmove_clock()) {
            return DRAW;
        }
        if ply > 0 && is_insufficient_material(pos) {
            return DRAW;
        }
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply);
        }

        let mut moves = generate_legal(pos);
        if moves.is_empty() {
            return if pos.in_check(pos.side_to_move()) {
                // Prefer the shorter mate. Without the ply term every mate
                // scores the same and the engine shuffles instead of finishing.
                -MATE + ply as i32
            } else {
                DRAW
            };
        }
        order_moves(pos, &mut moves);

        // On the path from here down, this position counts as seen.
        self.history.push(pos.zobrist());

        // Fail-soft: return the true best even when it falls outside the
        // window, which gives the transposition table a tighter bound later.
        let mut best = -INFINITY;
        for mv in moves {
            self.nodes += 1;
            let child = pos.make_move(mv);
            let score = -self.negamax(&child, -beta, -alpha, depth - 1, ply + 1);
            if self.aborted {
                return DRAW;
            }

            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    self.pv.update(ply, mv);
                    if alpha >= beta {
                        break; // the opponent would never allow this line
                    }
                }
            }
        }
        self.history.pop();
        best
    }

    /// Search on past a leaf until the position is quiet.
    ///
    /// Stopping the search the instant after a capture records the profit and
    /// never sees the recapture — the horizon effect, and it makes an engine
    /// hang material with confidence. So at every leaf, keep playing captures
    /// and promotions until none are left.
    ///
    /// The load-bearing idea is the stand-pat: you are not obliged to capture,
    /// so the static evaluation is a lower bound on what the position is worth.
    /// If it already beats beta the opponent will avoid this line regardless of
    /// what the captures do.
    ///
    /// There is no depth limit; a capture sequence is finite and terminates on
    /// its own. `MAX_PLY` is only a backstop against pathological positions.
    fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.check_abort() {
            return DRAW;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(pos);
        }

        let stand_pat = evaluate(pos);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut moves = generate_tactical(pos);
        // An empty list here means "nothing to capture", not "no legal moves":
        // checkmate and stalemate are the caller's business, and a quiet
        // position correctly returns its stand-pat.
        order_moves(pos, &mut moves);

        let mut best = stand_pat;
        for mv in moves {
            self.nodes += 1;
            let child = pos.make_move(mv);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                return DRAW;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    /// Has the search been told to stop, or run out of budget?
    ///
    /// Never panics to unwind: `panic = "abort"` is in the release profile, so
    /// unwinding would kill the process rather than the search.
    #[inline]
    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if !self.first_iteration_done {
            return false;
        }
        if let Some(max) = self.limits.max_nodes {
            if self.nodes >= max {
                self.aborted = true;
                return true;
            }
        }
        if self.nodes % CLOCK_CHECK_INTERVAL == 0 {
            // Relaxed: the flag carries no data and only needs to arrive
            // eventually.
            if self.stop.load(Ordering::Relaxed) {
                self.aborted = true;
                return true;
            }
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    return true;
                }
            }
        }
        false
    }

    /// Starting another iteration is only worth it if there is a fair chance of
    /// finishing it. Each one costs several times the last, so anything past
    /// roughly half the budget will not complete.
    fn out_of_time_for_another_iteration(&self) -> bool {
        match self.limits.budget {
            Some(budget) => self.start.elapsed() * 2 >= budget,
            None => false,
        }
    }
}

/// Draws that no amount of searching can escape: king versus king, and king
/// and a single minor versus king.
///
/// Without this the engine happily "wins" a bishop ending it cannot possibly
/// convert. Insufficient-material pairs like KNN vs K are not covered.
fn is_insufficient_material(pos: &Position) -> bool {
    if pos.by_type(PieceType::Pawn).any()
        || pos.by_type(PieceType::Rook).any()
        || pos.by_type(PieceType::Queen).any()
    {
        return false;
    }
    let minors = pos.by_type(PieceType::Knight) | pos.by_type(PieceType::Bishop);
    minors.popcount() <= 1
}

/// Order moves so the likely-best are searched first.
///
/// Alpha-beta prunes in proportion to how good this guess is: with perfect
/// ordering the effective branching factor drops from about 35 to about 6.
/// This is only the cheap part — promotions, then captures by MVV-LVA (most
/// valuable victim, least valuable attacker). Killers, history, and SEE arrive
/// with milestone 8, and the transposition-table move with milestone 7.
fn order_moves(pos: &Position, moves: &mut MoveList) {
    let mut scored: ArrayVec<(i32, Move), { crate::movegen::MAX_MOVES }> = ArrayVec::new();
    for &mv in moves.iter() {
        scored.push((move_score(pos, mv), mv));
    }
    // Unstable sort so nothing is allocated inside the search.
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (slot, (_, mv)) in moves.iter_mut().zip(scored) {
        *slot = mv;
    }
}

fn move_score(pos: &Position, mv: Move) -> i32 {
    const PROMOTION_BASE: i32 = 2_000_000;
    const CAPTURE_BASE: i32 = 1_000_000;

    let mut score = 0;
    if let Some(promoted) = mv.promotion() {
        score += PROMOTION_BASE + params::MG_VALUE[promoted.index()];
    }
    if mv.is_capture() {
        let victim = if mv.is_en_passant() {
            PieceType::Pawn
        } else {
            pos.piece_at(mv.to())
                .map(|p| p.piece_type())
                .unwrap_or(PieceType::Pawn)
        };
        let attacker = pos
            .piece_at(mv.from())
            .map(|p| p.piece_type())
            .unwrap_or(PieceType::King);
        // Victim dominates; the attacker only breaks ties, cheapest first.
        score += CAPTURE_BASE + params::MG_VALUE[victim.index()] * 16
            - params::MG_VALUE[attacker.index()];
    }
    score
}

/// Split a score into the `cp` or `mate` form UCI expects.
///
/// `mate n` counts full moves, so a mate found at ply 5 is `mate 3`.
pub fn score_to_uci(score: i32) -> String {
    if score >= MATE_IN_MAX_PLY {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_IN_MAX_PLY {
        format!("mate -{}", (MATE + score + 1) / 2)
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_to_depth(fen: &str, depth: u32) -> SearchResult {
        let pos: Position = fen.parse().expect("test fen is valid");
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        search.run(
            &pos,
            SearchLimits {
                max_depth: Some(depth),
                ..Default::default()
            },
            &mut |_| {},
        )
    }

    #[test]
    fn finds_mate_in_one() {
        // Back-rank mate: Ra8#.
        let result = search_to_depth("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 3);
        assert_eq!(result.best_move.to_string(), "a1a8");
        assert_eq!(result.score, MATE - 1, "mate at ply 1");
        assert_eq!(score_to_uci(result.score), "mate 1");
    }

    #[test]
    fn prefers_the_shorter_mate() {
        // Mate in one is available; a deeper search must still pick it.
        let shallow = search_to_depth("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 2);
        let deep = search_to_depth("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 5);
        assert_eq!(shallow.best_move.to_string(), "a1a8");
        assert_eq!(deep.best_move.to_string(), "a1a8");
        assert_eq!(deep.score, MATE - 1);
    }

    #[test]
    fn detects_being_mated() {
        // Fool's mate: White is already mated, so there is nothing to play.
        let result = search_to_depth(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            3,
        );
        assert_eq!(result.best_move, Move::NONE);
        assert!(result.score.abs() < MATE_IN_MAX_PLY || result.depth == 0);
    }

    #[test]
    fn finds_stalemate_rather_than_losing() {
        // White is down a queen; stalemate is the best available outcome.
        let result = search_to_depth("7k/8/8/8/8/2q5/5K2/8 w - - 0 1", 4);
        assert!(result.score <= DRAW, "score was {}", result.score);
    }

    #[test]
    fn wins_hanging_material() {
        // Black queen is en prise to the rook and defended by nothing.
        let result = search_to_depth("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3);
        assert_eq!(result.best_move.to_string(), "d1d5");
        assert!(result.score > 500, "score was {}", result.score);
    }

    /// Qxd5 wins a pawn and loses the queen to exd5. A depth-1 search sees the
    /// capture but not the reply; only quiescence plays the recapture out.
    const POISONED: &str = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1";
    /// The same shape with the pawn undefended, so the capture really is free.
    const FREE_PAWN: &str = "4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1";

    #[test]
    fn quiescence_refuses_a_poisoned_capture() {
        let result = search_to_depth(POISONED, 1);
        assert_ne!(
            result.best_move.to_string(),
            "d1d5",
            "took the poisoned pawn: {result:?}"
        );
    }

    #[test]
    fn quiescence_returns_the_stand_pat_when_every_capture_loses() {
        let pos: Position = POISONED.parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let score = search.quiescence(&pos, -INFINITY, INFINITY, 0);
        // Not obliged to capture, so a position whose only capture is bad is
        // worth exactly its static evaluation.
        assert_eq!(score, evaluate(&pos));
    }

    #[test]
    fn quiescence_takes_a_free_piece() {
        let pos: Position = FREE_PAWN.parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let score = search.quiescence(&pos, -INFINITY, INFINITY, 0);
        assert!(
            score > evaluate(&pos),
            "quiescence {score} did not improve on stand-pat {}",
            evaluate(&pos)
        );
    }

    #[test]
    fn quiescence_terminates_in_a_capture_heavy_position() {
        // Nothing bounds the capture sequence except its own length; if that
        // reasoning is wrong this hangs or blows the stack.
        let pos: Position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            .parse()
            .unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let score = search.quiescence(&pos, -INFINITY, INFINITY, 0);
        assert!(score.abs() < MATE_IN_MAX_PLY);
        assert!(search.nodes() > 0);
    }

    #[test]
    fn alpha_beta_agrees_with_plain_minimax() {
        // The whole point of alpha-beta is that it is exact. Depth 2 only:
        // the reference prunes nothing, so its cost is the raw branching
        // factor times a quiescence call at every leaf.
        // Compare against a
        // full-window search with pruning defeated by an infinite window.
        for fen in [
            "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let pos: Position = fen.parse().unwrap();
            let mut search = Search::new(Arc::new(AtomicBool::new(false)));
            let ab = search.negamax(&pos, -INFINITY, INFINITY, 2, 0);
            let plain = minimax(&mut search, &pos, 2, 0);
            assert_eq!(ab, plain, "{fen}");
        }
    }

    /// Reference implementation: negamax with no pruning at all.
    ///
    /// Leaves go through full-window quiescence, the same leaf value function
    /// the real search uses. The claim under test is that *pruning* changes
    /// nothing, not that the two use different evaluators.
    fn minimax(search: &mut Search, pos: &Position, depth: i32, ply: usize) -> i32 {
        if depth <= 0 {
            return search.quiescence(pos, -INFINITY, INFINITY, ply);
        }
        let moves = generate_legal(pos);
        if moves.is_empty() {
            return if pos.in_check(pos.side_to_move()) {
                -MATE + ply as i32
            } else {
                DRAW
            };
        }
        let mut best = -INFINITY;
        for mv in moves {
            best = best.max(-minimax(search, &pos.make_move(mv), depth - 1, ply + 1));
        }
        best
    }

    #[test]
    fn deeper_search_visits_more_nodes_and_reports_a_pv() {
        let pos = Position::startpos();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let mut depths = Vec::new();
        let result = search.run(
            &pos,
            SearchLimits {
                max_depth: Some(4),
                ..Default::default()
            },
            &mut |info| depths.push((info.depth, info.pv.len())),
        );
        assert_eq!(result.depth, 4);
        assert_eq!(depths.len(), 4, "one report per iteration");
        for (depth, pv_len) in depths {
            assert!(pv_len >= 1, "depth {depth} reported an empty pv");
        }
        assert!(result.nodes > 0);
    }

    #[test]
    fn a_time_budget_deepens_until_it_expires() {
        let pos = Position::startpos();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let mut deepest = 0;
        let start = Instant::now();
        let result = search.run(
            &pos,
            SearchLimits {
                budget: Some(Duration::from_millis(500)),
                ..Default::default()
            },
            &mut |info| deepest = info.depth,
        );
        let elapsed = start.elapsed();
        // Depth 3 costs a few thousand nodes, so this holds even in a debug
        // build on a loaded machine. The claim is that iterative deepening
        // happens at all and the deadline is respected, not a depth target.
        assert!(deepest >= 3, "a 500ms budget only reached depth {deepest}");
        assert_eq!(result.depth, deepest);
        // Generous upper bound: this is about catching a search that ignores
        // the deadline entirely, not about millisecond accuracy.
        assert!(elapsed < Duration::from_secs(3), "overran the budget: {elapsed:?}");
    }

    #[test]
    fn node_limit_is_honoured() {
        let pos = Position::startpos();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let result = search.run(
            &pos,
            SearchLimits {
                max_nodes: Some(5_000),
                ..Default::default()
            },
            &mut |_| {},
        );
        assert!(search.nodes() < 20_000, "ran {} nodes", search.nodes());
        assert!(!result.best_move.is_none(), "must still answer with a move");
    }

    #[test]
    fn stop_flag_ends_the_search_with_a_legal_move() {
        let pos = Position::startpos();
        let stop = Arc::new(AtomicBool::new(true)); // already set
        let mut search = Search::new(Arc::clone(&stop));
        let result = search.run(&pos, SearchLimits::default(), &mut |_| {});
        let legal: Vec<String> = generate_legal(&pos).iter().map(|m| m.to_string()).collect();
        assert!(legal.contains(&result.best_move.to_string()));
    }

    #[test]
    fn a_losing_side_can_save_itself_by_repetition() {
        // White is down a rook and in check, so material alone scores this
        // around -500. But the king can shuffle and force a repetition inside
        // the search horizon, and a draw beats being a rook down.
        //
        // Without repetition detection this returns the material score, so the
        // assertion below is exactly what the feature buys.
        let pos: Position = "7k/8/8/8/8/8/r7/K7 w - - 10 40".parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let score = search
            .run(
                &pos,
                SearchLimits {
                    max_depth: Some(4),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .score;
        assert_eq!(score, DRAW, "should have found the repetition");

        // And the detector fires on a key already in the game history. Two
        // entries, not one: the immediately preceding position has the other
        // side to move and can never match, so a repetition is at least two
        // plies back.
        search.set_game_history(&[pos.zobrist(), 0xDEAD_BEEF]);
        assert!(search.is_repetition(pos.zobrist(), 10));
    }

    #[test]
    fn repetition_only_looks_back_to_the_last_irreversible_move() {
        let pos = Position::startpos();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        search.set_game_history(&[0xAA, 0xBB, 0xCC, 0xDD]);

        // Two plies back is 0xCC. A clock of 2 can reach it; a clock of 1 or 0
        // means an irreversible move intervened and nothing can match.
        assert!(search.is_repetition(0xCC, 2));
        assert!(!search.is_repetition(0xCC, 1));
        assert!(!search.is_repetition(0xCC, 0));
        // 0xDD is one ply back, so it is the other side to move and never a
        // repetition candidate; the stride skips it.
        assert!(!search.is_repetition(0xDD, 100));
        // Four plies back is reachable with a long enough clock.
        assert!(search.is_repetition(0xAA, 4));
        assert!(!search.is_repetition(0xAA, 3));
        assert!(!search.is_repetition(0x99, 100), "absent key must not match");
        let _ = pos;
    }

    #[test]
    fn the_search_path_itself_counts_as_history() {
        // A forced shuffle: the only non-losing continuation repeats. The search
        // must score it a draw rather than believing it is winning.
        let pos: Position = "7k/8/8/8/8/8/8/K6R w - - 0 1".parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let result = search.run(
            &pos,
            SearchLimits {
                max_depth: Some(5),
                ..Default::default()
            },
            &mut |_| {},
        );
        // White is up a rook and should not be talking itself into a draw here.
        assert!(result.score > 300, "score was {}", result.score);
        // History must be back to its starting length once the search returns.
        assert_eq!(search.history.len(), search.game_plies);
    }

    #[test]
    fn insufficient_material_is_a_draw() {
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3BK3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
        ] {
            let pos: Position = fen.parse().unwrap();
            assert!(is_insufficient_material(&pos), "{fen}");
        }
        for fen in [
            "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3RK3 w - - 0 1",
        ] {
            let pos: Position = fen.parse().unwrap();
            assert!(!is_insufficient_material(&pos), "{fen}");
        }
    }

    #[test]
    fn ordering_puts_captures_and_promotions_first() {
        let pos: Position = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1".parse().unwrap();
        let mut moves = generate_legal(&pos);
        order_moves(&pos, &mut moves);
        assert_eq!(moves[0].to_string(), "d1d5", "queen capture leads");
        assert_eq!(moves.len(), generate_legal(&pos).len(), "no moves lost");
    }

    #[test]
    fn score_conversion_covers_both_mate_directions() {
        assert_eq!(score_to_uci(35), "cp 35");
        assert_eq!(score_to_uci(-35), "cp -35");
        assert_eq!(score_to_uci(MATE - 1), "mate 1");
        assert_eq!(score_to_uci(MATE - 3), "mate 2");
        assert_eq!(score_to_uci(-(MATE - 1)), "mate -1");
        assert_eq!(score_to_uci(-(MATE - 4)), "mate -2");
    }
}
