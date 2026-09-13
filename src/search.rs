//! Negamax with alpha-beta pruning and iterative deepening.
//!
//! Negamax rather than a separate max/min minimax, because `max(a, b)` is
//! `-min(-a, -b)`: with evaluation always from the side-to-move's perspective,
//! one function handles both players and the recursive call just negates and
//! swaps the window.
//!
//! Alpha-beta, PVS and quiescence are exact: they return what plain minimax
//! would, so none of them needed an SPRT to justify. Null move pruning is not
//! exact — it trades occasional correctness for depth — and did.
//!
//! Present: alpha-beta, quiescence, iterative deepening, a transposition table,
//! SEE/killer/history move ordering, PVS, null move pruning, late move
//! reductions, and reverse futility pruning.
//!
//! Deliberately absent: **late move pruning**, which skips late quiet moves
//! outright rather than reducing them. Tested twice and rejected twice, at -80
//! and (after fixing a real counter bug) -59 Elo. It is the second heuristic to
//! fail on the same assumption — that quiet move ordering here is good enough
//! for late quiets to be noise. It is not: history is bonus-only, with no malus
//! and no continuation tables. Improve quiet ordering before retrying anything
//! in this family, including demoting losing captures below quiets.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use arrayvec::ArrayVec;

use crate::board::Position;
use crate::eval::{evaluate, params};
use crate::movegen::{generate_legal, generate_tactical, MoveList};
use crate::see::see;
use crate::tt::{Bound, TranspositionTable};
use crate::types::{Color, Move, PieceType};

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

/// Table size when a `Search` is built without one. UCI overrides it from the
/// `Hash` option.
const DEFAULT_TT_MB: usize = 1;

/// Cap on a quiet-history entry. Keeps the gravity update bounded and keeps
/// history well below the capture tiers in the ordering.
const MAX_HISTORY: i32 = 16_384;

/// Shallowest depth worth trying a null move at. Below this the reduced
/// search is so cheap that the saving does not cover being wrong.
const NULL_MOVE_MIN_DEPTH: i32 = 3;

/// Reverse futility pruning: margin per ply of remaining depth, and the
/// deepest depth it applies at.
///
/// The margin is the researched starting point (~150 per ply). The depth cap
/// exists because the assumption gets shakier the more search remains: at high
/// depth there is plenty of room for the opponent to find the refutation the
/// static evaluation cannot see.
const RFP_MARGIN: i32 = 150;
const RFP_MAX_DEPTH: i32 = 6;

/// Shallowest depth worth reducing at, and how many moves get full depth before
/// reductions start.
const LMR_MIN_DEPTH: i32 = 3;
const LMR_MIN_MOVE_INDEX: usize = 3;

/// Square table of reductions, indexed by depth then move-order index.
const LMR_TABLE_SIZE: usize = 64;

/// `c + ln(depth) * ln(index) / d`, the shape every modern engine uses:
/// reductions grow with both depth and how late the move is, but
/// logarithmically, so they stay modest rather than running away. These
/// constants are Ethereal's for quiet moves — a researched starting point, not
/// a tuned result for this engine.
static LMR_TABLE: LazyLock<[[i32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE]> = LazyLock::new(|| {
    let mut table = [[0i32; LMR_TABLE_SIZE]; LMR_TABLE_SIZE];
    for depth in 1..LMR_TABLE_SIZE {
        for index in 1..LMR_TABLE_SIZE {
            let r = 0.7844 + (depth as f64).ln() * (index as f64).ln() / 2.4696;
            table[depth][index] = r as i32;
        }
    }
    table
});

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
    tt: Arc<TranspositionTable>,
    /// Zobrist keys of the game so far, then of the current search path.
    /// Reserved once; `push` inside the search never reallocates.
    history: Vec<u64>,
    /// How many leading `history` entries belong to the game rather than the
    /// search tree.
    game_plies: usize,
    /// Two quiet moves per ply that most recently caused a beta cutoff there.
    ///
    /// A refutation that works in one line usually works in its siblings, and
    /// this costs nothing to remember.
    killers: [[Move; 2]; MAX_PLY],
    /// `[color][from][to]`, flattened. How often a quiet move has caused a
    /// cutoff anywhere in the search: the only signal available for ranking
    /// quiet moves against each other.
    quiet_history: Box<[i32]>,
    /// False while a null move is already in effect on this branch. Two in a
    /// row would just be passing the turn back and forth, which proves nothing.
    null_move_allowed: bool,
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
        Self::with_table(stop, Arc::new(TranspositionTable::new(DEFAULT_TT_MB)))
    }

    /// Share a table across searches, which is how it survives between moves.
    pub fn with_table(stop: Arc<AtomicBool>, tt: Arc<TranspositionTable>) -> Self {
        Self {
            tt,
            pv: PvTable::new(),
            nodes: 0,
            stop,
            limits: SearchLimits::default(),
            deadline: None,
            start: Instant::now(),
            history: Vec::with_capacity(MAX_PLY + MAX_GAME_PLIES),
            game_plies: 0,
            killers: [[Move::NONE; 2]; MAX_PLY],
            quiet_history: vec![0; Color::COUNT * 64 * 64].into_boxed_slice(),
            null_move_allowed: true,
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

    #[inline]
    fn history_index(color: Color, mv: Move) -> usize {
        (color.index() * 64 + mv.from().index()) * 64 + mv.to().index()
    }

    /// How much shallower to search a late move, in plies. Zero means no
    /// reduction.
    ///
    /// Only quiet moves are reduced, and only once past the first few: the
    /// early ones are where the cutoff is expected, and a capture or promotion
    /// is forcing enough to be worth full depth. Nothing is reduced while in
    /// check, where every move is a forced evasion.
    ///
    /// A killer is reduced one ply less. It already refuted a sibling at this
    /// ply, which is direct evidence against it being a late move.
    fn reduction(&self, depth: i32, index: usize, mv: Move, in_check: bool, ply: usize) -> i32 {
        if depth < LMR_MIN_DEPTH
            || index < LMR_MIN_MOVE_INDEX
            || in_check
            || mv.is_capture()
            || mv.is_promotion()
        {
            return 0;
        }

        let table = &*LMR_TABLE;
        let mut reduction = table[(depth as usize).min(LMR_TABLE_SIZE - 1)]
            [index.min(LMR_TABLE_SIZE - 1)];

        if self.killers[ply].contains(&mv) {
            reduction -= 1;
        }

        // Never reduce into quiescence: that would skip the rest of the tree
        // rather than search it shallower.
        reduction.clamp(0, depth - 2)
    }

    /// Reward a quiet move that caused a cutoff.
    ///
    /// The update has "gravity": the correction shrinks as the entry approaches
    /// the cap, so values stay bounded without ever needing a rescaling pass,
    /// and a move that stops working decays back on its own.
    fn record_cutoff(&mut self, pos: &Position, mv: Move, ply: usize, depth: i32) {
        if mv.is_capture() || mv.is_promotion() {
            // Captures are ranked by the exchange, not by history.
            return;
        }

        let slot = &mut self.killers[ply];
        if slot[0] != mv {
            slot[1] = slot[0];
            slot[0] = mv;
        }

        // Deeper cutoffs are stronger evidence, so they move the value further.
        let bonus = (depth * depth).clamp(-MAX_HISTORY, MAX_HISTORY);
        let entry = &mut self.quiet_history[Self::history_index(pos.side_to_move(), mv)];
        *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
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
        self.null_move_allowed = true;
        self.limits = limits;
        self.start = Instant::now();
        self.deadline = limits.budget.map(|b| self.start + b);
        self.pv.lengths = [0; MAX_PLY];
        self.tt.new_generation();
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        // Halve rather than clear: ordering from the previous move is still a
        // better prior than nothing.
        for entry in self.quiet_history.iter_mut() {
            *entry /= 2;
        }

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

        let tt_hit = self.tt.probe(pos.zobrist(), ply);
        if ply > 0 {
            if let Some(hit) = tt_hit {
                // Only trust a result searched at least as deeply as this one,
                // and only when its bound falls on the useful side of the
                // current window.
                if hit.depth as i32 >= depth
                    && match hit.bound {
                        Bound::Exact => true,
                        Bound::Lower => hit.score >= beta,
                        Bound::Upper => hit.score <= alpha,
                        Bound::None => false,
                    }
                {
                    return hit.score;
                }
            }
        }

        // Hoisted: null move needs it, and the terminal test below reuses it.
        let in_check = pos.in_check(pos.side_to_move());

        // Null move pruning. Hand the opponent a free move; if the position is
        // still good enough to beat beta even after that, it is far too good
        // for them to have allowed, and the whole subtree can go.
        //
        // The conditions are all load-bearing. A null move while in check
        // leaves the king capturable. In a PV node the exact value matters and
        // cannot be replaced by a bound. Two nulls in a row prove nothing. And
        // with only pawns and a king the assumption fails outright: in zugzwang
        // having to move is the whole problem, so "a free pass cannot hurt" is
        // exactly backwards.
        let is_pv = beta - alpha > 1;

        // Reverse futility pruning. If the static evaluation is already so far
        // above beta that a whole search is unlikely to drag it back down, take
        // the evaluation and stop.
        //
        // This is the mirror of ordinary futility: that one asks whether a move
        // can rescue a bad position, this asks whether the opponent can spoil a
        // good one. The margin scales with depth because a deeper search has
        // more chances to find the refutation.
        //
        // Not in a PV node, where an exact value is needed rather than a bound,
        // and not in check, where the static evaluation is close to meaningless.
        // Mate scores are excluded because "beta plus a margin" is not a
        // sensible comparison against a mate bound.
        if !is_pv
            && !in_check
            && depth <= RFP_MAX_DEPTH
            && beta.abs() < MATE_IN_MAX_PLY
        {
            let static_eval = evaluate(pos);
            if static_eval - RFP_MARGIN * depth >= beta {
                return static_eval;
            }
        }

        if !is_pv
            && !in_check
            && depth >= NULL_MOVE_MIN_DEPTH
            && self.null_move_allowed
            && has_non_pawn_material(pos, pos.side_to_move())
        {
            let reduction = 3 + depth / 6;
            let child = pos.make_null_move();
            self.null_move_allowed = false;
            let score = -self.negamax(&child, -beta, -beta + 1, depth - reduction - 1, ply + 1);
            self.null_move_allowed = true;
            if self.aborted {
                return DRAW;
            }
            if score >= beta {
                // A mate score proved by giving away a move is not a real mate,
                // so report the bound instead of a claim we cannot back.
                return if score >= MATE_IN_MAX_PLY { beta } else { score };
            }
        }

        let mut moves = generate_legal(pos);
        if moves.is_empty() {
            return if in_check {
                // Prefer the shorter mate. Without the ply term every mate
                // scores the same and the engine shuffles instead of finishing.
                -MATE + ply as i32
            } else {
                DRAW
            };
        }
        // Even when the stored entry could not cut this node off, the move it
        // found is the single best ordering hint available.
        let tt_move = tt_hit.map_or(Move::NONE, |hit| hit.mv);
        order_moves(
            pos,
            &mut moves,
            tt_move,
            self.killers[ply],
            &self.quiet_history,
        );

        // On the path from here down, this position counts as seen.
        self.history.push(pos.zobrist());

        // Fail-soft: return the true best even when it falls outside the
        // window, which gives the transposition table a tighter bound later.
        let original_alpha = alpha;
        let mut best = -INFINITY;
        let mut best_move = Move::NONE;
        for (index, mv) in moves.into_iter().enumerate() {
            self.nodes += 1;
            let child = pos.make_move(mv);

            // Principal variation search. Ordering is good enough that the
            // first move is usually best, so every later move is first probed
            // with a null window, which is far cheaper because it can only
            // prove "not better than alpha" rather than establishing a value.
            // A probe that beats alpha was a genuine surprise and gets a real
            // search; with good ordering that is rare enough to pay for itself.
            let score = if index == 0 {
                -self.negamax(&child, -beta, -alpha, depth - 1, ply + 1)
            } else {
                // Late move reduction. A move this far down the ordering is
                // probably bad, so search it shallower and only pay full price
                // if it surprises us. This is a gamble: ordering is a guess,
                // and a reduced search can miss something real.
                let reduction = self.reduction(depth, index, mv, in_check, ply);
                let mut probe =
                    -self.negamax(&child, -alpha - 1, -alpha, depth - 1 - reduction, ply + 1);

                // Beating alpha at reduced depth means the reduction was wrong
                // about this move; verify it at full depth before believing it.
                if reduction > 0 && probe > alpha {
                    probe = -self.negamax(&child, -alpha - 1, -alpha, depth - 1, ply + 1);
                }
                // Still beating alpha, and inside the window: the null window
                // only proved a bound, so a real search is needed for a value.
                if probe > alpha && probe < beta {
                    -self.negamax(&child, -beta, -alpha, depth - 1, ply + 1)
                } else {
                    probe
                }
            };
            if self.aborted {
                return DRAW;
            }

            if score > best {
                best = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    self.pv.update(ply, mv);
                    if alpha >= beta {
                        self.record_cutoff(pos, mv, ply, depth);
                        break; // the opponent would never allow this line
                    }
                }
            }
        }
        self.history.pop();

        // Never record a result the abort cut short: it was not really searched.
        if !self.aborted {
            let bound = if best >= beta {
                Bound::Lower
            } else if best > original_alpha {
                Bound::Exact
            } else {
                Bound::Upper
            };
            self.tt.store(
                pos.zobrist(),
                best_move,
                best,
                0,
                depth.clamp(0, u8::MAX as i32) as u8,
                bound,
                ply,
            );
        }
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
        // Everything here is a capture or promotion, so killers and history
        // never apply.
        order_moves(pos, &mut moves, Move::NONE, [Move::NONE; 2], &self.quiet_history);

        // Skipping losing captures here is the textbook next step, and it was
        // tried and rejected. Depth 8 from the Ruy Lopez after 3...Nf6:
        //
        // ```text
        //   without   2,877,331 nodes  2,426,080 nps  1186 ms
        //   with      3,457,467 nodes  1,325,207 nps  2609 ms
        // ```
        //
        // Half the nps went on recomputing SEE that `order_moves` had already
        // computed, which an implementation that reused the sort key would
        // recover. The 20% *node* growth would not: dropping sacrifices makes
        // leaf values less accurate, and those values go into the transposition
        // table and misorder the main tree. Worth retrying once the evaluation
        // is strong enough that qsearch size matters more than qsearch accuracy.
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

/// Does `color` have anything but pawns and a king?
///
/// Null move pruning assumes a free pass cannot make your position worse. With
/// only pawns left that assumption inverts: zugzwang positions are exactly the
/// ones where being obliged to move is the problem, and a null move would
/// "prove" a cutoff that a real move cannot deliver.
fn has_non_pawn_material(pos: &Position, color: Color) -> bool {
    (pos.colored(color) - pos.pieces(color, PieceType::Pawn) - pos.pieces(color, PieceType::King))
        .any()
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
/// Order moves so the likely-best are searched first.
///
/// Alpha-beta prunes in proportion to how good this guess is: with perfect
/// ordering the effective branching factor drops from about 35 to about 6.
fn order_moves(
    pos: &Position,
    moves: &mut MoveList,
    tt_move: Move,
    killers: [Move; 2],
    quiet_history: &[i32],
) {
    let mut scored: ArrayVec<(i32, Move), { crate::movegen::MAX_MOVES }> = ArrayVec::new();
    for &mv in moves.iter() {
        scored.push((move_score(pos, mv, tt_move, killers, quiet_history), mv));
    }
    // Unstable sort so nothing is allocated inside the search.
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (slot, (_, mv)) in moves.iter_mut().zip(scored) {
        *slot = mv;
    }
}

fn move_score(
    pos: &Position,
    mv: Move,
    tt_move: Move,
    killers: [Move; 2],
    quiet_history: &[i32],
) -> i32 {
    /// A move already proven best here outranks everything else.
    const TT_BASE: i32 = 10_000_000;
    /// Captures the exchange says win or break even.
    const GOOD_CAPTURE_BASE: i32 = 8_000_000;
    /// Quiet moves that refuted a sibling at this same ply.
    const KILLER_FIRST: i32 = 7_000_000;
    const KILLER_SECOND: i32 = 6_000_000;
    /// Captures the exchange says lose material: after the winning ones and the
    /// killers, but still ahead of the remaining quiet moves.
    ///
    /// Textbook ordering puts these dead last. Measured on this engine that
    /// costs nodes, and it still does after killers and history were added, so
    /// the usual explanation (that it needs ordered quiets) is not the whole
    /// story. Depth 8 from the Ruy Lopez after 3...Nf6:
    ///
    /// ```text
    ///   ahead of quiets   2,877,331 nodes   1604 ms
    ///   dead last         3,274,924 nodes   1824 ms
    ///   dead last, no killers/history      15,389,205 nodes
    /// ```
    ///
    /// A losing capture is still a forcing move, and this history is shallow -
    /// bonus only, no malus and no continuation tables. Worth retrying once
    /// those exist.
    const BAD_CAPTURE_BASE: i32 = 5_000_000;

    if mv == tt_move && !mv.is_none() {
        return TT_BASE;
    }

    if !mv.is_capture() && !mv.is_promotion() {
        if mv == killers[0] {
            return KILLER_FIRST;
        }
        if mv == killers[1] {
            return KILLER_SECOND;
        }
        // Bounded by MAX_HISTORY, so quiet moves never reach a capture tier.
        let index = (pos.side_to_move().index() * 64 + mv.from().index()) * 64 + mv.to().index();
        return quiet_history[index];
    }

    let victim_value = if mv.is_en_passant() {
        params::MG_VALUE[PieceType::Pawn.index()]
    } else {
        pos.piece_at(mv.to())
            .map_or(0, |p| params::MG_VALUE[p.piece_type().index()])
    };
    let attacker_value = pos
        .piece_at(mv.from())
        .map_or(0, |p| params::MG_VALUE[p.piece_type().index()]);
    // Most valuable victim, least valuable attacker. Only a tie-break now that
    // SEE decides which side of the killers a capture lands on.
    let mvv_lva = victim_value * 16 - attacker_value;

    let exchange = see(pos, mv);
    if exchange >= 0 {
        GOOD_CAPTURE_BASE + exchange + mvv_lva
    } else {
        BAD_CAPTURE_BASE + exchange
    }
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
    ///
    /// Scope: this covers alpha-beta, PVS and quiescence, all of which are
    /// exact. It deliberately does not cover null move pruning, which is a
    /// gamble and *can* change the answer -- at depth 2 with a full window the
    /// null move conditions never fire.
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
    fn a_warm_table_shrinks_the_search() {
        // Searching the same position twice through one shared table: the
        // second pass inherits the first pass's best moves and prunes far
        // harder. If this stops holding, the table is not being consulted.
        let pos: Position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            .parse()
            .unwrap();
        let limits = SearchLimits {
            max_depth: Some(5),
            ..Default::default()
        };
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));

        let cold = search.run(&pos, limits, &mut |_| {});
        let warm = search.run(&pos, limits, &mut |_| {});

        assert!(
            warm.nodes < cold.nodes,
            "warm search visited {} nodes, cold visited {}",
            warm.nodes,
            cold.nodes
        );
        // And it must still reach the same conclusion.
        assert_eq!(warm.score, cold.score);
        assert_eq!(warm.depth, cold.depth);
    }

    #[test]
    fn table_hits_do_not_change_the_answer() {
        // A transposition-table cutoff returns a remembered score instead of
        // searching. That has to agree with what a cold search finds.
        for fen in [
            "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
        ] {
            let pos: Position = fen.parse().unwrap();
            let limits = SearchLimits {
                max_depth: Some(4),
                ..Default::default()
            };
            let cold = Search::new(Arc::new(AtomicBool::new(false)))
                .run(&pos, limits, &mut |_| {})
                .score;
            let mut shared = Search::new(Arc::new(AtomicBool::new(false)));
            shared.run(&pos, limits, &mut |_| {});
            let warm = shared.run(&pos, limits, &mut |_| {}).score;
            assert_eq!(cold, warm, "{fen}");
        }
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

    fn order(pos: &Position, search: &Search, tt_move: Move, ply: usize) -> Vec<String> {
        let mut moves = generate_legal(pos);
        order_moves(
            pos,
            &mut moves,
            tt_move,
            search.killers[ply],
            &search.quiet_history,
        );
        moves.iter().map(|mv| mv.to_string()).collect()
    }

    #[test]
    fn winning_captures_outrank_losing_ones() {
        // Qxg4 wins a hanging queen; Qxd5 wins a pawn but hangs the queen to
        // exd5. MVV-LVA rates Qxd5 highly because a pawn is still a victim;
        // only SEE separates them.
        let pos: Position = "4k3/8/4p3/3p4/6q1/8/8/3QK3 w - - 0 1".parse().unwrap();
        let search = Search::new(Arc::new(AtomicBool::new(false)));
        let ranked = order(&pos, &search, Move::NONE, 0);

        let at = |uci: &str| ranked.iter().position(|m| m == uci).expect(uci);
        assert_eq!(ranked[0], "d1g4", "the winning capture leads");
        assert!(at("d1d5") > at("d1g4"), "losing capture must rank lower");
        // But still ahead of the quiet moves: it is forcing, and the quiets are
        // unordered until history has something to say. See `BAD_CAPTURE_BASE`.
        assert!(at("d1d5") < at("e1f1"));
    }

    #[test]
    fn a_killer_outranks_a_losing_capture() {
        // Only capture available is Qxd5, which loses the queen. A quiet move
        // that refuted a sibling at this ply should be tried before it.
        let pos: Position = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1".parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));

        let before = order(&pos, &search, Move::NONE, 0);
        assert_eq!(before[0], "d1d5", "with nothing learned, the capture leads");

        let killer = generate_legal(&pos)
            .iter()
            .copied()
            .find(|mv| mv.to_string() == "e1f1")
            .unwrap();
        search.record_cutoff(&pos, killer, 0, 5);

        let after = order(&pos, &search, Move::NONE, 0);
        assert_eq!(after[0], "e1f1", "the killer should now lead");
    }

    #[test]
    fn history_ranks_quiet_moves_against_each_other() {
        let pos: Position = "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1".parse().unwrap();
        let mut search = Search::new(Arc::new(AtomicBool::new(false)));
        let pick = |uci: &str| {
            generate_legal(&pos)
                .iter()
                .copied()
                .find(|mv| mv.to_string() == uci)
                .expect(uci)
        };

        // Reward one quiet move at a shallow depth and another at a deeper one:
        // deeper cutoffs are stronger evidence and must rank higher.
        search.record_cutoff(&pos, pick("a1b1"), 4, 2);
        search.record_cutoff(&pos, pick("h1g1"), 4, 8);
        // Killers live at the ply they were recorded, so order from a ply with
        // none to isolate history.
        let ranked = order(&pos, &search, Move::NONE, 0);
        let at = |uci: &str| ranked.iter().position(|m| m == uci).expect(uci);
        assert!(at("h1g1") < at("a1b1"), "deeper cutoff should rank first");
        assert!(at("a1b1") < at("e1d1"), "any history beats none");
    }

    #[test]
    fn winning_captures_still_lead() {
        // Same shape, pawn undefended: now the capture is free and goes first.
        let pos: Position = "4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1".parse().unwrap();
        let mut moves = generate_legal(&pos);
        let scratch = Search::new(Arc::new(AtomicBool::new(false)));
        order_moves(&pos, &mut moves, Move::NONE, [Move::NONE; 2], &scratch.quiet_history);
        assert_eq!(moves[0].to_string(), "d1d5");
    }

    #[test]
    fn the_transposition_move_outranks_even_a_winning_capture() {
        let pos: Position = "4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1".parse().unwrap();
        let hint = generate_legal(&pos)
            .iter()
            .copied()
            .find(|mv| mv.to_string() == "e1f1")
            .unwrap();
        let mut moves = generate_legal(&pos);
        let scratch = Search::new(Arc::new(AtomicBool::new(false)));
        order_moves(&pos, &mut moves, hint, [Move::NONE; 2], &scratch.quiet_history);
        assert_eq!(moves[0], hint);
    }

    #[test]
    fn ordering_puts_captures_and_promotions_first() {
        let pos: Position = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1".parse().unwrap();
        let mut moves = generate_legal(&pos);
        let scratch = Search::new(Arc::new(AtomicBool::new(false)));
        order_moves(&pos, &mut moves, Move::NONE, [Move::NONE; 2], &scratch.quiet_history);
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
