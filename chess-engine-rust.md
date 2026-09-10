# Building a Chess Engine in Rust

A start-to-finish technical breakdown: board representation, magic bitboards, move generation, search, and NNUE training.

---

## Part 0 — Orientation

An engine is two halves bolted together.

**Search** explores move sequences and decides which position to trust. **Evaluation** scores a position as a number. Everything in this document is one or the other, plus the plumbing that lets them run fast.

The dependency order is strict and unforgiving:

```
bitboards → attacks → movegen → perft ✓ → UCI → search → eval → NNUE
```

You cannot skip forward. A movegen bug that survives to the search stage will look like a search bug, and you will lose days. Perft is the gate.

---

## Part 1 — Board representation

### 1.1 Bitboards

A bitboard is a `u64` where bit *n* means "something is true about square *n*." You keep one per piece type per colour. White knights on b1 and g1 is `0b01000010`, the low byte set at positions 1 and 6.

Fix the mapping now and never change it: **A1 = bit 0, B1 = bit 1, … H8 = bit 63.** Little-endian rank-file. It means `square = rank * 8 + file`, shifting left by 8 moves north, and most published code uses the same convention.

```rust
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Bitboard(0);

    #[inline]
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1u64 << sq.0)
    }
    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
    #[inline]
    pub fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }
    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;   // clears the lowest set bit
        sq
    }
}
```

Implement `Iterator` so you can write `for sq in knights { ... }`:

```rust
impl Iterator for Bitboard {
    type Item = Square;
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 { None } else { Some(self.pop_lsb()) }
    }
}
```

`count_ones` and `trailing_zeros` compile to single instructions (`popcnt`, `tzcnt`) with `target-cpu=native`. Don't hand-roll them.

Also implement `BitOr`, `BitAnd`, `BitXor`, `Not`, `Shl`, `Shr` and their assign variants so the newtype doesn't make the code ugly.

### 1.2 Squares, pieces, the position

```rust
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Square(pub u8);

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Color { White = 0, Black = 1 }

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType { Pawn, Knight, Bishop, Rook, Queen, King }
```

Keeping these `#[repr(u8)]` with values starting at zero lets you index arrays directly with `piece_type as usize`.

```rust
pub struct Position {
    pieces: [[Bitboard; 6]; 2],     // [color][piece_type]
    color_occ: [Bitboard; 2],
    occupied: Bitboard,
    mailbox: [Option<Piece>; 64],   // reverse lookup: what's on this square?
    stm: Color,
    castling: CastlingRights,       // u8 bitflags: WK, WQ, BK, BQ
    ep_square: Option<Square>,
    halfmove_clock: u8,
    fullmove: u16,
    zobrist: u64,
}
```

The redundancy between `pieces`, `color_occ`, `occupied` and `mailbox` is deliberate. Different operations want different views: movegen wants bitboards, "what did I just capture?" wants the mailbox. Keep them all in sync in `make_move` and add a debug-only `assert_invariants()` that verifies they agree — call it from every test.

### 1.3 FEN

Implement `FromStr` and `Display`. This is your only way to set up test positions, and every perft position is given as a FEN, so get it right early. Round-trip at least six FENs in a test, including positions with en passant squares and partial castling rights.

---

## Part 2 — Attack generation and magic bitboards

### 2.1 Leapers are easy

Knights, kings, and pawns don't care about blockers. Precompute a table of 64 bitboards each, generated at startup (or in a `const fn` if you want it at compile time):

```rust
static KNIGHT_ATTACKS: [Bitboard; 64] = /* generated */;
static KING_ATTACKS:   [Bitboard; 64] = /* generated */;
static PAWN_ATTACKS:   [[Bitboard; 64]; 2] = /* [color][square] */;
```

Generate them by shifting and masking off wraparound. A knight on h1 shifted "two right one up" must not appear on b3 — mask with `!FILE_A` and `!FILE_B` appropriately for each of the eight directions.

### 2.2 The sliding piece problem

A rook's attacks depend on where the blockers are. Naively you'd walk each ray square by square until you hit something — correct, but a loop with unpredictable branches in the hottest code in the engine.

You want a table lookup: `rook_attacks(square, occupancy)`. But occupancy is a `u64`, so a full table would need 2⁶⁴ entries per square. Obviously impossible.

Magic bitboards shrink that to something tiny by exploiting two facts.

### 2.3 Only some occupancy bits matter

For a rook on d4, the only squares whose occupancy can change the answer are the ones on the d-file and rank 4. Everything else is irrelevant.

Further: **edge squares don't matter either.** If there's a piece on d8, the rook attacks up to d8 and stops. If d8 is empty, the rook attacks up to d8 and stops (there's nothing beyond). Either way the answer on that ray is the same, so d8's occupancy is irrelevant.

So the *relevant occupancy mask* for a rook on d4 is: the d-file and rank 4, excluding d4 itself, excluding d1, d8, a4, h4.

```rust
static ROOK_MASK:   [Bitboard; 64] = /* generated */;
static BISHOP_MASK: [Bitboard; 64] = /* generated */;
```

Count the bits. A rook in the corner has 12 relevant squares. A rook in the centre has 10. Bishops range from 5 to 9. So the largest table any single square needs is 2¹² = 4096 entries — completely tractable.

### 2.4 The magic multiply

You now have a small number of relevant bits, but they're scattered across the 64-bit word. You need to compress them into a contiguous index.

```rust
let relevant = occupancy & ROOK_MASK[sq];
let index = (relevant.wrapping_mul(MAGIC[sq]) >> (64 - BITS[sq])) as usize;
let attacks = ROOK_TABLE[OFFSET[sq] + index];
```

That's the whole trick. The magic number is a carefully chosen 64-bit constant whose multiplication happens to scatter the relevant bits into the top `BITS[sq]` bits of the product without two different occupancies landing on the same index — or landing on the same index only when they produce the *same* attack set, which is harmless and actually desirable ("constructive collisions" let you use a smaller table).

There's no closed-form derivation. Multiplication by a random constant is a chaotic bit-mixing operation, and you find working magics by trying random numbers until one works.

### 2.5 Finding magics

Two pieces of machinery.

**Enumerating occupancy subsets.** You need every possible arrangement of blockers within a mask. The Carry-Rippler trick walks all 2ⁿ subsets of a bitboard in order:

```rust
let mut subset = 0u64;
loop {
    // use subset
    subset = subset.wrapping_sub(mask) & mask;
    if subset == 0 { break; }
}
```

**The search itself.**

```rust
fn find_magic(sq: Square, bits: u32, is_bishop: bool) -> u64 {
    let mask = if is_bishop { bishop_mask(sq) } else { rook_mask(sq) };
    let n = mask.count() as usize;

    // Precompute every (occupancy, correct_attacks) pair by slow ray-walking.
    let mut occupancies = vec![0u64; 1 << n];
    let mut attacks = vec![0u64; 1 << n];
    /* fill via Carry-Rippler + slow_attacks(sq, occ) */

    loop {
        // Magics with few set bits work better — AND three randoms together.
        let magic = rng.next() & rng.next() & rng.next();

        // Quick reject: the top byte of mask*magic should be well populated.
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 { continue; }

        let mut table = vec![0u64; 1 << bits];
        let mut used  = vec![false; 1 << bits];
        let mut ok = true;

        for i in 0..(1 << n) {
            let idx = (occupancies[i].wrapping_mul(magic) >> (64 - bits)) as usize;
            if !used[idx] {
                used[idx] = true;
                table[idx] = attacks[i];
            } else if table[idx] != attacks[i] {
                ok = false;   // destructive collision
                break;
            }
        }
        if ok { return magic; }
    }
}
```

Run this once, print the 128 constants, paste them into your source as a `static`. Keep the search code behind a feature flag — you'll never run it again, but you'll want it if you ever change the mask scheme.

**Ship the constants, don't search at startup.** Searching takes seconds; that's an unacceptable delay for an engine a GUI is waiting on.

### 2.6 The shared table

Rather than 64 separate tables, use one flat array with per-square offsets. Because collisions are allowed when attack sets match, a fitted table ("fancy magics") comes to roughly 100 KB for rooks and 5 KB for bishops — comfortably cache-friendly.

Bundle everything a square needs into one struct so a lookup touches one cache line:

```rust
#[derive(Copy, Clone)]
struct Magic {
    mask: u64,
    magic: u64,
    offset: u32,
    shift: u32,
}
```

### 2.7 Verify against a slow reference

Write `slow_attacks(sq, occ)` that walks rays one square at a time. It's obviously correct. Then test the magic version against it over ~10,000 random occupancy patterns per square, with a seeded PRNG so failures reproduce. This test is cheap insurance against a subtle mask bug that would otherwise surface as a mysterious perft mismatch.

### 2.8 The PEXT alternative

On x86 CPUs with BMI2, `_pext_u64` extracts masked bits into contiguous low bits in one instruction — exactly what the magic multiply is faking:

```rust
let index = unsafe { _pext_u64(occupancy, mask) } as usize;
```

No magic search, no collisions, smaller tables. The catch: PEXT is microcoded and extremely slow on AMD before Zen 3. Most engines ship both and select at runtime. Build magics first; add PEXT later as an optimization.

### 2.9 The universal entry point

```rust
pub fn attacks(pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard;
pub fn attackers_to(pos: &Position, sq: Square, occ: Bitboard) -> Bitboard;
```

`attackers_to` is the workhorse — used for check detection, legality, SEE, and king safety. It works by symmetry: a knight attacks `sq` if it sits on a square in `KNIGHT_ATTACKS[sq]`; a rook or queen attacks `sq` if it sits in `rook_attacks(sq, occ)`. Note the `occ` parameter is explicit, because several callers need to ask "what if this piece weren't there?"

---

## Part 3 — Move generation

### 3.1 Move encoding

Sixteen bits is enough and keeps move lists cache-dense:

```rust
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Move(u16);
// bits 0-5:   from square
// bits 6-11:  to square
// bits 12-15: flag
```

Sixteen flag values cover: quiet, double pawn push, king-side castle, queen-side castle, capture, en passant, four promotions, four capture-promotions. Encoding promotion piece in the flag means you never need a second field.

Use `ArrayVec<Move, 256>` for move lists — stack allocated, no heap traffic in the search. 256 is comfortably above the maximum legal move count in any real position (218).

### 3.2 Generating legal moves directly

You can generate pseudo-legal moves and filter by making each one and testing for check. It works and it's simpler. It's also roughly 2–3× slower, and you'll want to replace it eventually. Doing it right from the start is worth the extra thought.

The direct approach computes three masks before generating anything.

**Checkers.**

```rust
let checkers = attackers_to(pos, king_sq, occupied) & enemy;
```

- **Two or more checkers** → only king moves are legal. Generate those and stop. No piece can block or capture two checkers at once.
- **One checker** → every non-king move must either capture the checker or block the line:
  ```rust
  let check_mask = if checker_is_slider {
      between(king_sq, checker_sq) | checker_bb
  } else {
      checker_bb    // can't block a knight or pawn
  };
  ```
  `between(a, b)` is a precomputed `[[Bitboard; 64]; 64]` table of the squares strictly between two aligned squares (empty if not aligned).
- **No checkers** → `check_mask` is all ones.

**Pin masks.** For each enemy slider aligned with your king, look at the squares between them. If exactly one piece sits there and it's yours, it's pinned — it may only move along the king–slider ray.

```rust
let mut pinned = Bitboard::EMPTY;
let mut pin_ray = [Bitboard::EMPTY; 64];

let snipers = (rook_attacks(king_sq, Bitboard::EMPTY) & (enemy_rooks | enemy_queens))
            | (bishop_attacks(king_sq, Bitboard::EMPTY) & (enemy_bishops | enemy_queens));

for sniper_sq in snipers {
    let blockers = between(king_sq, sniper_sq) & occupied;
    if blockers.count() == 1 {
        let sq = blockers.lsb();
        if own.contains(sq) {
            pinned |= Bitboard::from_square(sq);
            pin_ray[sq.0 as usize] = between(king_sq, sniper_sq)
                                   | Bitboard::from_square(sniper_sq);
        }
    }
}
```

Note the `Bitboard::EMPTY` occupancy in the sniper computation — you want rays *through* pieces, not stopping at them.

Now a piece's legal destinations are `attacks & !own & check_mask & pin_mask_for_that_piece`, where the pin mask is all-ones for unpinned pieces.

**King danger squares.** The king's own rules are different — it can't move into check. Compute enemy attacks **with your king removed from the occupancy**:

```rust
let occ_without_king = occupied ^ Bitboard::from_square(king_sq);
```

This is the detail people get wrong. Without it, a king in check from a rook along the e-file looks like it can legally slide from e4 to e5 — still on the ray, still in check, but the rook's attack set appeared to stop at the king.

### 3.3 The en passant pin

There is one position type that defeats the pin logic above, and it is the single most common source of perft mismatches.

```
8/8/8/K2pP2r/8/8/8/7k w - d6
```

White king on a5, white pawn e5, black pawn d5 (just double-pushed), black rook h5. Capturing `exd6` removes **two** pawns from rank 5 simultaneously — the capturing pawn leaves e5 and the captured pawn leaves d5 — exposing the king to the rook. Neither pawn is pinned by the ordinary test, because each has the other as a second blocker.

Handle it as an explicit special case: for any en passant capture, construct the resulting occupancy with both pawns removed and verify the king isn't attacked along the rank.

```rust
let occ_after = (occupied ^ from_bb ^ captured_pawn_bb) | to_bb;
if (rook_attacks(king_sq, occ_after) & (enemy_rooks | enemy_queens)).is_empty() {
    // legal
}
```

En passant is rare enough that the cost of doing this the slow, obviously-correct way is irrelevant.

### 3.4 Make and unmake

Do **not** put the undo stack inside `Position`. In Rust that gives you a struct where every mutation needs `&mut self` on both the board and the history, and you'll end up cloning to escape the borrow checker — a quiet performance disaster.

```rust
#[derive(Copy, Clone)]
pub struct Undo {
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove_clock: u8,
    zobrist: u64,
}

impl Position {
    pub fn make_move(&mut self, mv: Move) -> Undo;
    pub fn unmake_move(&mut self, mv: Move, undo: Undo);
}
```

The `Undo` is `Copy` and lives on the search's own stack frame. Clean borrows, no allocation.

Things that must be handled and are easy to forget:

- Castling rights are lost when a **rook is captured on its home square**, not just when it moves.
- The en passant square is set **only** on a double pawn push, and should arguably only be set if an enemy pawn could actually capture (otherwise identical positions get different Zobrist keys and TT hits are missed).
- Promotion captures change two pieces at once.
- The halfmove clock resets on any pawn move or capture.

### 3.5 Perft — the gate

`perft(depth)` counts leaf nodes. `perft_divide` prints the count per root move, which is how you debug.

```rust
fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = pos.generate_legal();
    if depth == 1 { return moves.len() as u64; }   // bulk counting
    let mut nodes = 0;
    for mv in moves {
        let undo = pos.make_move(mv);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(mv, undo);
    }
    nodes
}
```

Test against the published values:

| Position | Depth | Nodes |
|---|---|---|
| startpos | 6 | 119,060,324 |
| Kiwipete | 5 | 193,690,690 |
| Position 3 | 7 | 178,633,661 |
| Position 4 | 6 | 706,045,033 |
| Position 5 | 5 | 89,941,194 |
| Position 6 | 5 | 164,075,551 |

Kiwipete (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -`) exercises castling, pins, and promotions all at once; it catches most bugs.

**Debugging a mismatch:** run `perft_divide` at the failing depth. Compare per-move counts against a known-good engine (`stockfish` has `go perft N`). Find the move whose subtree count differs, play it, recurse. In four or five steps you land on the exact position and depth-1 mismatch, which is always a specific move that shouldn't be generated or is missing.

Do not proceed past this section until all six positions match exactly.

---

## Part 4 — The tree, minimax, and search

### 4.1 The tree is implicit

You never build a tree in memory. The root is the current position, children are legal moves, and the "tree" exists only as the call stack: `make_move` before recursing, `unmake_move` after. A node is a stack frame.

What you *do* keep is a `SearchStack` — a fixed array indexed by ply, holding per-node data that outlives the recursion:

```rust
#[derive(Default, Clone, Copy)]
pub struct StackEntry {
    killers: [Option<Move>; 2],
    static_eval: i32,
    current_move: Option<Move>,
    in_check: bool,
}

pub struct Search {
    stack: [StackEntry; MAX_PLY],   // MAX_PLY = 128
    history: [[[i32; 64]; 64]; 2],  // [color][from][to]
    nodes: u64,
    stop: Arc<AtomicBool>,
    tt: Arc<TranspositionTable>,
}
```

Everything the search needs is preallocated. No `Vec`, no `Box`, no allocation of any kind inside the search.

### 4.2 Negamax

Textbook minimax has separate max and min cases. Nobody writes it that way, because `max(a, b) == -min(-a, -b)`. Define evaluation as *always from the perspective of the side to move*, and one function handles both players:

```rust
fn negamax(&mut self, pos: &mut Position, depth: i32, ply: usize) -> i32 {
    if depth == 0 { return evaluate(pos); }

    let mut best = -INFINITY;
    let moves = pos.generate_legal();

    if moves.is_empty() {
        return if pos.in_check() { -MATE + ply as i32 } else { 0 };
    }

    for mv in moves {
        let undo = pos.make_move(mv);
        let score = -self.negamax(pos, depth - 1, ply + 1);
        pos.unmake_move(mv, undo);
        best = best.max(score);
    }
    best
}
```

Two details in the terminal case. **No legal moves** means checkmate if in check, stalemate (score 0) otherwise. **Mate scores include the ply** — `-MATE + ply` — so that a mate in 3 scores higher than a mate in 5 and the engine actually finishes games instead of shuffling.

Use `MATE = 32000`, and treat anything above `MATE - MAX_PLY` as a mate score for TT and UCI purposes.

### 4.3 Alpha-beta

Carry a window. `alpha` is the best you've already secured elsewhere; `beta` is the best your opponent has already secured. If a score reaches beta, the opponent will never allow this line, so stop searching it.

```rust
fn negamax(&mut self, pos: &mut Position, mut alpha: i32, beta: i32,
           depth: i32, ply: usize) -> i32 {
    if depth == 0 { return self.quiescence(pos, alpha, beta, ply); }

    let mut best = -INFINITY;
    for mv in moves {
        let undo = pos.make_move(mv);
        let score = -self.negamax(pos, -beta, -alpha, depth - 1, ply + 1);
        pos.unmake_move(mv, undo);

        if score > best {
            best = score;
            if score > alpha {
                alpha = score;
                if alpha >= beta { break; }   // beta cutoff
            }
        }
    }
    best
}
```

Note the recursive call: `-beta, -alpha`, swapped and negated. Your alpha is the opponent's beta.

Use **fail-soft** (return `best`, which may lie outside the window) rather than fail-hard (clamp to the window). It gives the transposition table more precise bounds.

The payoff depends entirely on move ordering. With perfect ordering, the effective branching factor drops from ~35 to ~6, roughly doubling reachable depth. With random ordering you gain almost nothing. This is why so much of the remaining work is about guessing the best move before searching it.

### 4.4 Iterative deepening and time management

Search depth 1, then 2, then 3, until time runs out.

This sounds wasteful — but the tree grows exponentially, so all previous iterations together cost less than the final one. And the shallow searches fill the transposition table with best-move hints that make the deep search's ordering far better. Iterative deepening is a net *speedup*, not a cost.

```rust
pub fn go(&mut self, pos: &mut Position, limits: Limits) -> Move {
    let mut best_move = None;
    let mut prev_score = 0;

    for depth in 1..=limits.max_depth {
        let score = self.aspiration_search(pos, depth, prev_score);
        if self.stopped() { break; }         // discard partial iteration

        best_move = Some(self.pv[0]);
        prev_score = score;
        self.print_info(depth, score);

        if self.time_up_for_next_iteration() { break; }
    }
    best_move.unwrap()
}
```

**Time allocation:** roughly `remaining / 20 + increment * 3 / 4` per move, with a hard cap at some fraction of remaining time. Check the clock every 2048 nodes — checking every node costs measurably.

**Aborting:** set an atomic flag, unwind by returning a sentinel, and *discard the incomplete iteration*. Never panic to unwind; `panic = "abort"` is in your release profile and it would kill the process.

**Aspiration windows:** after depth 4 or so, start each iteration with a narrow window around the previous score instead of `(-INF, INF)`:

```rust
let mut delta = 25;
let mut alpha = prev - delta;
let mut beta  = prev + delta;
loop {
    let score = self.negamax(pos, alpha, beta, depth, 0);
    if score <= alpha      { beta = (alpha + beta) / 2; alpha -= delta; }
    else if score >= beta  { beta += delta; }
    else                   { return score; }
    delta += delta / 2;
}
```

Narrow windows prune much harder. The occasional re-search costs less than the savings.

### 4.5 Quiescence search

The horizon effect: if your search ends the instant after you capture a defended pawn with your queen, you record a pawn's profit and never see the recapture. The engine will confidently hang material.

The fix is to keep searching *captures* at the leaves until the position is quiet.

```rust
fn quiescence(&mut self, pos: &mut Position, mut alpha: i32,
              beta: i32, ply: usize) -> i32 {
    let stand_pat = evaluate(pos);
    if stand_pat >= beta { return stand_pat; }
    if stand_pat > alpha { alpha = stand_pat; }

    for mv in pos.generate_captures_and_promotions() {
        // delta pruning
        if stand_pat + see_value(mv) + 200 < alpha && !endgame { continue; }
        if see(pos, mv) < 0 { continue; }   // skip losing captures

        let undo = pos.make_move(mv);
        let score = -self.quiescence(pos, -beta, -alpha, ply + 1);
        pos.unmake_move(mv, undo);

        if score >= beta { return score; }
        if score > alpha { alpha = score; }
    }
    alpha
}
```

The **stand-pat** is the key idea: you're not forced to capture, so the static eval acts as a lower bound. There's no depth limit — the capture sequence terminates on its own — but cap recursion at `MAX_PLY` as a safety net.

This is one of the largest single strength gains in the whole project.

### 4.6 Zobrist hashing and the transposition table

**Zobrist:** a random 64-bit key per (piece, square), plus one for side-to-move, four for castling rights, eight for en passant file. The position's key is the XOR of all active ones. Because XOR is its own inverse, you update incrementally: moving a knight from b1 to c3 is `key ^= ZOBRIST[N][b1] ^ ZOBRIST[N][c3]`.

Generate the table with a `const fn` xorshift so it's baked into the binary. Add a `debug_assert!` in `make_move` comparing the incremental key against a from-scratch recomputation — an incremental-update bug produces corrupted TT hits and near-untraceable misbehaviour.

**The table.** Sixteen bytes per entry, four entries per cache line:

```rust
#[derive(Copy, Clone, Default)]
struct Entry {
    key: u16,        // upper bits of the Zobrist key, for verification
    mv: u16,
    score: i16,
    eval: i16,
    depth: u8,
    flags: u8,       // bound type (2 bits) + age (6 bits)
}
```

In Rust, this is the one place `unsafe` is genuinely warranted. The table is shared across search threads with deliberately racy access — `Mutex` or per-entry atomics cost far more than the occasional corrupted entry:

```rust
pub struct TranspositionTable {
    entries: Box<[UnsafeCell<Entry>]>,
}
unsafe impl Sync for TranspositionTable {}
```

Guard against torn writes by XOR-ing the key with the packed data on write and un-XOR-ing on read; a torn entry then fails the key check and is treated as a miss. Write a real `// SAFETY:` comment explaining the reasoning and the failure mode.

**Using it.** Probe on entry: if the stored depth is at least your remaining depth *and* the bound type permits, return the score. Regardless of depth, extract the stored move for ordering — that alone is worth a lot.

Bound types matter:
- `Exact` — the search completed within the window, score is exact.
- `LowerBound` — a beta cutoff, true score is at least this.
- `UpperBound` — nothing beat alpha, true score is at most this.

Only cut off when the bound is on the right side of your current window.

**Mate scores must be adjusted** relative to the current ply on both store and probe, since "mate in 3 from here" means something different at a different depth.

Also implement **threefold repetition** and the **fifty-move rule** from a position-history `Vec<u64>` of Zobrist keys. Missing these means the engine throws away won games.

### 4.7 Move ordering

The single highest-leverage component after alpha-beta itself. Order:

1. **TT move** — the best move from a previous search of this position.
2. **Winning and equal captures** — scored by static exchange evaluation (SEE), tie-broken by MVV-LVA (most valuable victim, least valuable attacker).
3. **Killer moves** — two quiet moves per ply that caused a cutoff elsewhere at the same depth.
4. **Quiet moves by history** — a `[color][from][to]` table incremented on cutoffs.
5. **Losing captures** — last.

**SEE** answers "if both sides keep recapturing on this square, do I come out ahead?" Implement it as a swap list: repeatedly take the least valuable attacker of each side in turn, building an array of running material scores, then propagate backwards with a max/min since either side can stop.

**History with gravity** keeps values bounded without periodic rescaling:

```rust
fn update_history(&mut self, c: Color, mv: Move, bonus: i32) {
    let entry = &mut self.history[c as usize][mv.from()][mv.to()];
    let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
    *entry += clamped - *entry * clamped.abs() / MAX_HISTORY;
}
```

**Generate lazily.** Don't build and score the whole list up front — most nodes cut off after one or two moves. Use a stage machine:

```rust
enum Stage { TTMove, GenCaptures, GoodCaptures, Killers, GenQuiets, Quiets, BadCaptures }

pub struct MovePicker { stage: Stage, /* ... */ }

impl MovePicker {
    pub fn next(&mut self, pos: &Position) -> Option<Move> { /* ... */ }
}
```

Deliberately *not* an `Iterator` impl — it needs `&Position` and mutable scoring state, and fighting the trait signature isn't worth it.

### 4.8 The heuristic layer

Everything above is exact: alpha-beta returns the same answer minimax would. Everything below is a gamble that trades occasional correctness for depth. Each one is worth Elo, and each one must be measured, not assumed.

**Principal variation search.** After the first move at a node, search the rest with a null window `(alpha, alpha+1)` — very cheap, and it only proves "this isn't better than alpha." Re-search properly on a fail high.

**Null move pruning.** Give the opponent a free move. If you're *still* above beta at reduced depth, the position is so good you can prune. Requires: not in check, non-PV node, depth ≥ 3, side to move has non-pawn material (otherwise zugzwang breaks the assumption), and no null move already on this branch. Reduction `R = 3 + depth / 6`.

**Late move reductions.** Moves late in your ordering are probably bad, so search them shallower. Reduction from a `log(depth) * log(move_index)` table computed at startup, reduced further for killers, checks, and high-history moves. Re-search at full depth if a reduced search beats alpha. One of the biggest single gains in modern engines.

**Reverse futility pruning.** At shallow depth, non-PV, not in check: if `static_eval - margin * depth >= beta`, just return. You're so far ahead that a shallow search is unlikely to change it.

**Futility pruning.** Near the leaves: if `static_eval + margin < alpha`, skip quiet moves. They can't rescue the position.

**Late move pruning.** At shallow depth, stop searching quiet moves entirely after a depth-dependent count.

**Check extensions.** The opposite: search *deeper* when in check, since forcing lines resolve quickly and misjudging them is expensive.

Add these **one at a time**, each behind its own SPRT run. Some will gain 20 Elo, some will lose 5, and you cannot predict which from reading the code.

### 4.9 Testing discipline

Before adding the heuristic layer, build the measurement pipeline. From here on your intuition about your own changes is unreliable.

```bash
fastchess -engine cmd=./target/release/engine_new name=new \
          -engine cmd=./baseline name=base \
          -each tc=8+0.08 -rounds 50000 -concurrency 15 \
          -openings file=UHO_Lichess_4852_v1.epd format=epd order=random \
          -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05
```

SPRT evaluates after every game and stops as soon as the evidence is conclusive either way — LLR above +2.94 accepts, below −2.94 rejects. Bad changes get rejected in a few hundred games; genuinely marginal ones take thousands, but you only pay when the answer is actually close.

Write it into `CLAUDE.md`: no search or eval change merges without a passing SPRT.

---

## Part 5 — Handcrafted evaluation

You need this before NNUE, for two reasons: it's what your engine plays with while you build everything else, and it's what generates your training data.

**Material and piece-square tables**, with separate midgame and endgame values interpolated by a phase factor:

```rust
let phase = (knights + bishops) * 1 + rooks * 2 + queens * 4;   // 0..24
let score = (mg * phase + eg * (24 - phase)) / 24;
```

Then, one SPRT-tested term at a time: passed pawns scaled by rank, isolated/doubled/backward pawns, rooks on open files, bishop pair, mobility over squares not attacked by enemy pawns, and king safety via attack units on the king zone.

Keep every constant in a single `params` module so a Texel tuner can mutate them later.

Since NNUE is the destination, don't over-invest here. This eval only needs to be good enough to produce sane training labels — roughly 2200–2500 is plenty.

---

## Part 6 — NNUE: architecture and inference

NNUE ("Efficiently Updatable Neural Network" — the acronym is backwards because it came from Japanese shogi engines) replaces your handcrafted evaluation with a small neural network designed to run on a CPU inside a search doing millions of nodes per second.

Three properties make that possible.

### 6.1 Why the architecture looks like this

**It's shallow and front-loaded.** Start with `768 → 512×2 → 1`: a huge sparse input layer, one hidden layer per perspective, and a single output neuron. Roughly 99% of the parameters live in the first layer. There is no deep stack — depth would cost more than it gains at this node rate.

**It's incrementally updatable.** This is the entire trick, and it's why the architecture is shaped this way. The first layer's output is just a *sum of weight columns*, one per active input feature. When you make a move, only two or three features change. So instead of a 768×512 matrix multiply, you subtract one column and add another — a few hundred additions instead of nearly 400,000 multiply-accumulates.

**It's integer-quantized.** Weights become `i16`, activations clamp to a known range, and AVX2 processes 16 lanes at a time. No floats, no GPU, no allocation.

### 6.2 Feature encoding

768 = 2 colours × 6 piece types × 64 squares. A feature is on when that piece is on that square.

Crucially there are **two perspectives**. The network sees the position twice: once from White's point of view and once from Black's. For the non-white perspective you flip the board vertically and swap piece colours, so "my pawn on my second rank" always maps to the same feature index regardless of which side you are.

```rust
const fn feature_index(persp: Color, pc: Color, pt: PieceType, sq: Square) -> usize {
    let (color_idx, sq_idx) = if persp == Color::White {
        (pc as usize, sq.0 as usize)
    } else {
        (1 - pc as usize, (sq.0 ^ 56) as usize)   // ^56 mirrors the rank
    };
    color_idx * 384 + (pt as usize) * 64 + sq_idx
}
```

At inference you concatenate the two accumulators **side-to-move first**. That ordering is what tells the network whose turn it is, and it must match exactly what the trainer used.

Later you'll move to **HalfKA/HalfKP** features — `(own king square, piece, square)` tuples, so the network learns piece values conditioned on king position. That's ~45,000 inputs and much stronger, at the cost of needing a full accumulator refresh whenever the king moves. Get 768 working first.

### 6.3 The accumulator

```rust
const HIDDEN: usize = 512;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    values: [[i16; HIDDEN]; 2],   // [perspective][neuron]
}
```

A **full refresh** starts from the layer bias and adds one weight column per piece on the board:

```rust
impl Accumulator {
    fn refresh(&mut self, pos: &Position, net: &Network) {
        for p in 0..2 {
            self.values[p] = net.feature_bias;
        }
        for sq in pos.occupied() {
            let piece = pos.piece_at(sq).unwrap();
            self.add_feature(piece, sq, net);
        }
    }

    #[inline]
    fn add_feature(&mut self, piece: Piece, sq: Square, net: &Network) {
        for p in [Color::White, Color::Black] {
            let idx = feature_index(p, piece.color, piece.pt, sq);
            let col = &net.feature_weights[idx];          // [i16; HIDDEN]
            for i in 0..HIDDEN {
                self.values[p as usize][i] += col[i];
            }
        }
    }
}
```

**Incremental update** is where the speed comes from. Keep a stack of accumulators indexed by ply — one per search frame — so `unmake` is a pop, not a recompute:

| Move type | Operations |
|---|---|
| Quiet | `sub(piece, from)`, `add(piece, to)` |
| Capture | `sub(piece, from)`, `sub(captured, to)`, `add(piece, to)` |
| Promotion | `sub(pawn, from)`, `add(promoted, to)` |
| Castling | two subs, two adds (king and rook) |
| En passant | `sub(pawn, from)`, `sub(enemy_pawn, ep_capture_sq)`, `add(pawn, to)` |

Fuse these into `sub_add`, `sub_sub_add`, and `sub_add_sub_add` functions so each does one pass over the array rather than two or four.

**The correctness test that matters:** run 10,000 random legal move sequences, and after every single `make_move` assert that the incrementally-updated accumulator is bit-identical to a full refresh. An incremental bug produces an engine that plays *almost* correctly and loses 300 Elo for no visible reason.

### 6.4 SCReLU and quantization

The activation is **squared clipped ReLU**: clamp to `[0, QA]`, then square. The squaring gives useful nonlinearity for free, and the clamp keeps values in a range where integer arithmetic can't overflow.

Standard constants:

```rust
const QA: i32 = 255;      // accumulator / feature weight scale
const QB: i32 = 64;       // output weight scale
const SCALE: i32 = 400;   // network output units → centipawns
```

Forward pass:

```rust
#[inline]
fn screlu(x: i16) -> i32 {
    let v = (x as i32).clamp(0, QA);
    v * v
}

pub fn evaluate(acc: &Accumulator, stm: Color, net: &Network) -> i32 {
    let (us, them) = (stm as usize, 1 - stm as usize);
    let mut sum: i32 = 0;

    for i in 0..HIDDEN {
        sum += screlu(acc.values[us][i])   * net.output_weights[i]          as i32;
        sum += screlu(acc.values[them][i]) * net.output_weights[i + HIDDEN] as i32;
    }

    // Divide by QA once to undo the squaring, then rescale to centipawns.
    (sum / QA + net.output_bias as i32) * SCALE / (QA * QB)
}
```

`bullet` handles quantization at export, so these constants must match your trainer config exactly. If your evals come out 100× too large or too small, this division chain is where to look.

**Loading the net:** embed it with `include_bytes!` and parse into a `#[repr(C, align(64))]` static so the weights are cache-line aligned for SIMD.

**SIMD:** write a scalar version first and keep it. Add an AVX2 path behind `#[target_feature(enable = "avx2")]` using `core::arch::x86_64`, selected at startup with `is_x86_feature_detected!`. Then add a test asserting both paths produce **bit-identical** output on a few thousand positions — a SIMD path that's subtly different from the scalar one is a nightmare to diagnose later.

Expect roughly 4–8× on the accumulator update, which is the hot loop.

---

## Part 7 — NNUE: data and training

This is the part people underestimate. The network is easy; the data pipeline is the project.

### 7.1 What a labelled position is

Three things:

1. **The position** — piece placement and side to move.
2. **A search score** — what your own engine thinks the position is worth, in centipawns from a consistent perspective.
3. **The game result** — 1.0 / 0.5 / 0.0, from that same perspective, from the self-play game this position came from.

The training target blends the two labels:

```
target = λ · sigmoid(score / 400) + (1 − λ) · result
```

with λ around 0.7. Both signals are flawed on their own. The search score is precise but inherits your engine's biases — it can only teach the network what your engine already knows. The game result is unbiased ground truth but extremely noisy, since one blunder forty moves later flips the label on a position that was genuinely fine. Blending gets precision from one and grounding from the other.

Note this is a bootstrap: **your engine labels its own training data.** The network learns to approximate a depth-8 search with a single forward pass, then a search using that network is stronger than the original, so the next round of data is better. That loop is where most of the strength comes from.

### 7.2 Data generation

Add a `datagen` subcommand to the engine.

```
for each game:
    start from the initial position
    play 8-10 random legal plies
    if |eval| > 200cp after the random opening: discard, start over
    loop:
        search to a fixed NODE COUNT (~5000 nodes)
        if position is quiet: record (position, score, ply)
        play the best move
        until game over (mate, stalemate, 50-move, repetition, or adjudication)
    write all recorded positions with the final result attached
```

The details that matter:

- **Fixed nodes, not fixed depth.** Node counts are hardware-independent, so data generated on your laptop and your desktop is consistent, and a mid-project speed optimization doesn't silently change your data distribution.
- **Randomized openings.** Without them every game starts from the same position and your data has almost no variety. Random plies work; an unbalanced opening book like `UHO_Lichess_4852` works better.
- **Reject lopsided openings.** Random plies sometimes produce a position already lost by a queen. Those games teach nothing.
- **Only quiet positions.** Skip anything where the side to move is in check or the best move is a capture. The network has no search — it cannot see a hanging queen — so training it on tactical positions teaches it noise. This is the same reason quiescence search exists.
- **Skip mate scores.** `sigmoid(31995 / 400)` saturates and contributes no gradient.
- **Skip the first few plies.** Opening positions are over-represented and nearly identical.

Parallelize with `rayon`: one `Position` and one small TT per thread, per-thread output buffers flushed to a shared file behind a `Mutex`. On a modern desktop expect a few million positions per hour per core.

**Volume:** 100 million positions is the minimum for a usable `768 → 512×2 → 1` net. Serious engines train HalfKA nets on billions. Start at 100M, get the pipeline working end to end, then scale.

**A note on data provenance:** generate your own. Training on Stockfish's networks or Stockfish-generated data raises derived-work questions, and most engine tournaments and rating lists have rules about it. Self-generated data also means the whole loop is genuinely yours.

### 7.3 The data format

Use the `bulletformat` crate rather than inventing a format. Its `ChessBoard` type is a packed 32-byte record — an occupancy bitboard plus nibble-packed piece identities, score, and result — and `bullet` reads it directly with no conversion step.

```rust
use bulletformat::ChessBoard;

let record = ChessBoard::from_raw(bitboards, stm, score, result)?;
writer.write_all(bytemuck::bytes_of(&record))?;
```

Check the crate docs for which perspective `score` and `result` are expected in, and write a sanity check that reads your file back and reconstructs a few positions as FENs you can eyeball. A perspective sign error produces a network that plays like it's trying to lose, and it is not obvious from the loss curve.

### 7.4 Training with bullet

Add [`bullet`](https://github.com/jw1912/bullet) as a second crate in your workspace. A minimal trainer:

```rust
use bullet_lib::{
    inputs, outputs, Activation, LocalSettings, Loss,
    TrainerBuilder, TrainingSchedule, WdlScheduler, LrScheduler,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])          // must match QA, QB in the engine
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(512)           // HIDDEN
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net001".to_string(),
        eval_scale: 400.0,                  // must match SCALE
        batch_size: 16_384,
        start_superbatch: 1,
        end_superbatch: 240,
        wdl_scheduler: WdlScheduler::Constant { value: 0.3 },   // 1 − λ
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.3, step: 60 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 20,
        ..Default::default()
    };

    trainer.run(&schedule, &LocalSettings {
        threads: 8,
        data_file_paths: vec!["data/selfplay.bin"],
        output_directory: "nets",
        ..Default::default()
    });
}
```

What each knob does:

- **`eval_scale: 400`** — the sigmoid divisor. Must be the same 400 your engine divides by. A mismatch produces a network whose evals are systematically compressed or exploded.
- **`quantisations: [255, 64]`** — QA and QB. Must match the engine's constants exactly.
- **`wdl_scheduler`** — how much weight the game result gets versus the search score. Constant 0.2–0.3 is a good default. Some setups ramp it up over training, moving from imitating the search early to trusting real outcomes late.
- **`batch_size: 16384`** — standard. bullet groups batches into "superbatches" of about 100 million positions.
- **`lr_scheduler`** — start at 0.001, drop by 0.3× every 60 superbatches. The drops matter; a flat learning rate plateaus early.
- **`end_superbatch: 240`** — with 100M positions this is roughly 240 passes over the data. Fewer for a first run while you're debugging.

Training a 768-input net on 100M positions takes a few hours on a CPU with enough threads, or well under an hour on a GPU.

### 7.5 Reading the loss curve

A healthy run: fast drop over the first few superbatches, then a long slow decline, with visible step-downs at each learning-rate drop.

| Symptom | Likely cause |
|---|---|
| Loss flat from the start | Learning rate far too low, or the data file isn't being read |
| Loss explodes to NaN | Learning rate too high |
| Loss drops then plateaus immediately | Not enough data — the net memorized it |
| Loss looks fine, engine plays badly | Perspective bug in datagen, or a quantization mismatch at inference |

That last row is the common one, and it's why the incremental-vs-refresh test and the FEN-reconstruction sanity check exist. A network that trains beautifully on mislabelled data will produce a beautiful loss curve and a terrible engine.

Absolute loss values aren't meaningful across configurations — only compare runs with identical `eval_scale` and WDL settings.

### 7.6 Integration and the iteration loop

Replace `evaluate()` with the NNUE forward pass, keeping the handcrafted eval behind a compile-time flag so you can A/B them. Then SPRT.

A first 768-net on 100M self-generated positions typically gains **150–300 Elo** over a decent handcrafted eval. If you gain nothing, or lose, the bug is almost always in the data pipeline rather than the network.

Then the loop runs:

```
generate data with current engine
    → train a net
        → SPRT vs current
            → if better, ship it and regenerate data with the stronger engine
```

Each cycle compounds, because better data comes from a better labeller. Three or four cycles at 768 before changing architecture is reasonable.

**After that**, in rough order of value:
1. More data — 100M → 500M → 1B.
2. Larger accumulator — 512 → 1024 → 1536.
3. HalfKA features with king buckets. The big one, and the point where a full refresh on king moves starts to matter.
4. Output buckets by material count, so endgames and middlegames get separate output layers.

---

## Part 8 — Milestones

| # | Milestone | Gate | Cumulative |
|---|---|---|---|
| 1 | Bitboards, FEN, position | Round-trip tests pass | — |
| 2 | Magic bitboards | Matches slow reference | — |
| 3 | Legal movegen, make/unmake | **All six perft positions exact** | — |
| 4 | UCI protocol | Loads and plays in Cute Chess | ~random |
| 5 | Eval + negamax + alpha-beta + ID | Beats a random mover 100/100 | ~1700 |
| 6 | Quiescence search | SPRT pass | ~2000 |
| 7 | Zobrist + TT | SPRT pass | ~2150 |
| 8 | Move ordering + SEE | SPRT pass, node count drops sharply | ~2300 |
| 9 | SPRT pipeline | Produces a verdict on a known-good change | — |
| 10 | PVS, null move, LMR, futility | SPRT each independently | ~2600 |
| 11 | Full handcrafted eval | SPRT each term | ~2750 |
| 12 | Datagen | 100M positions, FENs verify | — |
| 13 | Train first net | Clean loss curve | — |
| 14 | NNUE inference | Incremental == refresh, SPRT pass | ~3000 |

Elo figures are self-play estimates and will vary. Treat them as shape, not promise.

**Timing:** milestones 1–8 are a few focused weekends if you're comfortable in Rust; expect longer if bitboards are new. Milestone 3 alone can eat a weekend on a single perft mismatch. Milestones 12–14 are their own project again.

---

## Part 9 — Resources

**Reference**
- [Chess Programming Wiki](https://www.chessprogramming.org) — the reference for every term in this document
- [Perft results](https://www.chessprogramming.org/Perft_Results) — the six standard positions
- UCI specification (Stefan Meyer-Kahlen) — ~400 lines, worth reading once in full

**Rust engines worth reading**
- Viridithas — strong, well-organized, actively developed
- Akimbo — deliberately compact, good for seeing the minimum viable shape of each component
- Carp and Svart — readable, well-commented

**Tools**
- `bullet` — the NNUE trainer, and `bulletformat` for the data format
- `fastchess` — SPRT testing (successor to cutechess-cli)
- Cute Chess — GUI for watching games and manual testing
- `cozy-chess` — a well-tested Rust movegen crate; don't depend on it, but a good correctness reference

**Community**
- The Engine Programming Discord is where most current work happens and where you can get help with perft mismatches and SPRT setup. Higher signal than any forum.
