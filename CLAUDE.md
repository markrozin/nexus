# newchessbot — conventions

A chess engine in Rust: minimax with alpha-beta pruning, quiescence search, and
an NNUE evaluation. This file is the source of truth for cross-cutting
conventions.

## Layout

The engine is a library (`src/lib.rs`); the binary is only the stdin loop.

| Module | Holds |
| --- | --- |
| `types` | `Square`, `Move`, `Color`, `PieceType`, `Piece`, `CastlingRights` |
| `bitboard` | `Bitboard`, jump-piece tables, slow reference sliders |
| `magic` | magic bitboards, the offline constant search |
| `board` | `Position` (bitboards + mailbox), FEN I/O, `make_move` |
| `movegen` | legal move generation, `perft` |
| `eval` | material, tapered piece-square tables |
| `search` | negamax, alpha-beta, quiescence, iterative deepening, time management |
| `rng` | xorshift64\* PRNG |
| `uci` | protocol handler, search worker thread |

Not yet written: transposition table, NNUE. `go` runs alpha-beta with
quiescence, to a time or depth limit.

## Correctness

Move generation is guarded by `perft` against the Chess Programming Wiki's
reference node counts (`movegen::tests`). Any change to `movegen`, `make_move`,
or the attack code must keep those passing — they are the regression net for
every optimization that follows (magic bitboards, pin-aware generation,
make/unmake).

`Position::assert_invariants` is the second net: it checks that the mailbox and
the bitboards agree. `movegen::tests::make_move_preserves_the_dual_representation`
walks the move tree calling it at every node.

Full-depth perft targets, all six verified:

| Position | Depth | Nodes |
| --- | --- | --- |
| startpos | 6 | 119,060,324 |
| Kiwipete | 5 | 193,690,690 |
| Position 3 | 7 | 178,633,661 |
| Position 4 | 6 | 706,045,033 |
| Position 5 | 5 | 89,941,194 |
| Position 6 | 5 | 164,075,551 |

All six pass at full depth as of magics landing: ~1.45 billion nodes in about 90
seconds, 10-23 Mnps. That is `movegen::tests::perft_full_depth`, kept behind
`#[ignore]` so the fast suite stays fast:

    cargo test --release -- --ignored --nocapture

The default suite covers all six at depth 3-4 and runs in well under a second.

## Roadmap

Strict dependency order. Nothing here can be skipped or reordered:

    bitboards -> attacks -> movegen -> perft -> UCI -> search -> eval -> NNUE

A movegen bug that survives into search presents as a search bug and costs days.
Perft is the gate.

| # | Milestone | Gate | Status |
| --- | --- | --- | --- |
| 1 | Bitboards, FEN, position | FEN round-trips, invariants hold | done |
| 2 | Magic bitboards | matches the ray-walking reference | done |
| 3 | Legal movegen, make/unmake | all six perft positions, full depth | done |
| 4 | UCI | loads and plays in a GUI | done (random mover) |
| 5 | Eval + negamax + alpha-beta + ID | beats a random mover 100/100 | done |
| 6 | Quiescence | SPRT pass | done (+292 Elo) |
| 7 | Zobrist + TT | SPRT pass | next |
| 8 | Move ordering + SEE | SPRT pass, node count drops sharply | |
| 9 | SPRT pipeline | gives a verdict on a known-good change | done |
| 10 | PVS, null move, LMR, futility | SPRT each independently | |
| 11 | Handcrafted eval | SPRT each term | |
| 12 | Datagen | 100M positions, FENs verify | |
| 13 | First net | clean loss curve | |
| 14 | NNUE inference | incremental == refresh, SPRT pass | |

## Board representation

- **Bitboards, little-endian rank-file (LERF) mapping.** `A1 = bit 0`,
  `B1 = bit 1`, ..., `H1 = bit 7`, `A2 = bit 8`, ..., `H8 = bit 63`.
- Squares are indexed `0..64`. `square = rank * 8 + file`, with `rank 0` the
  first rank and `file 0` the A-file.
- Shift directions follow from the mapping: north `<< 8`, south `>> 8`,
  east `<< 1`, west `>> 1` (with file masking to stop wraparound).
- **`Position` stores the board twice**: `[[Bitboard; 6]; 2]` indexed by color
  then piece type (plus cached per-color and total occupancy) *and* an
  `[Option<Piece>; 64]` mailbox. Bitboards answer "where are all the white
  rooks"; the mailbox answers "what is on e4" without scanning six boards.
- Every write to the board goes through `put`, `remove`, or `move_piece`. Do not
  touch the bitboards or the mailbox directly — keeping the writes in one place
  is what makes the invariant tractable.
- The `Shl`/`Shr` impls on `Bitboard` do **not** mask files. Use `east`, `west`,
  or `forward` where wraparound matters.
- `Piece` is a 12-variant `#[repr(u8)]` enum with discriminant
  `color * 6 + piece_type`, so it indexes a flat table directly.

## Scores

- Scores are `i32`, in **centipawns**, always from the **side-to-move's
  perspective** (negamax convention). Positive means the side to move is better.
- Mate scores use a large sentinel band near the `i32` range; keep a margin so
  `-score` and `alpha`/`beta` window arithmetic never overflow. Never use
  `i32::MIN` as a window bound (`-i32::MIN` overflows) — use a named
  `INFINITY` constant strictly inside the range.

## Threading

- Stdin is read on the main thread; `go` spawns a search worker with
  `std::thread::spawn`. The two share an `Arc<AtomicBool>` stop flag, polled
  with `Ordering::Relaxed` — the flag carries no data, so it only needs to
  become visible eventually, and `join` provides the real synchronization.
- `stop`, `ucinewgame`, and `quit` all join the worker before returning, so
  `bestmove` is always written before the next command is processed. Tests rely
  on this for deterministic transcripts.
- Protocol output goes through a `uci::Sink` (cloneable + `Send`), so the
  command loop and the worker write to the same place and tests can capture a
  whole session into a buffer.

## Search

- **No heap allocation inside search.** No `Vec`, `Box`, `String`, or
  collection growth on any path reachable from the main search or quiescence
  search. Move lists are `arrayvec::ArrayVec` sized to the legal maximum.
  Anything that must persist (transposition table, history tables) is allocated
  once at init and reused.
- Quiescence search extends the leaf with captures (and check evasions /
  promotions as decided later) to reach a quiet position before calling eval.

- **Never panic to abort a search.** `panic = "abort"` is in the release profile,
  so unwinding would kill the process rather than unwind the tree. Signal through
  the `AtomicBool`, return a sentinel, and discard the incomplete iteration.
- Poll the clock every 2048 nodes. Checking every node costs measurably.

## Types

- **Newtype wrappers, not bare integers.** `Square`, `Move`, `Bitboard` (and
  peers like `Piece`, `Color`, `File`, `Rank`) are `#[repr(transparent)]`
  structs over their integer, with named constructors and accessors. Do not pass
  a bare `u8`/`u16`/`u64` where one of these is meant.
- Prefer total constructors; where an invariant can't be checked cheaply, use a
  clearly named `from_u8_unchecked`-style constructor with a `// SAFETY:` /
  `// INVARIANT:` note at each call site.

## `unsafe`

- Every `unsafe` block carries a `// SAFETY:` comment immediately above it
  justifying why the operation is sound (bounds already checked, invariant held,
  etc.). No exceptions. A block with no justification is a bug.

## Inlining

- Put `#[inline]` on small, hot, cross-module functions (accessors, bit twiddlers,
  `Move` field extraction).
- Do **not** scatter `#[inline(always)]`. Reserve it for a measured win, with a
  comment noting the benchmark that justified it.

## Testing discipline

**No search or evaluation change merges without a passing SPRT.** From milestone
5 onward, intuition about our own changes stops being reliable: some heuristics
gain 20 Elo and some lose 5, and reading the code does not tell you which.

    fastchess -engine cmd=./target/release/newchessbot name=new \
              -engine cmd=./baseline name=base \
              -each tc=8+0.08 -rounds 50000 -concurrency 15 \
              -openings file=UHO_Lichess_4852_v1.epd format=epd order=random \
              -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05

Add heuristics one at a time, each behind its own run. SPRT stops as soon as the
evidence is conclusive, so bad changes are rejected in a few hundred games.

### Running it on this machine

`fastchess` came from `winget install Disservin.FastChess` and lives at
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Disservin.FastChess_*\fastchess-windows-x86-64\fastchess.exe`.
It needs **absolute** paths for `cmd=` and `file=`; relative ones fail with
"process creation failed".

The baseline to test against is `baseline/newchessbot-m5.exe` (gitignored;
rebuild from the commit named in `baseline/VERSION.txt`). Snapshot a new one
whenever a change passes.

Openings come from `books/random8.epd`, generated by
`cargo build --release --features bookgen --bin bookgen` then
`./target/release/bookgen 2000 > books/random8.epd`. The engine is
deterministic, so without a book every game is identical and a match carries no
information. The book is ours rather than a published one: the standard suites
are built to force decisiveness between engines strong enough to draw, which is
not this engine yet.

**Validate the harness before trusting a verdict.** Run the baseline against
itself first; it should land near zero. A 12-game control read -232 Elo and an
80-game control of the same two binaries read -4.3 +/- 49, which is the whole
lesson about small samples in one line.

Use `-concurrency 3` on this 8-core box: each game is two engine processes, and
oversubscription shows up as timing noise.
### Results so far

| Change | Result | Games | Elo |
| --- | --- | --- | --- |
| Quiescence search (milestone 6) | H1 accepted | 354 | +292.3 +/- 36.2 |

A note on pacing: with `elo0=0 elo1=5`, each game contributes a bounded amount
to the LLR, so a verdict costs a few hundred games no matter how large the true
effect is. A +292 Elo change and a +10 Elo change both take roughly the same
number of games to accept. Do not read a slow verdict as a weak result.



## NNUE

- Quantization constants are a contract with the trainer: `QA = 255`, `QB = 64`,
  `SCALE = 400`, and the hidden size must match the `bullet` schedule exactly. A
  mismatch shows up as evals that are systematically compressed or exploded, not
  as a training failure.
- Feature ordering is part of that contract too: accumulators are concatenated
  side-to-move first, and that is what tells the network whose turn it is.
- Incremental accumulator updates must be bit-identical to a full refresh, tested
  over thousands of random move sequences. Same class of bug as mailbox/bitboard
  drift, and it gets the same treatment.
- Every SIMD path keeps its scalar twin, plus a test asserting the two agree bit
  for bit.
- **Generate our own training data.** Training on networks or output produced by
  another engine raises derived-work questions and is barred by many rating
  lists.

## Profiles

- `cargo build --release` — `lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`, `opt-level = 3`. Use for playing strength and benchmarks.
- `cargo build --profile dev-fast` — `opt-level = 2` with debug assertions and
  overflow checks on. Use for day-to-day iteration.

## Features

- `datagen` — enables rayon for parallel self-play data generation. Off by
  default. Nothing behind this feature may be referenced from the search hot
  path.

## Dependencies

Keep them minimal. Current set: `arrayvec` (move lists), `rayon` (datagen only,
feature-gated), `criterion` (dev-only, benchmarks). No `serde`. No `rand` in the
hot path — if randomness is needed there, use a small explicit PRNG (e.g. xorshift)
written in-tree.

## Benchmarks

`benches/engine.rs` is a criterion target wired up but empty. Add perft, movegen,
eval, and fixed-depth search benchmarks as the corresponding code lands.
