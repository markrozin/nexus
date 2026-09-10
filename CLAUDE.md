# newchessbot — conventions

A chess engine in Rust: minimax with alpha-beta pruning, quiescence search, and
an NNUE evaluation. This file is the source of truth for cross-cutting
conventions.

## Layout

The engine is a library (`src/lib.rs`); the binary is only the stdin loop.

| Module | Holds |
| --- | --- |
| `types` | `Square`, `Move`, `Color`, `PieceType`, `Piece`, `CastleRights` |
| `bitboard` | `Bitboard`, jump-piece tables, sliding attacks |
| `board` | `Position`, FEN I/O, `make_move` |
| `movegen` | legal move generation, `perft` |
| `rng` | xorshift64\* PRNG |
| `uci` | protocol handler, search worker thread |

Not yet written: search, evaluation, NNUE, transposition table. `go` answers
with a random legal move.

## Correctness

Move generation is guarded by `perft` against the Chess Programming Wiki's
reference node counts (`movegen::tests`). Any change to `movegen`, `make_move`,
or the attack code must keep those passing — they are the regression net for
every optimization that follows (magic bitboards, pin-aware generation,
make/unmake).

## Board representation

- **Bitboards, little-endian rank-file (LERF) mapping.** `A1 = bit 0`,
  `B1 = bit 1`, ..., `H1 = bit 7`, `A2 = bit 8`, ..., `H8 = bit 63`.
- Squares are indexed `0..64`. `square = rank * 8 + file`, with `rank 0` the
  first rank and `file 0` the A-file.
- Shift directions follow from the mapping: north `<< 8`, south `>> 8`,
  east `<< 1`, west `>> 1` (with file masking to stop wraparound).

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
