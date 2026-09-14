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
| `zobrist` | position hashing, const-built key tables |
| `tt` | transposition table, lock-free via relaxed atomics |
| `see` | static exchange evaluation |
| `uci` | protocol handler, search worker thread |
| `nnue` | network loader, accumulators, inference; the embedded net |

`go` runs alpha-beta with quiescence, a transposition table, PVS, null move
pruning, late move reductions and reverse futility pruning, to a time or depth
limit, evaluating leaves with the handcrafted evaluation by default.

NNUE inference is wired in but opt-in until a network passes an SPRT against
the handcrafted evaluation: `setoption name EvalFile value <embedded>` uses the
net compiled in from `networks/default.bin`, a path loads any other net without
rebuilding (which is how candidate nets get tested), and `<handcrafted>` is the
default. Flip the default in `Search::with_table` and `Uci::new` only on a
passing SPRT.

The handcrafted `eval` stays. Datagen and bookgen pin it explicitly
(`Evaluator::Handcrafted`): datagen's quiet filter and `datacheck` compare
against it, and moving datagen to the network means changing those together.

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
| 7 | Zobrist + TT | SPRT pass | done (+28, +121 Elo) |
| 8 | Move ordering + SEE | SPRT pass, node count drops sharply | done (+60 Elo, -43% nodes) |
| 9 | SPRT pipeline | gives a verdict on a known-good change | done |
| 10 | PVS, null move, LMR, futility | SPRT each independently | done (+166, +87, +86; LMP rejected) |
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

### The heuristic layer

Reference values, researched rather than guessed. Each still needs its own SPRT
here; these are starting points, not verdicts.

- **LMR** reduces by `c + ln(depth) * ln(move_index) / d`, not a bare log
  product. Ethereal uses `0.7844 + ln*ln/2.4696` for quiets, Obsidian
  `0.99 + ln*ln/3.14`, Weiss separate constants for captures and quiets.
  Applies from depth 3 and after the first few moves; re-search at full depth
  when the reduced search beats alpha.
- **Null move** reduction R of 3 to 4, optionally scaled by `depth/3`. Skip in
  check, in PV nodes, with a null already on the branch, and with no non-pawn
  material. A further guard worth testing: require static eval above beta.
- **Reverse futility** returns early when `eval >= beta + margin * depth`, with
  margin around 150. Skip in check and in PV nodes.
- **Futility** applies at frontier nodes only; captures and checks are exempt.
  The deep and extended variants are historical, and modern engines prefer
  move-count pruning instead.
- **Singular extensions** are worth roughly 10 to 36 Elo depending on engine,
  well below their original billing.

Move ordering note: searching SEE-losing captures *before* quiet moves rather
than last is an established split, not an anomaly, and it is what measured
better here. See `BAD_CAPTURE_BASE`.

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

### Measure node counts before spending an SPRT

A fixed-depth `go depth N` on one position takes about two minutes and reports
nodes, nps and time. An SPRT takes twenty to forty-five. For anything that
changes move ordering or pruning, check the node count first — it caught an
ordering change that was 3x *worse* than the baseline, which would otherwise
have cost most of an hour to learn.

Compare wall-clock, not just nodes: SEE ordering cut nodes 43% but also cost 13%
of nps, and only the product matters. Node count is a proxy; the SPRT is still
the gate.

And node counts flatter pruning heuristics specifically. LMR cut nodes ~14x
against null move's ~6-10x, yet measured +87 Elo against null move's +166: some
of a reduction's saving comes from searching less accurately, not just more
cheaply. Treat a large node drop from a *pruning* change as weaker evidence
than the same drop from an *ordering* change.

But do not turn that into a quantitative prediction. Reverse futility cut nodes
only 15-39% at fixed depth, far less than LMR, and measured the same +86 Elo.
Fixed-depth node counts understate what a pruning change buys at fixed *time*,
because the engine spends the saving on reaching deeper. Node counts are for
catching changes that are clearly *worse*; they do not rank the good ones.


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

Use `-concurrency 2`. This box reports 8 processors but has **4 physical
cores** (i5-10210U, a 15W mobile part); `nproc` counts threads. Each game is two
engine processes, so concurrency 2 fills the physical cores exactly and
concurrency 3 oversubscribes them by half.

Earlier runs in the table below used concurrency 3. The verdicts stand -- the
load is symmetric and the baseline-vs-baseline control confirmed no bias -- but
the extra variance inflates the games needed per verdict, which is part of why
the PVS run burned 5000 games without resolving.

Always pass `-log file=<abs path> level=warn`, and never pipe the run through
`tail`: the SPRT verdict line prints before the per-player summary, so a tail
window silently discards it. `config.json` in the working directory keeps the
W/L/D and pentanomial tallies if you lose the console output anyway.
### Results so far

| Change | Result | Games | Elo |
| --- | --- | --- | --- |
| Quiescence search (milestone 6) | H1 accepted | 354 | +292.3 +/- 36.2 |
| Zobrist + repetition detection (milestone 7a) | H1 accepted | 1864 | +28.4 |
| Transposition table (milestone 7b) | H1 accepted | 600 | +121.1 +/- 25.6 |
| Move ordering: SEE, killers, history (milestone 8) | H1 accepted | 1084 | +60.2 +/- 18.0 |
| PVS alone (milestone 10) | **no verdict** | 5000 | +6.5 +/- 7.5 |
| PVS + null move (milestone 10) | H1 accepted | 468 | +166.0 +/- 29.1 |
| Late move reductions (milestone 10) | H1 accepted | 752 | +86.8 +/- 21.1 |
| Reverse futility pruning (milestone 10) | H1 accepted | 730 | +86.5 +/- 21.2 |
| Late move pruning (milestone 10) | **H0 accepted** | 798 | -79.7 +/- 21.6 |
| Late move pruning, counter bug fixed | **H0 accepted** | 998 | -58.7 +/- 18.1 |
| First NNUE (128x2, 915K HCE-labelled positions) vs m10c | **H0 accepted** | 322 | -385.9 +/- 61.9 |
| Same net at fixed 20K nodes/move (removes the nps gap) | 13-83-4 | 100 | about -300 |

The first network is not a bug: accumulators match a refresh through real
searches, material values are sane, and it tracks held-out labels (r 0.915) as
well as training ones (0.90), so it is not overfitting either. It is an
imperfect copy of the handcrafted evaluation it was labelled by, at half the
nps. A net only beats its teacher once enough game-result signal (hundreds of
millions of positions) or better labels outweigh the imitation.

Ruled out, with `netcheck --mappings` on held-out positions: a feature
mismatch with bullet. The engine's mapping fits best (r 0.915); the nearest
wrong one, knight and bishop swapped, gives 0.888, and the rest collapse or go
negative. Found instead: the net's centipawns run about 1.6x the labels
(recorded = 0.615 * net), most likely from the WDL 0.4 blend pulling targets
toward the result extremes. The next diagnostic is a WDL 0.0 net on the same
data: it should reproduce the handcrafted evaluation almost exactly and play
about level with it at fixed nodes. If it does not, something beyond data is
wrong.

**A slow verdict means a small effect.** The number of games SPRT needs falls as
the true gain grows, because the LLR drifts in proportion to how far the effect
sits from the midpoint of the two hypotheses. Our own runs are monotonic in it:

    +292 Elo    354 games
    +121 Elo    600 games
    +60  Elo   1084 games
    +28  Elo   1864 games
    +13  Elo   4000+ games

So budget by expected size. A change worth a few Elo costs hours at
`elo0=0 elo1=5`; if that is too slow, widen the bounds or run a fixed-game A/B
and accept a point estimate instead of a decision.

(An earlier version of this file claimed the opposite -- that effect size did
not change the game count. That was wrong, and was a rationalisation of one run
rather than a reading of the data.)

### Quiet move ordering is the binding constraint

Two independent experiments have now failed on the same assumption:

| attempt | result | what it assumed |
| --- | --- | --- |
| SEE: losing captures ordered last | 3x worse node count | quiets outrank a forcing move |
| Late move pruning (twice) | -80, then -59 Elo | late quiets are noise |

Both are standard, both are in the outline, and both lose here. The shared cause
is that quiet ordering is weak: history is bonus-only, with no malus, no
continuation history and no counter-move heuristic. Killers plus a single
bonus-only table is not enough signal to justify skipping quiet moves or ranking
them above forcing ones.

So the quiet-pruning family is blocked behind quiet-ordering quality. Improve
history first -- malus on moves that failed to cut, continuation tables -- and
only then retry late move pruning and the losing-capture demotion. Retrying
either before that is spending an hour to re-learn this.

### Background processes

**Confirm a background task actually died.** A stray process skews every timing
measurement taken afterwards, and it will not announce itself. One hung
`python` invocation sat at 19% of a core for 22 hours and was present during
every SPRT in this file. Match verdicts survived it, because it loads both
engines equally and the baseline-vs-baseline control confirmed no bias, but
every nps figure taken during that window is understated.

    Get-Process | Where-Object { $_.StartTime -lt (Get-Date).AddHours(-1) }

Two rules that would have prevented it:

- Do not invoke a tool speculatively to see whether it exists. That hang came
  from a `python ... || sed ...` fallback where the `sed` alone was the entire
  fix.
- Never report killing a process without checking it is gone. Reporting an
  action not taken is worse than the hang.





## NNUE

Researched against `bullet` and the literature, not assumed. Anything here that
contradicts the original outline is a deliberate correction.

### The contract with the trainer

- `QA = 255`, `QB = 64`, `SCALE = 400`, confirmed by bullet `examples/simple.rs`.
  A mismatch shows up as evals systematically compressed or exploded, not as a
  training failure.
- Accumulators concatenate side-to-move first. That ordering is what tells the
  network whose turn it is.
- SCReLU is the activation to use; it is dominant and produces the strongest
  network of the three in common use.
- bullet writes `quantised.bin` **little-endian, column-major**, weights shaped
  `output_size x input_size`, padded to a multiple of 64 bytes. The loader has to
  match that exactly.
- bullet `simple.rs` runs a WDL scheduler constant of **0.75** where the original
  outline says 0.3. Check which way round bullet defines it before trusting
  either number.

### Data

- Bulletformat is still supported and is the recommended choice for small nets;
  binpacks are for when loading becomes the bottleneck. Viriformat is what most
  people generating their own data use.
- There is a text intermediate, `<FEN> | <score> | <result>`, score and result
  **white-relative**. Emitting that and converting avoids hand-writing binary
  records, which is exactly where a perspective sign error would hide.
- Filter for quiet positions properly. The outline says skip checks and
  positions whose best move is a capture; measured work gives a sharper test:
  no checks, `|static - qsearch| <= 60`, and `|static - negamax| <= 70`
  centipawns. Deduplicate, or the net overfits.
- Fixed nodes, not fixed depth, so data is hardware-independent. About 5k nodes
  per position, 7 or 8 random opening plies, discard openings already lopsided.
- End a datagen game on insufficient material. Bare kings score a flat draw,
  pass the quiet filter, and would otherwise be recorded all the way to the
  fifty-move rule. bullet `validate` flagged 112 of them in a 16K sample before
  the check existed. Run `validate` on every converted file, and shuffle before
  training: datagen writes whole games in sequence, so unshuffled batches are
  dozens of near-identical positions from one game.

### Scale, honestly

A first 768 -> Nx2 -> 1 net on self-generated data is worth somewhere around
+100 to +200 Elo over a handcrafted evaluation. That is the realistic target.
Competitive engines are far past it: Viridithas currently runs 16 input buckets
and 8 output buckets over 2048x2 -> 16 -> 32 -> 1, and top engines use
accumulators of 1024 to 3072. The gap is architecture *and* data volume, and
data volume is CPU-bound, not GPU-bound.

### Verification

- Incremental accumulator updates must be bit-identical to a full refresh over
  thousands of random move sequences. Same class of bug as mailbox/bitboard
  drift, and it gets the same treatment.
- Every SIMD path keeps its scalar twin, plus a test asserting the two agree bit
  for bit.
- **Training data source: the Lichess evaluation database (decided 2026-09-13).**
  ~410M positions evaluated by Stockfish in users' browsers, CC0, so legally
  unrestricted. The trade-off was taken knowingly: a net trained on it learns
  Stockfish's judgement, TCEC's NNUE guideline asks for data from the engine's
  own search or eval, and much of the engine community frowns on training on
  another engine's output. Our own self-play data remains the fallback if
  originality ever matters. Caveats for the converter: no game results (train
  at WDL 0.0), positions skew toward human analysis rather than self-play, and
  nothing is quiet-filtered.

## Hardware

    Intel Core i5-10210U   4 physical cores / 8 threads, 15W mobile
    Intel UHD Graphics     no CUDA, ROCm or Metal
    15.8 GB RAM

Consequences worth knowing before planning any long run:

- `nproc` reports 8. That is threads. Size match concurrency off **4**.
- Sustained load thermally throttles a U-series part, so long runs are slower
  per unit than short ones. Do not extrapolate a 2-minute benchmark to a
  20-hour job.
- **`bullet` cannot train here.** It needs CUDA, ROCm or Metal. Training a first
  768 net is under an hour on almost any rented GPU, so rent for that step.
- Beware: bullet *compiles* fine with no GPU backend, which is a trap. That build
  runs on a mock runtime that panics at the first gradient ("This is a mock
  runtime! It can not actually do anything!"). A clean build here proves the
  trainer matches the bullet API, not that it trains. An earlier plan to
  smoke-train locally for free rested on the compile succeeding; the first real
  training run has to happen on the rental.
- Datagen is the job this machine is worst at: 100M positions at 5k nodes is
  roughly 5e11 nodes, which is **15-25 hours here**, against under two on a
  rented 32-64 core box. Datagen is embarrassingly parallel, so the CPU rental
  buys more than the GPU rental does.

## Rental guardrails

The budget for all NNUE training is **$15**. These rules exist so that money is
spent computing, not waiting or debugging.

What the platform does, stated plainly: vast.ai bills an instance from the
moment it starts until it is **destroyed**. Setup is billed exactly like
training. A *stopped* instance still bills for disk, and bandwidth is billed per
byte in every state. So "only pay while the model trains" is not something any
script can guarantee. What is guaranteed instead is that time not spent
training is short, capped, and cannot be forgotten about.

1. **Never rent before `trainer/vast/prepare.sh` prints PREFLIGHT PASSED.** It
   processes and validates the data, confirms the trainer and engine build
   against their lockfiles, checks the scripts are LF and parse, and builds and
   re-verifies the upload bundle. Everything that can fail without a GPU fails
   here, for free.
2. **Fail fast on the instance.** `run.sh` checks the GPU, the exact CUDA
   libraries bullet links, the self-destroy key, and the budget before
   installing Rust or compiling anything.
3. **Hard spending cap.** `run.sh` requires `PRICE_PER_HOUR` and caps instance
   time at `MAX_DOLLARS / PRICE_PER_HOUR` hours (default `MAX_DOLLARS=3`),
   enforced with `timeout`, with the download window reserved inside the cap.
   Boot and upload time before the script starts is billed but not counted, so
   start it promptly.
4. **The instance destroys itself.** vast.ai injects `CONTAINER_ID` and a
   `CONTAINER_API_KEY` scoped to starting, stopping or destroying only that
   instance, so no account key ever goes on the box. On any exit -- success,
   failure, cap reached, or a dropped SSH session -- `run.sh` destroys the
   instance after a download window: 15 minutes on success, 5 on failure.
   Destroy, never stop. The only opt-out is deliberate:
   `touch /tmp/KEEP_INSTANCE`. `run.sh` refuses to start at all if
   self-destroy is unavailable, unless `NO_AUTODESTROY=1` is set on purpose.
5. **Validate before the expensive step.** A smoke train on 16K verified
   positions, then an on-box `netcheck`, must pass before the real training run
   starts. A perspective bug costs cents to find, not dollars.
6. **Check the bill afterwards.** Confirm in the vast.ai console that no
   instance remains listed and the charge matches the estimate. If one is still
   listed, destroy it by hand.

Measured, first session (2026-09-13, RTX 4090 at $0.372/hr): 12 minutes from
rent to destroy, **$0.07 total**. Almost all of it was boot, upload and the
CUDA build; training 914K positions for 40 superbatches took **6.7 seconds**
at 7.3M positions/sec. GPU time is not the constraint at this net size --
data is. Destroy by hand as soon as the network is downloaded and verified
rather than waiting out the download window.

Lichess sessions (`prepare.sh --lichess`, then `run.sh lichess`) download the
database on the instance, so rent with at least 100 GB of disk and a fast
connection, and set `MAX_DOLLARS` for a multi-hour run.

**Do not stream from database.lichess.org.** Session 2 (2026-09-13) failed
after 19 minutes: the server closed the connection mid-transfer (curl exit 18)
after ~190 MB, and a separate test measured it at ~170 KB/s, i.e. ~36 hours for
the 22 GB file. The guardrails worked -- the instance destroyed itself and the
session cost ~$0.13 -- but the source is unusable at this size. The same CC0
data is mirrored on Hugging Face (`Lichess/chess-position-evaluations`,
parquet), served from a CDN.

Running from the browser: the Jupyter "direct HTTPS" links need vast's root
certificate installed, which changes browser security settings -- don't. The
Instance Portal (plain HTTP on the instance IP) lists Cloudflare tunnels with
normal certificates; use the tunnel for port 8080 for Jupyter, the terminal,
and uploads through the contents API.

## Profiles

- `cargo build --release` — `lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`, `opt-level = 3`. Use for playing strength and benchmarks.
- `cargo build --profile dev-fast` — `opt-level = 2` with debug assertions and
  overflow checks on. Use for day-to-day iteration.

## Features

- `datagen` — self-play data generation for NNUE training. Off by default.
  Nothing behind this feature may be referenced from the search hot path.
- `magicgen` — the offline magic-constant search. Run once; the output is
  committed as `src/magic_constants.rs`.
- `bookgen` — opening-book generation for SPRT testing.

## Dependencies

Keep them minimal. Current set: `arrayvec` (move lists), `criterion`
(dev-only, benchmarks), and `ruzstd` (optional, behind `datagen` only: decodes
the Lichess evaluation database for `lichesseval`, in pure Rust so it builds on
a bare rental). That is the whole list, and the engine build itself still
depends on `arrayvec` alone.

`rayon` was dropped: datagen parallelises with `std::thread::scope`, one
independent worker per thread with nothing shared, so the dependency bought
nothing. An unused dependency is worse than no dependency.

No `serde`. No `rand` in the hot path — if randomness is needed there, use a
small explicit PRNG (e.g. xorshift) written in-tree.

## Benchmarks

`benches/engine.rs` is a criterion target wired up but empty. Add perft, movegen,
eval, and fixed-depth search benchmarks as the corresponding code lands.
