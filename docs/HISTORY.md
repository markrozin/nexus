# How Nexus got here

A narrative record of the decisions behind the code, written so the project can
be picked up on another machine — or months later — without re-deriving them.
`CLAUDE.md` holds the conventions and the full results tables; this file holds
the story and the reasoning. Dates are the commit dates.

## 1. Foundations (2026-09-09 to 09-10)

Built in strict dependency order, because a move generation bug that reaches
the search presents as a search bug and costs days:

    bitboards -> attacks -> movegen -> perft -> UCI -> search -> eval -> NNUE

The board is stored twice — bitboards for "where are all the white rooks",
a mailbox for "what is on e4" — with every write funnelled through `put`,
`remove` and `move_piece` so the two cannot drift apart. Sliding pieces use
magic bitboards; the slow ray-walking version stays as the reference.

**The gate:** all six Chess Programming Wiki perft positions at full depth,
~1.45 billion nodes. Nothing proceeded until they matched exactly.

## 2. Search, one SPRT at a time (09-10 to 09-13)

Every heuristic landed behind its own sequential test against the previous
build. The measured gains, in order: quiescence +292, Zobrist/repetition +28,
transposition table +121, move ordering +60, PVS plus null move +166, late move
reductions +87, reverse futility +86.

Three findings worth keeping:

- **Late move pruning lost**, twice: -80 Elo, then -59 after a counter bug was
  fixed. It is standard in other engines and it does not work here.
- **Losing captures belong ahead of quiet moves**, not last as the textbooks
  say. Ordered last they measured 3x worse node counts.
- Both failures share a cause: quiet move ordering is weak (history is
  bonus-only, with no malus, no continuation tables, no counter-move table).
  The quiet-pruning family stays blocked until that improves.

**The method is the point.** Reading the code does not tell you which
heuristics gain 20 Elo and which lose 5. A 12-game control run of two identical
binaries read -232 Elo, which is why verdicts come from SPRT and not from
eyeballing a handful of games.

## 3. The first network failed, and why (09-13)

Trained on 915K self-play positions labelled by the handcrafted evaluation. It
lost to that same evaluation by **386 Elo**.

Not a bug, and the diagnosis mattered more than the fix:

- Incremental accumulators matched full refreshes.
- It tracked held-out labels (r 0.915) as well as training ones (0.90), so it
  was not overfitting.
- `netcheck --mappings` ruled out a feature-layout mismatch with the trainer:
  the engine's mapping fit best, the nearest wrong one (knight/bishop swapped)
  clearly worse.

**Conclusion:** the net was an imperfect copy of its teacher at half the speed.
A student cannot beat its teacher by imitating it. The fix was better labels,
not better code.

## 4. Data: the decision that unlocked everything (09-13)

Chose the **Lichess evaluation database** — ~410M positions analysed by
Stockfish in users' browsers, CC0 — over self-generated data, read in practice
from the deduplicated Hugging Face mirror
`mateuszgrzyb/lichess-stockfish-normalized` (CC BY 4.0; credit it and Lichess).

The trade-off was taken knowingly: the network learns Stockfish's judgement
rather than its own, which the engine community frowns on and TCEC's guideline
forbids. Against that, this laptop generates ~1M self-play positions overnight
while the database offers 191.5M usable ones immediately. The self-play
generator remains as a fallback if originality ever matters more than strength.

**Do not stream from `database.lichess.org`** — it closed the connection after
~190 MB at ~170 KB/s, i.e. 36 hours for the 22 GB file.

## 5. Renting GPUs without wasting money (09-13 to 09-14)

`bullet` needs CUDA, so training cannot happen on this laptop. Budget for the
whole project: **$15**. vast.ai bills from instance start until *destroy*, so
"only pay while training" is not something any script can promise; what the
guardrails promise instead is that non-training time is short, capped, and
cannot be forgotten.

Five sessions failed before one worked, each leaving a rule in `CLAUDE.md`:
a bare image never starts sshd (use the official CUDA template); an
`a && b && nohup c &` over ssh holds the connection open for the whole run
(upload a launcher, start it detached); an unbounded quiescence check in the
data converter stalled for six hours until the spending cap killed it (bound
every long stage and add stall detectors).

Session 6 worked end to end: 23 minutes, ~$0.20, 191.5M positions converted and
a net trained. Total spend across all sessions: **~$3.30 of $15**.

## 6. The network that worked (09-15 to 09-17)

- 128-wide net on 191.5M Lichess positions beat the handcrafted engine **34-0**
  and became the default.
- A later session trained 256 and 512 widths in one rental (~$0.47). The
  **512-wide** net beat the 128 by **+87 Elo** over 360 games and now ships as
  `networks/default.bin`. Width costs speed (1.66 -> 1.19 Mnps) and pays for it.
- Width below 512 is a build feature (`hidden128`, `hidden256`) because the
  accumulator is a fixed-size stack array.

## 7. Making it fast (09-17)

The output layer was ~40% of a leaf evaluation. The cause was 64-bit
multiplies, which no instruction set vectorises; neither `target-cpu=native`
nor restructuring the scalar loop recovered it.

The fix: prove at load time that the loaded weights cannot overflow 32-bit
lanes, then hand-write the AVX2 path. **238ns -> 138ns per evaluation, +17%
nodes per second at identical node counts, +22.6 Elo over 740 games.** The
scalar twin stays, and a test asserts every path agrees bit for bit — a
one-centipawn divergence would make the engine play differently per machine.

## 8. Absolute strength (09-18)

Self-play margins say nothing about real rating, so the engine played engines
with published ratings (10+0.1, equal hash, `sprt/gauntlet.sh`):

| Opponent | Rating | Result |
| --- | --- | --- |
| Cinnamon 2.5 | 2308 | 16-0-1 |
| Stash v21 | 2714 | 34-3-3 |
| Stash v25 | 2937 | 121-60-19, +109 +/- 42 |
| Anna 1.2 | ~3575 self-reported | 0-30 |

That puts Nexus around **3045** under these conditions, **2800+** stated
conservatively. Anna's lead is architectural: 1024-wide threat-input nets on
Leela plus Stockfish data, aspiration windows, ProbCut, singular extensions.

## 9. Where to pick up

Free, on any laptop:

1. **Quiet move ordering** — history malus, continuation history, counter-move
   table. This unblocks the pruning family that failed twice.
2. **Aspiration windows** — standard, and absent.
3. **Singular extensions** — worth 10-36 Elo elsewhere.

Paid, ~$0.50 a session: a **1024-wide net**, and eventually threat inputs.

Also open: multi-threaded search (the `Threads` option is parsed and ignored),
and on Apple Silicon a NEON path to match what AVX2 does on x86.
