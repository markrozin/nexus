# Nexus

A chess engine written from scratch in Rust: bitboards with magic sliders,
alpha-beta search with quiescence, and an NNUE evaluation.

## Status

| Part | State |
| --- | --- |
| Board, move generation | All six Chess Programming Wiki perft positions pass at full depth (~1.45 billion nodes) |
| Search | Negamax, alpha-beta, transposition table, PVS, null move, late move reductions, reverse futility |
| Evaluation | Tapered piece-square tables by default; NNUE (768->128x2->1, SCReLU) opt-in |
| UCI | `uci`, `position`, `go`, `stop`, `setoption` (Hash, Threads, EvalFile) |

Every search or evaluation change is gated on an SPRT against the previous
build; the measured results are in [CLAUDE.md](CLAUDE.md), including the
changes that were tested and **rejected**.

## Building

    cargo build --release          # the engine
    cargo test --release           # the test suite
    cargo test --release -- --ignored --nocapture   # full-depth perft, ~90s

## Networks

`networks/candidates/lichess-wdl0-40.bin` is trained on 191.5M positions from
the Lichess evaluation database. Load one without rebuilding:

    setoption name EvalFile value networks/candidates/lichess-wdl0-40.bin

`<handcrafted>` selects the piece-square evaluation, `<embedded>` the network
compiled into the binary.

## Playing in a browser

`web/` compiles the engine to WebAssembly behind a small C ABI (no
wasm-bindgen) and `web/build.sh` inlines the module into a single self-contained
page:

    rustup target add wasm32-unknown-unknown
    bash web/build.sh              # writes dist/web/nexus.html

Open that file in any browser to play. Strength is set by node count per move,
because WebAssembly has no clock for the engine to time itself by.

## Training

`trainer/` is a separate crate that trains networks with
[bullet](https://github.com/jw1912/bullet) on a rented GPU.
`trainer/vast/rent.sh` runs a whole session through the vast.ai API -- rent,
upload, convert data, train, fetch the network, destroy the instance -- under a
spending cap, with the destroy verified against the account rather than assumed.

## Credits

Training data: the [Lichess evaluation database](https://database.lichess.org/)
(CC0), via the deduplicated mirror
[mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)
(CC BY 4.0). Those evaluations come from Stockfish running in Lichess users'
browsers.
