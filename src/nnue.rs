//! NNUE evaluation: a `(768 -> HIDDEN)x2 -> 1` network with SCReLU, loaded
//! from the `quantised.bin` that bullet writes.
//!
//! # The contract with the trainer
//!
//! Every constant and layout choice here has to match bullet exactly. A
//! mismatch does not fail loudly: it produces evaluations that are wrong by a
//! consistent factor or sign, and an engine that plays badly for no visible
//! reason. So each one is written down with where it came from.
//!
//! - `QA = 255`, `QB = 64`, `SCALE = 400`, from bullet `examples/simple.rs`.
//! - The file holds four little-endian `i16` tensors in this order: feature
//!   weights at `QA`, feature bias at `QA`, output weights at `QB`, output bias
//!   at `QA * QB`. It is padded to a multiple of 64 bytes.
//! - Weights are column-major with shape `output x input`. For the feature
//!   layer that makes feature `j`'s weights one contiguous run at
//!   `j * HIDDEN` -- exactly the slice an accumulator update adds.
//! - The two accumulators are concatenated **side to move first**. That order is
//!   the only thing that tells the network whose turn it is.
//!
//! # Features
//!
//! bullet's `Chess768` works on a board stored side-to-move relative:
//!
//! ```text
//! stm = [0, 384][colour] + 64 * piece + square
//! ntm = [384, 0][colour] + 64 * piece + (square ^ 56)
//! ```
//!
//! This engine keeps an absolute board, so the same indices come out as
//! [`feature`]: a piece of the perspective's own colour sits in the first 384,
//! and a Black perspective sees the board mirrored vertically.
//!
//! One assumption is not checked against bullet source directly: that its
//! piece-type numbering is pawn, knight, bishop, rook, queen, king, the same as
//! [`crate::types::PieceType`]. That is the universal convention, and a trained network would
//! expose a mismatch immediately -- evaluations stop correlating with the
//! recorded search scores. Validate it that way before trusting a net.

use std::path::Path;
use std::sync::LazyLock;

use crate::board::Position;
use crate::types::{Color, Piece, Square};

/// The network compiled into the binary, so the engine plays at full strength
/// with no files beside it. Replace `networks/default.bin` to ship a new net;
/// the loader's size check fails the build-time test if its shape is wrong.
static EMBEDDED_BYTES: &[u8] = include_bytes!("../networks/default.bin");

static EMBEDDED: LazyLock<Network> = LazyLock::new(|| {
    Network::from_bytes(EMBEDDED_BYTES).expect("networks/default.bin matches HIDDEN")
});

/// The embedded network, parsed once on first use.
pub fn embedded() -> &'static Network {
    &EMBEDDED
}

pub const INPUTS: usize = 768;

/// Accumulator width. Must match the trainer's `HIDDEN_SIZE`; the loader
/// rejects a file of any other size rather than reading garbage.
pub const HIDDEN: usize = 128;

pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

/// Number of `i16` values in a quantised network of this shape.
const PAYLOAD_I16S: usize = INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN + 1;

/// Feature index of `piece` on `sq`, as seen from `perspective`.
///
/// Equivalent to bullet's `Chess768` once its side-to-move-relative board is
/// translated to an absolute one.
#[inline]
pub fn feature(perspective: Color, piece: Piece, sq: Square) -> usize {
    let own = if piece.color() == perspective { 0 } else { 384 };
    let square = match perspective {
        Color::White => sq.index(),
        Color::Black => sq.index() ^ 56,
    };
    own + piece.piece_type().index() * 64 + square
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    /// The file is not the size a `HIDDEN`-wide network would be. Almost always
    /// a network trained with a different hidden size.
    WrongSize { expected: usize, found: usize },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "cannot read network: {e}"),
            LoadError::WrongSize { expected, found } => write!(
                f,
                "network is {found} bytes but a {HIDDEN}-wide net is {expected} \
                 (or that rounded up to 64): was it trained with a different hidden size?"
            ),
        }
    }
}

impl std::error::Error for LoadError {}

pub struct Network {
    /// `INPUTS * HIDDEN`; feature `j` at `j * HIDDEN`.
    feature_weights: Box<[i16]>,
    feature_bias: Box<[i16]>,
    /// `2 * HIDDEN`: the side-to-move half, then the other.
    output_weights: Box<[i16]>,
    output_bias: i16,
}

impl Network {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let bytes = std::fs::read(path).map_err(LoadError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a bullet `quantised.bin`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let exact = PAYLOAD_I16S * 2;
        let padded = exact.div_ceil(64) * 64;
        if bytes.len() != exact && bytes.len() != padded {
            return Err(LoadError::WrongSize {
                expected: exact,
                found: bytes.len(),
            });
        }

        let mut values = bytes
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]));
        let mut take = |n: usize| -> Box<[i16]> { values.by_ref().take(n).collect() };

        let feature_weights = take(INPUTS * HIDDEN);
        let feature_bias = take(HIDDEN);
        let output_weights = take(2 * HIDDEN);
        let output_bias = take(1)[0];

        Ok(Self {
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        })
    }

    #[inline]
    fn column(&self, index: usize) -> &[i16] {
        &self.feature_weights[index * HIDDEN..(index + 1) * HIDDEN]
    }

    /// Evaluate from a ready accumulator, in centipawns, side-to-move relative.
    ///
    /// The arithmetic follows bullet's reference inference exactly, including
    /// where each division falls, so a network scores the same here as there.
    /// The running sum is 64-bit: with trained weights it can come within a few
    /// percent of `i32::MAX`, and a silent wrap would be far worse than the
    /// negligible cost.
    pub fn evaluate(&self, acc: &Accumulator, side_to_move: Color) -> i32 {
        let us = &acc.values[side_to_move.index()];
        let them = &acc.values[side_to_move.flip().index()];

        let mut sum: i64 = 0;
        for i in 0..HIDDEN {
            sum += screlu(us[i]) as i64 * self.output_weights[i] as i64;
            sum += screlu(them[i]) as i64 * self.output_weights[HIDDEN + i] as i64;
        }

        // SCReLU squares a QA-scaled value, so the sum carries QA*QA*QB. One
        // division by QA brings it to the bias's QA*QB scale; the last removes
        // that and applies the centipawn scale.
        let mut out = sum / QA as i64;
        out += self.output_bias as i64;
        out *= SCALE as i64;
        out /= (QA * QB) as i64;
        out as i32
    }

    /// Full evaluation of a position from scratch. Correct but slow: it rebuilds
    /// the accumulator every call. The search should keep accumulators up to
    /// date incrementally instead.
    pub fn evaluate_position(&self, pos: &Position) -> i32 {
        self.evaluate(&Accumulator::refresh(pos, self), pos.side_to_move())
    }
}

/// Squared clipped ReLU on a QA-scaled value.
#[inline]
fn screlu(x: i16) -> i32 {
    let v = (x as i32).clamp(0, QA);
    v * v
}

/// The first-layer output for both perspectives, indexed by absolute colour.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Accumulator {
    values: [[i16; HIDDEN]; Color::COUNT],
}

impl std::fmt::Debug for Accumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Accumulator {{ white[0..4]: {:?}, black[0..4]: {:?} }}",
            &self.values[0][..4], &self.values[1][..4])
    }
}

impl Accumulator {
    /// Build from scratch: start at the bias, add one column per piece.
    pub fn refresh(pos: &Position, net: &Network) -> Self {
        let mut acc = Self {
            values: [[0; HIDDEN]; Color::COUNT],
        };
        for perspective in Color::ALL {
            acc.values[perspective.index()].copy_from_slice(&net.feature_bias);
        }
        for sq in Square::ALL {
            if let Some(piece) = pos.piece_at(sq) {
                acc.add(net, piece, sq);
            }
        }
        acc
    }

    #[inline]
    pub fn add(&mut self, net: &Network, piece: Piece, sq: Square) {
        for perspective in Color::ALL {
            let column = net.column(feature(perspective, piece, sq));
            for (value, weight) in self.values[perspective.index()].iter_mut().zip(column) {
                // Trained weights are clipped well inside i16 across a full
                // board; wrapping matches what a SIMD path would do regardless.
                *value = value.wrapping_add(*weight);
            }
        }
    }

    #[inline]
    pub fn sub(&mut self, net: &Network, piece: Piece, sq: Square) {
        for perspective in Color::ALL {
            let column = net.column(feature(perspective, piece, sq));
            for (value, weight) in self.values[perspective.index()].iter_mut().zip(column) {
                *value = value.wrapping_sub(*weight);
            }
        }
    }

    /// Bring an accumulator for `before` up to date with `after`.
    ///
    /// Diffs the two mailboxes rather than decoding the move, which makes
    /// castling, en passant, promotion and capture all the same case: whatever
    /// left a square is subtracted, whatever arrived is added. A move touches at
    /// most four squares, so this is a handful of column updates plus a 64-way
    /// compare -- far cheaper than a refresh, and with nothing move-specific to
    /// get wrong.
    pub fn update(&mut self, net: &Network, before: &Position, after: &Position) {
        for sq in Square::ALL {
            let old = before.piece_at(sq);
            let new = after.piece_at(sq);
            if old == new {
                continue;
            }
            if let Some(piece) = old {
                self.sub(net, piece, sq);
            }
            if let Some(piece) = new {
                self.add(net, piece, sq);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;
    use crate::rng::Rng;

    /// A network in bullet's exact on-disk layout, with small deterministic
    /// weights so nothing overflows. Going through the byte format means the
    /// tests exercise the loader, not just the arithmetic.
    fn synthetic_bytes(seed: u64, pad: bool) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        let mut small = || (rng.below(101) as i32 - 50) as i16;
        let mut values = Vec::with_capacity(PAYLOAD_I16S);
        for _ in 0..PAYLOAD_I16S {
            values.push(small());
        }
        let mut bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        if pad {
            bytes.resize(bytes.len().div_ceil(64) * 64, 0);
        }
        bytes
    }

    fn synthetic(seed: u64) -> Network {
        Network::from_bytes(&synthetic_bytes(seed, true)).expect("synthetic net loads")
    }

    #[test]
    fn features_cover_the_input_space_exactly_once() {
        for perspective in Color::ALL {
            let mut seen = vec![false; INPUTS];
            for piece in Piece::ALL {
                for sq in Square::ALL {
                    let index = feature(perspective, piece, sq);
                    assert!(index < INPUTS);
                    assert!(!seen[index], "{perspective:?} {piece:?} {sq} collides");
                    seen[index] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "every input index is reachable");
        }
    }

    #[test]
    fn features_match_bullet_chess768() {
        // bullet indexes a side-to-move-relative board. Build that board by hand
        // from the absolute one and check both formulas agree for every piece on
        // every square, from both sides.
        for stm in Color::ALL {
            for piece in Piece::ALL {
                for sq in Square::ALL {
                    // Relative colour bit and square as bullet stores them.
                    let c = usize::from(piece.color() != stm);
                    let rel_sq = match stm {
                        Color::White => sq.index(),
                        Color::Black => sq.index() ^ 56,
                    };
                    let pc = 64 * piece.piece_type().index();
                    let bullet_stm = [0, 384][c] + pc + rel_sq;
                    let bullet_ntm = [384, 0][c] + pc + (rel_sq ^ 56);

                    assert_eq!(feature(stm, piece, sq), bullet_stm, "stm {stm:?} {piece:?} {sq}");
                    assert_eq!(
                        feature(stm.flip(), piece, sq),
                        bullet_ntm,
                        "ntm {stm:?} {piece:?} {sq}"
                    );
                }
            }
        }
    }

    #[test]
    fn loader_accepts_exact_and_padded_and_rejects_other_sizes() {
        assert!(Network::from_bytes(&synthetic_bytes(1, false)).is_ok());
        assert!(Network::from_bytes(&synthetic_bytes(1, true)).is_ok());
        let mut short = synthetic_bytes(1, false);
        short.pop();
        short.pop();
        assert!(matches!(
            Network::from_bytes(&short),
            Err(LoadError::WrongSize { .. })
        ));
        // Padding is the only slack allowed; the same net padded or not must
        // evaluate identically.
        let exact = Network::from_bytes(&synthetic_bytes(7, false)).unwrap();
        let padded = Network::from_bytes(&synthetic_bytes(7, true)).unwrap();
        let pos = Position::startpos();
        assert_eq!(exact.evaluate_position(&pos), padded.evaluate_position(&pos));
    }

    /// Mirror a FEN vertically and swap colours: the same game from the other
    /// side's chair.
    fn mirror(fen: &str) -> String {
        let f: Vec<&str> = fen.split(' ').collect();
        let swap = |c: char| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c.to_ascii_uppercase()
            }
        };
        let placement = f[0]
            .split('/')
            .rev()
            .map(|rank| rank.chars().map(swap).collect::<String>())
            .collect::<Vec<_>>()
            .join("/");
        let stm = if f[1] == "w" { "b" } else { "w" };
        let castling = if f[2] == "-" {
            "-".to_string()
        } else {
            f[2].chars().map(swap).collect()
        };
        let ep = if f[3] == "-" {
            "-".to_string()
        } else {
            let mut chars = f[3].chars();
            let file = chars.next().unwrap();
            let rank = match chars.next().unwrap() {
                '3' => '6',
                '6' => '3',
                other => other,
            };
            format!("{file}{rank}")
        };
        format!("{placement} {stm} {castling} {ep} {} {}", f[4], f[5])
    }

    #[test]
    fn evaluation_is_colour_symmetric() {
        // The decisive perspective test. A position and its colour mirror are the
        // same game for the side to move, so any correct network must score
        // them identically. Getting the square flip, the colour swap or the
        // concatenation order wrong breaks this for almost every position.
        let net = synthetic(42);
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        ] {
            let pos: Position = fen.parse().unwrap();
            let flipped: Position = mirror(fen).parse().unwrap_or_else(|e| panic!("{fen}: {e}"));
            assert_eq!(
                net.evaluate_position(&pos),
                net.evaluate_position(&flipped),
                "{fen} vs its mirror"
            );
        }
    }

    #[test]
    fn incremental_updates_match_a_full_refresh() {
        // Play random games and, after every single move, require the
        // incrementally updated accumulator to be bit-identical to a refresh.
        // An incremental bug produces an engine that plays almost correctly and
        // loses a great deal of strength for no visible reason.
        let net = synthetic(9);
        let mut rng = Rng::new(0xACC);
        for _ in 0..40 {
            let mut pos = Position::startpos();
            let mut acc = Accumulator::refresh(&pos, &net);
            for _ in 0..120 {
                let moves = generate_legal(&pos);
                let Some(pick) = rng.choose(moves.len()) else {
                    break;
                };
                let next = pos.make_move(moves[pick]);
                acc.update(&net, &pos, &next);
                assert_eq!(
                    acc,
                    Accumulator::refresh(&next, &net),
                    "drift after {} in {}",
                    moves[pick],
                    pos.to_fen()
                );
                pos = next;
            }
        }
    }

    #[test]
    fn special_moves_update_correctly() {
        // Random games reach these only occasionally, so check each directly.
        let net = synthetic(3);
        for (fen, uci) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6"),
            ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", "g2h1q"),
            ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1", "b7a8n"),
        ] {
            let pos: Position = fen.parse().unwrap();
            let mv = generate_legal(&pos)
                .iter()
                .copied()
                .find(|m| m.to_string() == uci)
                .unwrap_or_else(|| panic!("{uci} not legal in {fen}"));
            let next = pos.make_move(mv);
            let mut acc = Accumulator::refresh(&pos, &net);
            acc.update(&net, &pos, &next);
            assert_eq!(acc, Accumulator::refresh(&next, &net), "{uci} in {fen}");
        }
    }

    #[test]
    fn the_embedded_network_loads_and_behaves_like_a_chess_evaluation() {
        // A trained net, not a synthetic one: the start position is level, and
        // being a queen up is clearly good for the side that has it.
        let net = embedded();
        let start = net.evaluate_position(&Position::startpos());
        assert!(start.abs() < 100, "start position scored {start}");

        let queen_up: Position = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        let score = net.evaluate_position(&queen_up);
        assert!(score > 400, "a queen up scored {score}");
        // The mirror swaps colours *and* the side to move, so the side to move is
        // still the one a queen up: same score, not the negation.
        let mirrored: Position = mirror(&queen_up.to_fen()).parse().unwrap();
        assert_eq!(net.evaluate_position(&mirrored), score, "colour mirror");
        // Hand the move to the side a queen down instead, and it must be bad.
        let other_to_move: Position = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
            .parse()
            .unwrap();
        let down = net.evaluate_position(&other_to_move);
        assert!(down < -400, "a queen down scored {down}");
    }

    #[test]
    fn evaluation_arithmetic_matches_bullet_reference() {
        // Recompute one evaluation by the exact steps in bullet's example, in
        // plain i32 as bullet does it, and require agreement.
        let net = synthetic(11);
        let pos: Position = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
            .parse()
            .unwrap();
        let acc = Accumulator::refresh(&pos, &net);
        let stm = pos.side_to_move();

        let mut output: i32 = 0;
        let us = &acc.values[stm.index()];
        let them = &acc.values[stm.flip().index()];
        for (input, weight) in us.iter().chain(them).zip(net.output_weights.iter()) {
            output += screlu(*input) * i32::from(*weight);
        }
        output /= QA;
        output += i32::from(net.output_bias);
        output *= SCALE;
        output /= QA * QB;

        assert_eq!(net.evaluate(&acc, stm), output);
    }
}
