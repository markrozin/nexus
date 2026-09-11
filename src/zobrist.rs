//! Zobrist hashing: a 64-bit key identifying a position.
//!
//! One random word per (piece, square), plus one for the side to move, one per
//! castling-rights combination, and one per en passant file. A position's key
//! is the XOR of all the active ones.
//!
//! XOR is its own inverse, which is the whole point: moving a knight from b1 to
//! c3 is `key ^= PIECE[N][b1] ^ PIECE[N][c3]`, so the key rides along with
//! `make_move` instead of being recomputed. [`crate::board::Position`] does that
//! incrementally and `assert_invariants` checks the result against
//! [`compute`] from scratch — an incremental-update bug shows up later as
//! corrupted transposition-table hits, which is close to untraceable.
//!
//! The table is built by a `const fn`, so it is baked into the binary. Nothing
//! is generated at startup and the keys are identical across runs, which keeps
//! searches reproducible.

use crate::types::{CastlingRights, Color, Piece, PieceType, Square};

/// Number of distinct castling-rights states: four independent bits.
const CASTLING_STATES: usize = 16;

/// xorshift64. Enough mixing for hash keys and evaluable at compile time.
const fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

struct Keys {
    piece_square: [[u64; Square::COUNT]; Piece::COUNT],
    castling: [u64; CASTLING_STATES],
    en_passant_file: [u64; 8],
    side_to_move: u64,
}

const fn build_keys() -> Keys {
    let mut keys = Keys {
        piece_square: [[0; Square::COUNT]; Piece::COUNT],
        castling: [0; CASTLING_STATES],
        en_passant_file: [0; 8],
        side_to_move: 0,
    };

    // Any fixed seed does; this one is arbitrary.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    let mut piece = 0;
    while piece < Piece::COUNT {
        let mut sq = 0;
        while sq < Square::COUNT {
            state = xorshift(state);
            keys.piece_square[piece][sq] = state;
            sq += 1;
        }
        piece += 1;
    }

    // Index 0 (no rights) stays zero, so a bare board with White to move and no
    // en passant hashes to 0 and `Position::empty` can stay a `const fn`.
    let mut rights = 1;
    while rights < CASTLING_STATES {
        state = xorshift(state);
        keys.castling[rights] = state;
        rights += 1;
    }

    let mut file = 0;
    while file < 8 {
        state = xorshift(state);
        keys.en_passant_file[file] = state;
        file += 1;
    }

    state = xorshift(state);
    keys.side_to_move = state;
    keys
}

static KEYS: Keys = build_keys();

#[inline]
pub fn piece_square(piece: Piece, sq: Square) -> u64 {
    KEYS.piece_square[piece.index()][sq.index()]
}

/// Indexed by the whole rights mask rather than per-right, so a change costs
/// one XOR out and one XOR in regardless of how many rights moved.
#[inline]
pub fn castling(rights: CastlingRights) -> u64 {
    KEYS.castling[rights.bits() as usize]
}

/// Only the file matters: the rank is implied by the side to move.
#[inline]
pub fn en_passant(sq: Square) -> u64 {
    KEYS.en_passant_file[sq.file() as usize]
}

/// XOR-ed in exactly when it is Black to move.
#[inline]
pub fn side_to_move() -> u64 {
    KEYS.side_to_move
}

/// The en passant contribution for a position, which is zero unless a pawn can
/// actually make the capture.
///
/// A FEN records the en passant square after any double push, whether or not it
/// is capturable. Hashing it unconditionally would give two positions that play
/// identically different keys, and the transposition table would miss every
/// such match.
#[inline]
pub fn en_passant_if_capturable(
    ep_square: Option<Square>,
    side_to_move: Color,
    their_pawns: crate::bitboard::Bitboard,
) -> u64 {
    match ep_square {
        // A pawn of `side_to_move` attacks `ep` exactly where an enemy pawn
        // standing on `ep` would attack.
        Some(ep)
            if (crate::bitboard::pawn_attacks(side_to_move.flip(), ep) & their_pawns).any() =>
        {
            en_passant(ep)
        }
        _ => 0,
    }
}

/// Recompute a key from scratch. The reference [`crate::board::Position`]
/// incremental updates are checked against.
pub fn compute(pos: &crate::board::Position) -> u64 {
    let mut key = 0u64;
    for sq in Square::ALL {
        if let Some(piece) = pos.piece_at(sq) {
            key ^= piece_square(piece, sq);
        }
    }
    key ^= castling(pos.castling());
    key ^= en_passant_if_capturable(
        pos.ep_square(),
        pos.side_to_move(),
        pos.pieces(pos.side_to_move(), PieceType::Pawn),
    );
    if pos.side_to_move() == Color::Black {
        key ^= side_to_move();
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_key_is_distinct_and_nonzero() {
        let mut seen = HashSet::new();
        for piece in Piece::ALL {
            for sq in Square::ALL {
                let key = piece_square(piece, sq);
                assert_ne!(key, 0, "{piece:?} on {sq}");
                assert!(seen.insert(key), "duplicate key for {piece:?} on {sq}");
            }
        }
        // No-rights is deliberately zero; the other fifteen are distinct.
        assert_eq!(castling(CastlingRights::NONE), 0);
        for bits in 1..CASTLING_STATES as u8 {
            let key = castling(CastlingRights::from_bits(bits));
            assert_ne!(key, 0);
            assert!(seen.insert(key), "duplicate castling key for {bits:04b}");
        }
        for file in 0..8 {
            let key = en_passant(Square::new(file, 2));
            assert!(seen.insert(key), "duplicate ep key for file {file}");
        }
        assert!(seen.insert(side_to_move()));
        assert_eq!(seen.len(), Piece::COUNT * Square::COUNT + 15 + 8 + 1);
    }

    #[test]
    fn en_passant_only_hashes_when_a_capture_exists() {
        use crate::board::Position;

        // Black just played d7d5. White has a pawn on e5 and can take.
        let capturable: Position = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
            .parse()
            .unwrap();
        // Same double push, but White has no pawn in range.
        let idle: Position = "rnbqkbnr/ppp1pppp/8/3p4/8/7P/PPPPPPP1/RNBQKBNR w KQkq d6 0 3"
            .parse()
            .unwrap();

        assert_ne!(
            en_passant_if_capturable(
                capturable.ep_square(),
                capturable.side_to_move(),
                capturable.pieces(Color::White, PieceType::Pawn)
            ),
            0
        );
        assert_eq!(
            en_passant_if_capturable(
                idle.ep_square(),
                idle.side_to_move(),
                idle.pieces(Color::White, PieceType::Pawn)
            ),
            0,
            "an uncapturable en passant square must not change the key"
        );

        // And the idle position must hash the same as the one with no ep square
        // recorded at all, since they play identically.
        let no_ep: Position = "rnbqkbnr/ppp1pppp/8/3p4/8/7P/PPPPPPP1/RNBQKBNR w KQkq - 0 3"
            .parse()
            .unwrap();
        assert_eq!(compute(&idle), compute(&no_ep));
    }
}
