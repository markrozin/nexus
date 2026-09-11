//! Static exchange evaluation.
//!
//! Answers one question: if both sides keep recapturing on this square until
//! neither wants to continue, does the side to move come out ahead?
//!
//! MVV-LVA alone cannot tell `RxP` where the pawn is defended (losing a rook for
//! a pawn) from `RxP` where it is not (winning a pawn). Both look like "rook
//! takes pawn". SEE separates them, which is what lets the search order losing
//! captures last and lets quiescence skip them entirely.
//!
//! The method is the classic swap list: play out the exchange always using the
//! least valuable attacker, recording the running balance, then fold the list
//! back from the end. The fold is what models *choice* — either side can stop
//! when continuing would lose material, so each entry is the better of
//! "capture" and "stand pat".
//!
//! X-rays fall out of recomputing the attacker set against a shrinking
//! occupancy: removing the rook in front of another rook reveals the one behind.

use crate::board::Position;
use crate::types::{Move, PieceType, Square};

/// Piece values for exchange arithmetic only.
///
/// Separate from [`crate::eval::params`] on purpose: these want to be plain and
/// stable, not tuned. The king is priced so that a king capture never looks
/// profitable — a king cannot legally be captured, but it can appear in an
/// attacker set.
pub const SEE_VALUE: [i32; PieceType::COUNT] = [100, 320, 330, 500, 950, 20_000];

#[inline]
pub fn value_of(piece_type: PieceType) -> i32 {
    SEE_VALUE[piece_type.index()]
}

/// Material the side to move nets from the exchange on `mv.to()`, in
/// centipawns. Negative means the capture loses material.
///
/// `mv` must be a capture or a promotion; anything else scores 0.
pub fn see(pos: &Position, mv: Move) -> i32 {
    let to = mv.to();
    let from = mv.from();
    let us = pos.side_to_move();

    let Some(moving) = pos.piece_at(from) else {
        return 0;
    };

    // What the first capture wins, and what it leaves standing on the square.
    let mut gain = [0i32; 32];
    gain[0] = if mv.is_en_passant() {
        value_of(PieceType::Pawn)
    } else {
        pos.piece_at(to).map_or(0, |p| value_of(p.piece_type()))
    };
    let mut on_square = value_of(moving.piece_type());
    if let Some(promoted) = mv.promotion() {
        // The pawn is gone and the promoted piece stands there instead.
        gain[0] += value_of(promoted) - value_of(PieceType::Pawn);
        on_square = value_of(promoted);
    }

    let mut occupied = pos.occupied() ^ from.bb();
    if mv.is_en_passant() {
        let taken = to
            .offset(-us.pawn_push())
            .expect("en passant target is never on an edge rank");
        occupied ^= taken.bb();
    }

    let mut side = us.flip();
    let mut depth = 0usize;

    loop {
        depth += 1;
        // If the opponent recaptures, this is what they net.
        gain[depth] = on_square - gain[depth - 1];

        // Neither side can improve on standing pat, so the rest is irrelevant.
        if gain[depth].max(-gain[depth - 1]) < 0 {
            break;
        }
        if depth >= gain.len() - 1 {
            break;
        }

        // Masking by `occupied` drops attackers already spent, and recomputing
        // against the shrunken set is what reveals x-rays.
        let attackers = pos.attackers_to(to, occupied) & occupied;
        let Some((square, piece_type)) = least_valuable(pos, attackers, side) else {
            break;
        };
        occupied ^= square.bb();
        on_square = value_of(piece_type);
        side = side.flip();
    }

    // Fold back: at each step the mover takes the better of capturing and
    // walking away, which is a negamax over the swap list.
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -((-gain[depth - 1]).max(gain[depth]));
    }
    gain[0]
}

/// Cheapest attacker of `side` in `attackers`, with its kind.
fn least_valuable(
    pos: &Position,
    attackers: crate::bitboard::Bitboard,
    side: crate::types::Color,
) -> Option<(Square, PieceType)> {
    for piece_type in PieceType::ALL {
        let candidates = attackers & pos.pieces(side, piece_type);
        if let Some(square) = candidates.lsb() {
            return Some((square, piece_type));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;

    /// Find the legal move written in coordinate notation.
    fn find(pos: &Position, uci: &str) -> Move {
        generate_legal(pos)
            .iter()
            .copied()
            .find(|mv| mv.to_string() == uci)
            .unwrap_or_else(|| panic!("{uci} is not legal here"))
    }

    fn see_of(fen: &str, uci: &str) -> i32 {
        let pos: Position = fen.parse().expect("test fen is valid");
        see(&pos, find(&pos, uci))
    }

    #[test]
    fn an_undefended_capture_wins_the_piece() {
        assert_eq!(see_of("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5"), 100);
        assert_eq!(see_of("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"), 950);
    }

    #[test]
    fn a_defended_capture_costs_the_attacker() {
        // Rook takes a pawn defended by a pawn: win 100, lose 500.
        assert_eq!(see_of("4k3/8/4p3/3p4/8/8/8/3RK3 w - - 0 1", "d1d5"), -400);
        // Queen takes the same pawn: much worse.
        assert_eq!(see_of("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", "d1d5"), -850);
    }

    #[test]
    fn an_even_trade_is_zero() {
        // Rd1xd5, recaptured by the rook on d8.
        assert_eq!(see_of("3rk3/8/8/3r4/8/8/8/3RK3 w - - 0 1", "d1d5"), 0);
    }

    #[test]
    fn x_rays_are_counted() {
        // Three White rooks stacked on d1/d2/d3 against two Black rooks on
        // d7/d8, over a Black pawn on d5. Winning the pawn only works out if
        // each departing rook reveals the one behind it.
        let fen = "3rk3/3r4/8/3p4/8/3R4/3R4/3RK3 w - - 0 1";
        assert_eq!(see_of(fen, "d3d5"), 100);

        // One attacker short, and the same exchange loses two rooks for a rook
        // and a pawn.
        let short = "3rk3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1";
        assert_eq!(see_of(short, "d2d5"), -400);
    }

    #[test]
    fn the_cheapest_attacker_goes_first() {
        // Both a pawn and a queen can take on d5. Using the pawn wins a piece;
        // using the queen would not. SEE must assume the pawn recaptures.
        let fen = "4k3/8/2p5/3r4/8/8/8/3QK3 w - - 0 1";
        // Qxd5 is met by cxd5: win a rook, lose a queen.
        assert_eq!(see_of(fen, "d1d5"), 500 - 950);
    }

    #[test]
    fn en_passant_is_handled() {
        // The captured pawn is the one that double-pushed, not the one on the
        // target square -- so the target has to be read as empty and the pawn
        // removed from its own square.
        let lone = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
        assert_eq!(see_of(lone, "e5d6"), 100);

        // With c7 covering d6 it is an even trade, which only comes out right
        // if the recapture is counted.
        let defended = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        assert_eq!(see_of(defended, "e5d6"), 0);
    }

    #[test]
    fn promotion_counts_the_new_piece() {
        // a7a8q with nothing attacking a8: the pawn becomes a queen.
        let fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
        assert_eq!(
            see_of(fen, "a7a8q"),
            value_of(PieceType::Queen) - value_of(PieceType::Pawn)
        );
    }

    #[test]
    fn a_quiet_move_scores_nothing() {
        let pos: Position = "4k3/8/8/8/8/8/8/3RK3 w - - 0 1".parse().unwrap();
        assert_eq!(see(&pos, find(&pos, "d1d4")), 0);
    }
}
