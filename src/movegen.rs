//! Move generation.
//!
//! [`generate_pseudo_legal`] produces every move that follows the movement
//! rules; [`generate_legal`] additionally filters out moves that leave the
//! mover's own king attacked. The filter is copy-make plus a king-attack test,
//! which is the simple correct approach — pin-aware generation is a later
//! optimization, and perft guards the swap.
//!
//! Move lists live in an [`ArrayVec`] on the stack: no heap allocation.

use arrayvec::ArrayVec;

use crate::bitboard::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
    Bitboard,
};
use crate::board::Position;
use crate::types::{CastleRights, Color, Move, PieceType, Square};

/// Upper bound on legal moves in a position is 218; round up for headroom.
pub const MAX_MOVES: usize = 256;

pub type MoveList = ArrayVec<Move, MAX_MOVES>;

/// Every legal move in `pos`. An empty list means checkmate or stalemate;
/// distinguish with [`Position::in_check`].
pub fn generate_legal(pos: &Position) -> MoveList {
    let us = pos.side_to_move();
    let mut legal = MoveList::new();
    for mv in generate_pseudo_legal(pos) {
        if !pos.make_move(mv).in_check(us) {
            legal.push(mv);
        }
    }
    legal
}

/// Every move that follows the movement rules, ignoring whether it leaves the
/// mover in check. Castling is fully validated here (empty path, not castling
/// out of, through, or into check) because those conditions are not expressible
/// as a king-attack test on the resulting position.
pub fn generate_pseudo_legal(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    let us = pos.side_to_move();
    let own = pos.colored(us);
    let occupied = pos.occupied();

    generate_pawn_moves(pos, us, &mut list);

    for from in pos.pieces(us, PieceType::Knight) {
        push_targets(pos, from, knight_attacks(from) - own, &mut list);
    }
    for from in pos.pieces(us, PieceType::Bishop) {
        push_targets(pos, from, bishop_attacks(from, occupied) - own, &mut list);
    }
    for from in pos.pieces(us, PieceType::Rook) {
        push_targets(pos, from, rook_attacks(from, occupied) - own, &mut list);
    }
    for from in pos.pieces(us, PieceType::Queen) {
        push_targets(pos, from, queen_attacks(from, occupied) - own, &mut list);
    }
    for from in pos.pieces(us, PieceType::King) {
        push_targets(pos, from, king_attacks(from) - own, &mut list);
    }

    generate_castles(pos, us, &mut list);
    list
}

/// Add quiet/capture moves from `from` to each square in `targets`.
fn push_targets(pos: &Position, from: Square, targets: Bitboard, list: &mut MoveList) {
    let them = pos.colored(pos.side_to_move().flip());
    for to in targets {
        let flag = if them.contains(to) {
            Move::CAPTURE
        } else {
            Move::QUIET
        };
        list.push(Move::new(from, to, flag));
    }
}

/// Add the four promotion choices for a pawn reaching the last rank.
fn push_promotions(from: Square, to: Square, capture: bool, list: &mut MoveList) {
    let flags = if capture {
        [
            Move::PROMO_CAP_Q,
            Move::PROMO_CAP_R,
            Move::PROMO_CAP_B,
            Move::PROMO_CAP_N,
        ]
    } else {
        [Move::PROMO_Q, Move::PROMO_R, Move::PROMO_B, Move::PROMO_N]
    };
    for flag in flags {
        list.push(Move::new(from, to, flag));
    }
}

fn generate_pawn_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let pawns = pos.pieces(us, PieceType::Pawn);
    if pawns.is_empty() {
        return;
    }
    let empty = !pos.occupied();
    let them = pos.colored(us.flip());
    let push = us.pawn_push();
    let (last_rank, double_rank) = match us {
        Color::White => (Bitboard::RANK_8, Bitboard::RANK_4),
        Color::Black => (Bitboard::RANK_1, Bitboard::RANK_5),
    };

    let single = pawns.forward(us) & empty;
    for to in single {
        let from = to.offset(-push).expect("pushed pawn came from the board");
        if last_rank.contains(to) {
            push_promotions(from, to, false, list);
        } else {
            list.push(Move::new(from, to, Move::QUIET));
        }
    }

    for to in single.forward(us) & empty & double_rank {
        let from = to
            .offset(-push * 2)
            .expect("double-pushed pawn came from the board");
        list.push(Move::new(from, to, Move::DOUBLE_PAWN));
    }

    // `east`/`west` are absolute board directions, so the origin offset differs
    // by color: subtracting the push direction undoes the rank change.
    for (targets, file_delta) in [
        (pawns.forward(us).east() & them, -1i8),
        (pawns.forward(us).west() & them, 1i8),
    ] {
        for to in targets {
            let from = to
                .offset(-push + file_delta)
                .expect("capturing pawn came from the board");
            if last_rank.contains(to) {
                push_promotions(from, to, true, list);
            } else {
                list.push(Move::new(from, to, Move::CAPTURE));
            }
        }
    }

    if let Some(ep) = pos.ep_square() {
        // Pawns that could capture onto `ep` are exactly those an enemy pawn on
        // `ep` would attack.
        for from in pawn_attacks(us.flip(), ep) & pawns {
            list.push(Move::new(from, ep, Move::EN_PASSANT));
        }
    }
}

fn generate_castles(pos: &Position, us: Color, list: &mut MoveList) {
    let occupied = pos.occupied();
    let them = us.flip();
    let (king_from, king_rook, queen_rook) = match us {
        Color::White => (Square::E1, Square::H1, Square::A1),
        Color::Black => (Square::E8, Square::H8, Square::A8),
    };

    // A GUI can hand us a FEN whose castling flags disagree with the board, so
    // check that the king and rook are actually home rather than trusting them.
    if pos.pieces(us, PieceType::King).lsb() != Some(king_from) {
        return;
    }
    let rooks = pos.pieces(us, PieceType::Rook);
    if pos.is_attacked(king_from, them) {
        return;
    }

    let file = |f: u8| Square::new(f, king_from.rank());
    let empty_between = |squares: &[Square]| squares.iter().all(|&s| !occupied.contains(s));
    let safe = |squares: &[Square]| squares.iter().all(|&s| !pos.is_attacked(s, them));

    if pos.castling().contains(CastleRights::king_side(us))
        && rooks.contains(king_rook)
        && empty_between(&[file(5), file(6)])
        && safe(&[file(5), file(6)])
    {
        list.push(Move::new(king_from, file(6), Move::KING_CASTLE));
    }

    if pos.castling().contains(CastleRights::queen_side(us))
        && rooks.contains(queen_rook)
        && empty_between(&[file(1), file(2), file(3)])
        && safe(&[file(2), file(3)])
    {
        list.push(Move::new(king_from, file(2), Move::QUEEN_CASTLE));
    }
}

/// Count leaf nodes at `depth`. The standard move-generation correctness test.
pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|&mv| perft(&pos.make_move(mv), depth - 1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::START_FEN;

    /// Reference node counts from the Chess Programming Wiki's perft results.
    fn check_perft(fen: &str, expected: &[u64]) {
        let pos = Position::from_fen(fen).expect("test fen is valid");
        for (i, &want) in expected.iter().enumerate() {
            let depth = i as u32 + 1;
            let got = perft(&pos, depth);
            assert_eq!(got, want, "perft({depth}) for {fen}");
        }
    }

    #[test]
    fn perft_startpos() {
        check_perft(START_FEN, &[20, 400, 8902, 197_281]);
    }

    #[test]
    fn perft_kiwipete() {
        check_perft(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2039, 97_862],
        );
    }

    #[test]
    fn perft_endgame_with_promotions_and_ep() {
        check_perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43_238]);
    }

    #[test]
    fn perft_position_four() {
        check_perft(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9467],
        );
    }

    #[test]
    fn perft_position_five() {
        check_perft(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1486, 62_379],
        );
    }

    #[test]
    fn checkmate_has_no_legal_moves() {
        // Fool's mate.
        let pos = Position::from_fen(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .unwrap();
        assert!(generate_legal(&pos).is_empty());
        assert!(pos.in_check(Color::White));
    }

    #[test]
    fn stalemate_has_no_legal_moves() {
        let pos = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert!(generate_legal(&pos).is_empty());
        assert!(!pos.in_check(Color::Black));
    }

    #[test]
    fn castling_is_generated_when_legal() {
        let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let uci: Vec<String> = generate_legal(&pos).iter().map(|m| m.to_string()).collect();
        assert!(uci.contains(&"e1g1".to_string()));
        assert!(uci.contains(&"e1c1".to_string()));
    }

    #[test]
    fn cannot_castle_through_check() {
        // Black rook on f8 covers f1, so the white king may not pass through it.
        let pos = Position::from_fen("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let uci: Vec<String> = generate_legal(&pos).iter().map(|m| m.to_string()).collect();
        assert!(!uci.contains(&"e1g1".to_string()));
        assert!(uci.contains(&"e1c1".to_string()));
    }
}
