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

use crate::bitboard::{king_attacks, knight_attacks, pawn_attacks, Bitboard};
use crate::board::Position;
use crate::magic::{bishop_attacks, queen_attacks, rook_attacks};
use crate::types::{CastlingRights, Color, Move, PieceType, Square};

/// Upper bound on legal moves in a position is 218; round up for headroom.
pub const MAX_MOVES: usize = 256;

pub type MoveList = ArrayVec<Move, MAX_MOVES>;

/// What to generate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenMode {
    /// Everything legal.
    All,
    /// Captures, en passant, and promotions only — what quiescence searches.
    /// Castling is excluded by construction, and quiet moves are skipped.
    Tactical,
}

/// Every legal move in `pos`. An empty list means checkmate or stalemate;
/// distinguish with [`Position::in_check`].
pub fn generate_legal(pos: &Position) -> MoveList {
    generate(pos, GenMode::All)
}

/// Legal captures, en passant, and promotions.
///
/// An empty list here means nothing tactical is available, *not* that the
/// position is terminal — quiescence relies on that distinction.
pub fn generate_tactical(pos: &Position) -> MoveList {
    generate(pos, GenMode::Tactical)
}

fn generate(pos: &Position, mode: GenMode) -> MoveList {
    let us = pos.side_to_move();
    let mut legal = MoveList::new();
    for mv in generate_pseudo_legal(pos, mode) {
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
pub fn generate_pseudo_legal(pos: &Position, mode: GenMode) -> MoveList {
    let mut list = MoveList::new();
    let us = pos.side_to_move();
    let own = pos.colored(us);
    let occupied = pos.occupied();

    // In tactical mode a piece may only land on an enemy square; `!own` for a
    // full generation is the same mask with the empty squares added back.
    let targets = match mode {
        GenMode::All => !own,
        GenMode::Tactical => pos.colored(us.flip()),
    };

    generate_pawn_moves(pos, us, mode, &mut list);

    for from in pos.pieces(us, PieceType::Knight) {
        push_targets(pos, from, knight_attacks(from) & targets, &mut list);
    }
    for from in pos.pieces(us, PieceType::Bishop) {
        push_targets(pos, from, bishop_attacks(from, occupied) & targets, &mut list);
    }
    for from in pos.pieces(us, PieceType::Rook) {
        push_targets(pos, from, rook_attacks(from, occupied) & targets, &mut list);
    }
    for from in pos.pieces(us, PieceType::Queen) {
        push_targets(pos, from, queen_attacks(from, occupied) & targets, &mut list);
    }
    for from in pos.pieces(us, PieceType::King) {
        push_targets(pos, from, king_attacks(from) & targets, &mut list);
    }

    if mode == GenMode::All {
        generate_castles(pos, us, &mut list);
    }
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

fn generate_pawn_moves(pos: &Position, us: Color, mode: GenMode, list: &mut MoveList) {
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

    // A push onto the last rank is tactical even though it captures nothing, so
    // tactical mode keeps the promoting subset of the quiet pushes.
    let single = pawns.forward(us) & empty;
    let pushes = match mode {
        GenMode::All => single,
        GenMode::Tactical => single & last_rank,
    };
    for to in pushes {
        let from = to.offset(-push).expect("pushed pawn came from the board");
        if last_rank.contains(to) {
            push_promotions(from, to, false, list);
        } else {
            list.push(Move::new(from, to, Move::QUIET));
        }
    }

    if mode == GenMode::All {
        for to in single.forward(us) & empty & double_rank {
            let from = to
                .offset(-push * 2)
                .expect("double-pushed pawn came from the board");
            list.push(Move::new(from, to, Move::DOUBLE_PAWN));
        }
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

    if pos.castling().contains(CastlingRights::king_side(us))
        && rooks.contains(king_rook)
        && empty_between(&[file(5), file(6)])
        && safe(&[file(5), file(6)])
    {
        list.push(Move::new(king_from, file(6), Move::KING_CASTLE));
    }

    if pos.castling().contains(CastlingRights::queen_side(us))
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

    /// `generate_tactical` must be exactly the capture-and-promotion subset of
    /// `generate_legal` — no extras, and nothing dropped. Quiescence relies on
    /// both halves.
    #[test]
    fn tactical_generation_is_the_capture_subset() {
        for fen in [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let pos: Position = fen.parse().expect("test fen is valid");
            let all = generate_legal(&pos);
            let tactical = generate_tactical(&pos);

            for mv in &tactical {
                assert!(all.contains(mv), "{mv} is not legal in {fen}");
                assert!(
                    mv.is_capture() || mv.is_promotion(),
                    "{mv} is quiet but was generated as tactical in {fen}"
                );
                assert!(!mv.is_castle(), "castling is never tactical");
            }
            let expected = all
                .iter()
                .filter(|mv| mv.is_capture() || mv.is_promotion())
                .count();
            assert_eq!(tactical.len(), expected, "count mismatch in {fen}");
        }
    }

    #[test]
    fn tactical_generation_keeps_en_passant_and_promotions() {
        // En passant is a capture even though the target square is empty.
        let ep: Position = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
            .parse()
            .unwrap();
        let uci: Vec<String> = generate_tactical(&ep).iter().map(|m| m.to_string()).collect();
        assert!(uci.contains(&"e5f6".to_string()), "{uci:?}");

        // A quiet push onto the last rank is tactical: all four promotions.
        let promo: Position = "8/P7/8/8/8/8/8/K6k w - - 0 1".parse().unwrap();
        let uci: Vec<String> = generate_tactical(&promo)
            .iter()
            .map(|m| m.to_string())
            .collect();
        for want in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            assert!(uci.contains(&want.to_string()), "{want} missing from {uci:?}");
        }
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

    /// Walk the tree, checking at every node that the mailbox and the bitboards
    /// still agree. This is what proves `make_move` maintains both halves of
    /// the representation, not just the half a given test happens to read.
    fn walk_checking_invariants(pos: &Position, depth: u32) {
        pos.assert_invariants();
        if depth == 0 {
            return;
        }
        for mv in generate_legal(pos) {
            walk_checking_invariants(&pos.make_move(mv), depth - 1);
        }
    }

    #[test]
    fn make_move_preserves_the_dual_representation() {
        walk_checking_invariants(&Position::startpos(), 3);
        for fen in [
            // Castling, captures, and a pinned position.
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Promotions on both sides.
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            // En passant and a rook endgame.
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let pos: Position = fen.parse().expect("test fen is valid");
            walk_checking_invariants(&pos, 2);
        }
    }

    #[test]
    fn perft_position_six() {
        check_perft(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2079, 89_890],
        );
    }

    /// The milestone 3 gate: all six reference positions at full depth, roughly
    /// 1.45 billion nodes. Ignored by default; run explicitly with
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "~1.45 billion nodes; run explicitly, in release"]
    fn perft_full_depth() {
        let cases: [(&str, u32, u64); 6] = [
            (START_FEN, 6, 119_060_324),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                5,
                193_690_690,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 7, 178_633_661),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                6,
                706_045_033,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                5,
                89_941_194,
            ),
            (
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                5,
                164_075_551,
            ),
        ];
        for (fen, depth, want) in cases {
            let pos: Position = fen.parse().expect("reference fen is valid");
            let start = std::time::Instant::now();
            let got = perft(&pos, depth);
            let mnps = got as f64 / start.elapsed().as_secs_f64() / 1e6;
            println!("perft({depth}) = {got:>12}  {mnps:6.1} Mnps  {fen}");
            assert_eq!(got, want, "perft({depth}) for {fen}");
        }
    }
}
