//! Handcrafted evaluation: material plus tapered piece-square tables.
//!
//! Returns centipawns from the side-to-move's perspective, per the negamax
//! convention in `CLAUDE.md`.
//!
//! Tapering: a knight on d5 is worth more in a queens-on middlegame than in a
//! king-and-pawn ending, and a king belongs behind pawns in one and in the
//! centre in the other. So every term has a midgame and an endgame value, and
//! the score interpolates between them on a phase counter derived from the
//! remaining non-pawn material.
//!
//! Every constant lives in [`params`] so a Texel tuner can mutate them later.
//! They are hand-set placeholders, not tuned values: expect them to move once
//! there is a tuner and an SPRT pipeline.

use crate::board::Position;
use crate::types::{Color, PieceType};

/// Piece-square tables and material values.
///
/// Tables are written rank 8 first, so they read like a board from White's
/// side. [`table_index`] does the flip.
#[rustfmt::skip]
pub mod params {
    /// Midgame material values, indexed by [`super::PieceType::index`].
    pub const MG_VALUE: [i32; 6] = [100, 320, 330, 500, 950, 0];
    /// Endgame material values.
    pub const EG_VALUE: [i32; 6] = [120, 330, 350, 550, 1000, 0];

    /// Phase weight per piece. Pawns and kings do not count.
    pub const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];
    /// Sum of `PHASE` over a full starting array: 4*1 + 4*1 + 4*2 + 2*4.
    pub const PHASE_TOTAL: i32 = 24;

    /// Push pawns toward promotion, claim the centre, and discourage moving the
    /// pawns in front of a castled king.
    pub const MG_PAWN: [i32; 64] = [
          0,   0,   0,   0,   0,   0,   0,   0,
         50,  50,  50,  50,  50,  50,  50,  50,
         10,  10,  20,  30,  30,  20,  10,  10,
          5,   5,  10,  25,  25,  10,   5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          5,  10,  10, -20, -20,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ];

    /// In the ending only the rank matters: a passed pawn is a passed pawn.
    pub const EG_PAWN: [i32; 64] = [
          0,   0,   0,   0,   0,   0,   0,   0,
         90,  90,  90,  90,  90,  90,  90,  90,
         55,  55,  55,  55,  55,  55,  55,  55,
         30,  30,  30,  30,  30,  30,  30,  30,
         15,  15,  15,  15,  15,  15,  15,  15,
          5,   5,   5,   5,   5,   5,   5,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
    ];

    /// Knights want the centre and hate the rim.
    pub const MG_KNIGHT: [i32; 64] = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ];

    pub const EG_KNIGHT: [i32; 64] = MG_KNIGHT;

    pub const MG_BISHOP: [i32; 64] = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ];

    pub const EG_BISHOP: [i32; 64] = MG_BISHOP;

    /// Rooks belong on the seventh and on central files.
    pub const MG_ROOK: [i32; 64] = [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10,  10,  10,  10,  10,   5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          0,   0,   0,   5,   5,   0,   0,   0,
    ];

    pub const EG_ROOK: [i32; 64] = MG_ROOK;

    pub const MG_QUEEN: [i32; 64] = [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ];

    pub const EG_QUEEN: [i32; 64] = MG_QUEEN;

    /// Midgame: stay tucked behind the castled pawns.
    pub const MG_KING: [i32; 64] = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20,
    ];

    /// Endgame: the king is a fighting piece and belongs in the centre.
    ///
    /// This doubles as a mop-up term. With overwhelming material the winning
    /// side is drawn toward the centre and the losing king is pushed to the
    /// edge, which is where it gets mated — without it a search this shallow
    /// shuffles in won endings instead of finishing them.
    pub const EG_KING: [i32; 64] = [
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10,   0,   0, -10, -20, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -30,   0,   0,   0,   0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50,
    ];

    /// Indexed by [`super::PieceType::index`].
    pub const MG_TABLE: [[i32; 64]; 6] =
        [MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING];
    pub const EG_TABLE: [[i32; 64]; 6] =
        [EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING];
}

/// Map a square to its piece-square-table slot.
///
/// Tables are written rank 8 first, so a White piece indexes the mirror of its
/// square and a Black piece indexes it directly — which is also exactly the
/// vertical flip that lets both colors share one table.
#[inline]
const fn table_index(color: Color, sq_index: usize) -> usize {
    match color {
        Color::White => sq_index ^ 56,
        Color::Black => sq_index,
    }
}

/// Remaining non-pawn material, `0..=24`. Clamped, since promotions can push a
/// position past a full starting array.
pub fn phase(pos: &Position) -> i32 {
    let mut phase = 0;
    for kind in PieceType::ALL {
        phase += params::PHASE[kind.index()] * pos.by_type(kind).popcount() as i32;
    }
    phase.min(params::PHASE_TOTAL)
}

/// Static evaluation in centipawns, from the side-to-move's perspective.
pub fn evaluate(pos: &Position) -> i32 {
    let mut mg = 0i32;
    let mut eg = 0i32;

    for color in Color::ALL {
        let sign = if color == pos.side_to_move() { 1 } else { -1 };
        for kind in PieceType::ALL {
            let (mg_value, eg_value) = (params::MG_VALUE[kind.index()], params::EG_VALUE[kind.index()]);
            let (mg_table, eg_table) = (&params::MG_TABLE[kind.index()], &params::EG_TABLE[kind.index()]);
            for sq in pos.pieces(color, kind) {
                let idx = table_index(color, sq.index());
                mg += sign * (mg_value + mg_table[idx]);
                eg += sign * (eg_value + eg_table[idx]);
            }
        }
    }

    let phase = phase(pos);
    (mg * phase + eg * (params::PHASE_TOTAL - phase)) / params::PHASE_TOTAL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn startpos_is_balanced() {
        assert_eq!(evaluate(&Position::startpos()), 0);
        assert_eq!(phase(&Position::startpos()), params::PHASE_TOTAL);
    }

    #[test]
    fn evaluation_is_symmetric_under_color_flip() {
        // The same structure with colors and ranks swapped must score the same
        // for whoever is to move. This catches a sign or table-orientation bug.
        let pairs = [
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ),
            (
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
            ),
            (
                "r3k2r/pppq1ppp/2n2n2/8/8/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
                "r3k2r/pppq1ppp/2n2n2/8/8/2N2N2/PPPQ1PPP/R3K2R b KQkq - 0 1",
            ),
        ];
        for (a, b) in pairs {
            let pa: Position = a.parse().unwrap();
            let pb: Position = b.parse().unwrap();
            assert_eq!(evaluate(&pa), evaluate(&pb), "{a} vs {b}");
        }
    }

    #[test]
    fn material_advantage_dominates() {
        // White is a whole queen up, and it is White to move.
        let pos: Position = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1".parse().unwrap();
        assert!(evaluate(&pos) > 800, "got {}", evaluate(&pos));
        // Same position, Black to move: the sign flips.
        let flipped: Position = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1".parse().unwrap();
        assert_eq!(evaluate(&pos), -evaluate(&flipped));
    }

    #[test]
    fn tables_are_read_the_right_way_up() {
        // A white pawn one step from promotion must score far above one at home.
        let advanced: Position = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let home: Position = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1".parse().unwrap();
        assert!(
            evaluate(&advanced) > evaluate(&home) + 40,
            "advanced {} vs home {}",
            evaluate(&advanced),
            evaluate(&home)
        );

        // And the same for Black, mirrored.
        let advanced_b: Position = "4k3/8/8/8/8/8/p7/4K3 b - - 0 1".parse().unwrap();
        let home_b: Position = "4k3/p7/8/8/8/8/8/4K3 b - - 0 1".parse().unwrap();
        assert!(evaluate(&advanced_b) > evaluate(&home_b) + 40);
    }

    #[test]
    fn phase_falls_as_pieces_come_off() {
        let full = Position::startpos();
        let bare: Position = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let middling: Position = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        assert_eq!(phase(&full), 24);
        assert_eq!(phase(&bare), 0);
        assert_eq!(phase(&middling), 8, "four rooks");
    }

    #[test]
    fn king_centralization_only_matters_in_the_ending() {
        // With no other material the endgame table dominates: centre beats edge.
        let centre: Position = "8/8/8/3K4/8/8/8/7k w - - 0 1".parse().unwrap();
        let corner: Position = "8/8/8/8/8/8/8/K6k w - - 0 1".parse().unwrap();
        assert!(evaluate(&centre) > evaluate(&corner));
        assert_eq!(table_index(Color::White, Square::A1.index()), 56);
        assert_eq!(table_index(Color::Black, Square::A8.index()), 56);
    }
}
