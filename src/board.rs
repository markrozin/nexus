//! [`Position`]: the board state, FEN I/O, and move application.
//!
//! `Position` is `Copy` and small, so the engine uses copy-make: applying a
//! move produces a new `Position` and there is no unmake path to keep in sync.
//! If profiling later shows the copy dominating, this is where make/unmake
//! would go.

use core::fmt;

use crate::bitboard::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks, Bitboard,
};
use crate::types::{CastleRights, Color, Move, Piece, PieceType, Square};

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A FEN string that could not be parsed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FenError(String);

impl FenError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl std::error::Error for FenError {}

/// A complete chess position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    by_color: [Bitboard; 2],
    by_type: [Bitboard; PieceType::COUNT],
    side_to_move: Color,
    castling: CastleRights,
    ep_square: Option<Square>,
    halfmove_clock: u16,
    fullmove_number: u16,
}

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Position {
    /// A board with no pieces on it. White to move, no castling rights.
    pub const fn empty() -> Self {
        Self {
            by_color: [Bitboard::EMPTY; 2],
            by_type: [Bitboard::EMPTY; PieceType::COUNT],
            side_to_move: Color::White,
            castling: CastleRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(START_FEN).expect("START_FEN is valid")
    }

    // -- accessors ----------------------------------------------------------

    #[inline]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline]
    pub const fn castling(&self) -> CastleRights {
        self.castling
    }

    #[inline]
    pub const fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    #[inline]
    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    #[inline]
    pub const fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }

    /// Every piece belonging to `color`.
    #[inline]
    pub fn colored(&self, color: Color) -> Bitboard {
        self.by_color[color.index()]
    }

    /// Every `kind` belonging to `color`.
    #[inline]
    pub fn pieces(&self, color: Color, kind: PieceType) -> Bitboard {
        self.by_color[color.index()] & self.by_type[kind.index()]
    }

    /// Every `kind`, both colors.
    #[inline]
    pub fn by_type(&self, kind: PieceType) -> Bitboard {
        self.by_type[kind.index()]
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        let mask = sq.bb();
        if (self.occupied() & mask).is_empty() {
            return None;
        }
        let color = if (self.by_color[0] & mask).any() {
            Color::White
        } else {
            Color::Black
        };
        for kind in PieceType::ALL {
            if (self.by_type[kind.index()] & mask).any() {
                return Some(Piece::new(color, kind));
            }
        }
        None
    }

    /// `None` only for malformed positions; real games always have both kings.
    #[inline]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        self.pieces(color, PieceType::King).lsb()
    }

    // -- queries ------------------------------------------------------------

    /// Is `sq` attacked by any piece of color `by`?
    pub fn is_attacked(&self, sq: Square, by: Color) -> bool {
        // A pawn of color `by` attacks `sq` exactly when a pawn of the opposite
        // color standing on `sq` would attack that pawn's square.
        if (pawn_attacks(by.flip(), sq) & self.pieces(by, PieceType::Pawn)).any() {
            return true;
        }
        if (knight_attacks(sq) & self.pieces(by, PieceType::Knight)).any() {
            return true;
        }
        if (king_attacks(sq) & self.pieces(by, PieceType::King)).any() {
            return true;
        }

        let occ = self.occupied();
        let queens = self.pieces(by, PieceType::Queen);
        if (bishop_attacks(sq, occ) & (self.pieces(by, PieceType::Bishop) | queens)).any() {
            return true;
        }
        if (rook_attacks(sq, occ) & (self.pieces(by, PieceType::Rook) | queens)).any() {
            return true;
        }
        false
    }

    /// Is `color`'s king currently attacked?
    pub fn in_check(&self, color: Color) -> bool {
        match self.king_square(color) {
            Some(king) => self.is_attacked(king, color.flip()),
            None => false,
        }
    }

    // -- mutation -----------------------------------------------------------

    pub fn put(&mut self, sq: Square, piece: Piece) {
        self.by_color[piece.color.index()].set(sq);
        self.by_type[piece.kind.index()].set(sq);
    }

    fn remove(&mut self, sq: Square) {
        let mask = !sq.bb();
        self.by_color[0] &= mask;
        self.by_color[1] &= mask;
        for bb in &mut self.by_type {
            *bb &= mask;
        }
    }

    /// Move whatever stands on `from` to `to`. `to` must already be empty.
    fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self
            .piece_at(from)
            .expect("move_piece called with an empty origin square");
        self.remove(from);
        self.put(to, piece);
    }

    /// Apply `mv`, returning the resulting position. `mv` must be pseudo-legal
    /// for `self`; legality (leaving your own king in check) is not checked
    /// here — that is [`crate::movegen`]'s job.
    pub fn make_move(&self, mv: Move) -> Self {
        let mut next = *self;
        let us = self.side_to_move;
        let from = mv.from();
        let to = mv.to();
        let moving = self
            .piece_at(from)
            .expect("make_move called with an empty origin square")
            .kind;

        next.ep_square = None;
        next.halfmove_clock = next.halfmove_clock.saturating_add(1);

        if mv.is_en_passant() {
            let captured = to
                .offset(-us.pawn_push())
                .expect("en passant target is never on an edge rank");
            next.remove(captured);
            next.halfmove_clock = 0;
        } else if mv.is_capture() {
            next.remove(to);
            next.halfmove_clock = 0;
        }

        next.move_piece(from, to);

        if moving == PieceType::Pawn {
            next.halfmove_clock = 0;
        }

        if let Some(promo) = mv.promotion() {
            next.remove(to);
            next.put(to, Piece::new(us, promo));
        }

        if mv.is_double_pawn() {
            next.ep_square = from.offset(us.pawn_push());
        }

        if mv.is_castle() {
            let (rook_from, rook_to) = match (us, mv.flag()) {
                (Color::White, Move::KING_CASTLE) => (Square::H1, Square::F1),
                (Color::White, _) => (Square::A1, Square::D1),
                (Color::Black, Move::KING_CASTLE) => (Square::H8, Square::F8),
                (Color::Black, _) => (Square::A8, Square::D8),
            };
            next.move_piece(rook_from, rook_to);
        }

        // A king or rook leaving home, or a rook being captured on its home
        // square, both cost the corresponding right.
        next.castling.remove(rights_touched(from));
        next.castling.remove(rights_touched(to));

        next.side_to_move = us.flip();
        if us == Color::Black {
            next.fullmove_number += 1;
        }
        next
    }

    // -- FEN ----------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let mut fields = fen.split_whitespace();
        let placement = fields
            .next()
            .ok_or_else(|| FenError::new("empty string"))?;

        let mut pos = Position::empty();
        let mut rank: i8 = 7;
        let mut file: u8 = 0;
        for c in placement.chars() {
            match c {
                '/' => {
                    if file != 8 {
                        return Err(FenError::new(format!("rank {} has {file} files", rank + 1)));
                    }
                    rank -= 1;
                    file = 0;
                    if rank < 0 {
                        return Err(FenError::new("more than 8 ranks"));
                    }
                }
                '1'..='8' => {
                    file += c as u8 - b'0';
                    if file > 8 {
                        return Err(FenError::new(format!("rank {} overflows", rank + 1)));
                    }
                }
                _ => {
                    let piece =
                        Piece::from_char(c).ok_or_else(|| FenError::new(format!("bad piece {c:?}")))?;
                    if file > 7 {
                        return Err(FenError::new(format!("rank {} overflows", rank + 1)));
                    }
                    pos.put(Square::new(file, rank as u8), piece);
                    file += 1;
                }
            }
        }
        if rank != 0 || file != 8 {
            return Err(FenError::new("placement does not cover 8 ranks"));
        }

        pos.side_to_move = match fields.next().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(FenError::new(format!("bad side to move {other:?}"))),
        };

        let castling = fields.next().unwrap_or("-");
        if castling != "-" {
            for c in castling.chars() {
                match c {
                    'K' => pos.castling.add(CastleRights::WHITE_KING),
                    'Q' => pos.castling.add(CastleRights::WHITE_QUEEN),
                    'k' => pos.castling.add(CastleRights::BLACK_KING),
                    'q' => pos.castling.add(CastleRights::BLACK_QUEEN),
                    other => {
                        return Err(FenError::new(format!("bad castling flag {other:?}")));
                    }
                }
            }
        }

        let ep = fields.next().unwrap_or("-");
        pos.ep_square = if ep == "-" {
            None
        } else {
            Some(Square::from_uci(ep).ok_or_else(|| FenError::new(format!("bad ep square {ep:?}")))?)
        };

        pos.halfmove_clock = fields.next().unwrap_or("0").parse().unwrap_or(0);
        pos.fullmove_number = fields.next().unwrap_or("1").parse().unwrap_or(1);

        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut out = String::with_capacity(80);
        for rank in (0..8).rev() {
            let mut gap = 0;
            for file in 0..8 {
                match self.piece_at(Square::new(file, rank)) {
                    Some(piece) => {
                        if gap > 0 {
                            out.push_str(&gap.to_string());
                            gap = 0;
                        }
                        out.push(piece.to_char());
                    }
                    None => gap += 1,
                }
            }
            if gap > 0 {
                out.push_str(&gap.to_string());
            }
            if rank > 0 {
                out.push('/');
            }
        }
        out.push(' ');
        out.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });
        out.push_str(&format!(" {} ", self.castling));
        match self.ep_square {
            Some(sq) => out.push_str(&sq.to_string()),
            None => out.push('-'),
        }
        out.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));
        out
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let c = self
                    .piece_at(Square::new(file, rank))
                    .map_or('.', Piece::to_char);
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  a b c d e f g h")?;
        write!(f, "{}", self.to_fen())
    }
}

/// Castling rights lost when a piece moves from, or onto, `sq`.
fn rights_touched(sq: Square) -> CastleRights {
    match sq {
        Square::A1 => CastleRights::WHITE_QUEEN,
        Square::H1 => CastleRights::WHITE_KING,
        Square::E1 => CastleRights::both(Color::White),
        Square::A8 => CastleRights::BLACK_QUEEN,
        Square::H8 => CastleRights::BLACK_KING,
        Square::E8 => CastleRights::both(Color::Black),
        _ => CastleRights::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_roundtrips_through_fen() {
        assert_eq!(Position::startpos().to_fen(), START_FEN);
    }

    #[test]
    fn startpos_has_the_right_pieces() {
        let pos = Position::startpos();
        assert_eq!(pos.occupied().popcount(), 32);
        assert_eq!(pos.pieces(Color::White, PieceType::Pawn).popcount(), 8);
        assert_eq!(pos.king_square(Color::White), Some(Square::E1));
        assert_eq!(pos.king_square(Color::Black), Some(Square::E8));
        assert_eq!(pos.castling(), CastleRights::ALL);
        assert!(!pos.in_check(Color::White));
    }

    #[test]
    fn rejects_malformed_fen() {
        assert!(Position::from_fen("").is_err());
        assert!(Position::from_fen("8/8/8/8/8/8/8 w - - 0 1").is_err());
        assert!(Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w - - 0 1").is_err());
    }

    #[test]
    fn double_pawn_push_sets_ep_square() {
        let pos = Position::startpos();
        let e2 = Square::from_uci("e2").unwrap();
        let e4 = Square::from_uci("e4").unwrap();
        let next = pos.make_move(Move::new(e2, e4, Move::DOUBLE_PAWN));
        assert_eq!(next.ep_square(), Square::from_uci("e3"));
        assert_eq!(next.side_to_move(), Color::Black);
        assert_eq!(next.halfmove_clock(), 0);
    }

    #[test]
    fn castling_moves_the_rook_and_clears_rights() {
        let pos =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid test fen");
        let next = pos.make_move(Move::new(Square::E1, Square::G1, Move::KING_CASTLE));
        assert_eq!(
            next.piece_at(Square::G1),
            Some(Piece::new(Color::White, PieceType::King))
        );
        assert_eq!(
            next.piece_at(Square::F1),
            Some(Piece::new(Color::White, PieceType::Rook))
        );
        assert_eq!(next.piece_at(Square::H1), None);
        assert_eq!(next.castling(), CastleRights::both(Color::Black));
    }

    #[test]
    fn capturing_a_rook_on_its_home_square_clears_that_right() {
        let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        // Ra1xa8 takes the rook that black's queenside right depends on.
        let next = pos.make_move(Move::new(Square::A1, Square::A8, Move::CAPTURE));
        assert!(!next.castling().contains(CastleRights::BLACK_QUEEN));
        assert!(!next.castling().contains(CastleRights::WHITE_QUEEN));
        assert!(next.castling().contains(CastleRights::BLACK_KING));
    }

    #[test]
    fn detects_check() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R b K - 0 1").unwrap();
        assert!(!pos.in_check(Color::Black));
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1").unwrap();
        assert!(pos.in_check(Color::Black));
    }
}
