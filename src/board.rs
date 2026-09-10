//! [`Position`]: the board state, FEN I/O, and move application.
//!
//! The board is represented twice over, and the two must always agree:
//!
//! - **Bitboards**, `[[Bitboard; 6]; 2]` indexed by color then piece type, plus
//!   a cached occupancy per color and a total. Answers "where are all the white
//!   rooks" in one load.
//! - **A mailbox**, `[Option<Piece>; 64]`. Answers "what is on e4" in one load,
//!   which the bitboards alone cannot do without scanning six boards.
//!
//! Keeping both is the usual trade: a little more work in `put`/`remove`, much
//! less everywhere else. [`Position::assert_invariants`] checks that they have
//! not drifted apart, and the tests call it after every mutation.
//!
//! `Position` is `Copy`, so the engine uses copy-make: applying a move produces
//! a new `Position` and there is no unmake path to keep in sync.

use core::fmt;
use core::str::FromStr;

use crate::bitboard::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks, Bitboard,
};
use crate::types::{CastlingRights, Color, Move, Piece, PieceType, Square};

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---------------------------------------------------------------------------
// FEN errors
// ---------------------------------------------------------------------------

/// Why a FEN string could not be parsed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FenError {
    /// The string was empty or contained no fields.
    MissingPlacement,
    /// The piece placement field described more than eight ranks.
    TooManyRanks,
    /// The piece placement field described fewer than eight ranks.
    NotEnoughRanks { found: u8 },
    /// A rank did not describe exactly eight files.
    RankWidth { rank: u8, width: usize },
    /// A placement character was neither a digit `1..=8` nor a piece letter.
    BadPiece(char),
    /// The side-to-move field was not `w` or `b`.
    BadSideToMove(String),
    /// The castling field held something other than `-` or `KQkq` letters.
    BadCastlingRights(String),
    /// The en passant field was not `-` or a square.
    BadEnPassant(String),
    /// The halfmove clock was not a number.
    BadHalfmoveClock(String),
    /// The fullmove number was not a number.
    BadFullmoveNumber(String),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: ")?;
        match self {
            FenError::MissingPlacement => f.write_str("no piece placement field"),
            FenError::TooManyRanks => f.write_str("more than 8 ranks"),
            FenError::NotEnoughRanks { found } => {
                write!(f, "placement describes {found} ranks, expected 8")
            }
            FenError::RankWidth { rank, width } => {
                write!(f, "rank {rank} describes {width} files, expected 8")
            }
            FenError::BadPiece(c) => write!(f, "unrecognized piece character {c:?}"),
            FenError::BadSideToMove(s) => write!(f, "side to move {s:?}, expected \"w\" or \"b\""),
            FenError::BadCastlingRights(s) => write!(f, "castling rights {s:?}"),
            FenError::BadEnPassant(s) => write!(f, "en passant square {s:?}"),
            FenError::BadHalfmoveClock(s) => write!(f, "halfmove clock {s:?}"),
            FenError::BadFullmoveNumber(s) => write!(f, "fullmove number {s:?}"),
        }
    }
}

impl std::error::Error for FenError {}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// A complete chess position.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Indexed `[color][piece_type]`.
    piece_bb: [[Bitboard; PieceType::COUNT]; Color::COUNT],
    /// Cached union of `piece_bb[color]`.
    color_bb: [Bitboard; Color::COUNT],
    /// Cached union of both `color_bb` entries.
    occupancy: Bitboard,
    /// What stands on each square, indexed by [`Square::index`].
    mailbox: [Option<Piece>; Square::COUNT],
    side_to_move: Color,
    castling: CastlingRights,
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
            piece_bb: [[Bitboard::EMPTY; PieceType::COUNT]; Color::COUNT],
            color_bb: [Bitboard::EMPTY; Color::COUNT],
            occupancy: Bitboard::EMPTY,
            mailbox: [None; Square::COUNT],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(START_FEN).expect("START_FEN is valid")
    }

    /// Parse a FEN string. Wrapper over the [`FromStr`] impl.
    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        fen.parse()
    }

    /// Serialize to FEN. Wrapper over the [`fmt::Display`] impl.
    pub fn to_fen(&self) -> String {
        self.to_string()
    }

    // -- accessors ----------------------------------------------------------

    #[inline]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline]
    pub const fn castling(&self) -> CastlingRights {
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

    /// Every occupied square.
    #[inline]
    pub const fn occupied(&self) -> Bitboard {
        self.occupancy
    }

    /// Every piece belonging to `color`.
    #[inline]
    pub fn colored(&self, color: Color) -> Bitboard {
        self.color_bb[color.index()]
    }

    /// Every `kind` belonging to `color`.
    #[inline]
    pub fn pieces(&self, color: Color, kind: PieceType) -> Bitboard {
        self.piece_bb[color.index()][kind.index()]
    }

    /// Every `kind`, both colors.
    #[inline]
    pub fn by_type(&self, kind: PieceType) -> Bitboard {
        self.piece_bb[0][kind.index()] | self.piece_bb[1][kind.index()]
    }

    /// What stands on `sq`. A single mailbox lookup.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index()]
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

    /// Is the king of `color` currently attacked?
    pub fn in_check(&self, color: Color) -> bool {
        match self.king_square(color) {
            Some(king) => self.is_attacked(king, color.flip()),
            None => false,
        }
    }

    // -- mutation -----------------------------------------------------------
    //
    // These three are the only places the mailbox and the bitboards are
    // written. Keeping them in one place is what makes the invariant tractable.

    /// Place `piece` on an empty square.
    fn put(&mut self, sq: Square, piece: Piece) {
        debug_assert!(
            self.mailbox[sq.index()].is_none(),
            "put onto occupied square {sq}"
        );
        let mask = sq.bb();
        self.mailbox[sq.index()] = Some(piece);
        self.piece_bb[piece.color().index()][piece.piece_type().index()] |= mask;
        self.color_bb[piece.color().index()] |= mask;
        self.occupancy |= mask;
    }

    /// Clear `sq`, returning whatever stood there.
    fn remove(&mut self, sq: Square) -> Option<Piece> {
        let piece = self.mailbox[sq.index()].take()?;
        let mask = sq.bb();
        self.piece_bb[piece.color().index()][piece.piece_type().index()] ^= mask;
        self.color_bb[piece.color().index()] ^= mask;
        self.occupancy ^= mask;
        Some(piece)
    }

    /// Move whatever stands on `from` to `to`. `to` must already be empty.
    fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self
            .remove(from)
            .expect("move_piece called with an empty origin square");
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
            .piece_type();

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

    // -- invariants ---------------------------------------------------------

    /// Panic if the mailbox and the bitboards disagree.
    ///
    /// Debug builds only: in release this compiles to nothing, so it is safe to
    /// sprinkle at call sites. The tests call it after every mutation, which is
    /// what actually keeps the dual representation honest.
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        let mut from_mailbox = [Bitboard::EMPTY; Piece::COUNT];
        for sq in Square::ALL {
            match self.mailbox[sq.index()] {
                Some(piece) => from_mailbox[piece.index()].set(sq),
                None => assert!(
                    !self.occupancy.contains(sq),
                    "{sq} is empty in the mailbox but set in occupancy"
                ),
            }
        }

        let mut rebuilt_colors = [Bitboard::EMPTY; Color::COUNT];
        for color in Color::ALL {
            for kind in PieceType::ALL {
                let bb = self.piece_bb[color.index()][kind.index()];
                assert_eq!(
                    bb,
                    from_mailbox[Piece::new(color, kind).index()],
                    "{color:?} {kind:?} bitboard disagrees with the mailbox"
                );
                // Two piece bitboards must never claim the same square.
                assert!(
                    (rebuilt_colors[color.index()] & bb).is_empty(),
                    "{color:?} {kind:?} overlaps another {color:?} bitboard"
                );
                rebuilt_colors[color.index()] |= bb;
            }
        }

        assert!(
            (rebuilt_colors[0] & rebuilt_colors[1]).is_empty(),
            "a square is occupied by both colors"
        );
        for color in Color::ALL {
            assert_eq!(
                self.color_bb[color.index()],
                rebuilt_colors[color.index()],
                "cached {color:?} occupancy is stale"
            );
        }
        assert_eq!(
            self.occupancy,
            rebuilt_colors[0] | rebuilt_colors[1],
            "cached total occupancy is stale"
        );

        for color in Color::ALL {
            assert!(
                self.pieces(color, PieceType::King).popcount() <= 1,
                "{color:?} has more than one king"
            );
        }

        if let Some(ep) = self.ep_square {
            assert!(
                ep.rank() == 2 || ep.rank() == 5,
                "en passant square {ep} is not on the third or sixth rank"
            );
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn assert_invariants(&self) {}
}

// ---------------------------------------------------------------------------
// FEN parsing
// ---------------------------------------------------------------------------

impl FromStr for Position {
    type Err = FenError;

    fn from_str(fen: &str) -> Result<Self, Self::Err> {
        let mut fields = fen.split_whitespace();
        let placement = fields.next().ok_or(FenError::MissingPlacement)?;

        let mut pos = Position::empty();
        let mut rank: i8 = 7;
        let mut file: u8 = 0;
        for c in placement.chars() {
            match c {
                '/' => {
                    if file != 8 {
                        return Err(FenError::RankWidth {
                            rank: rank as u8 + 1,
                            width: file as usize,
                        });
                    }
                    rank -= 1;
                    file = 0;
                    if rank < 0 {
                        return Err(FenError::TooManyRanks);
                    }
                }
                '1'..='8' => {
                    file += c as u8 - b'0';
                    if file > 8 {
                        return Err(FenError::RankWidth {
                            rank: rank as u8 + 1,
                            width: file as usize,
                        });
                    }
                }
                _ => {
                    let piece = Piece::from_char(c).ok_or(FenError::BadPiece(c))?;
                    if file > 7 {
                        return Err(FenError::RankWidth {
                            rank: rank as u8 + 1,
                            width: file as usize + 1,
                        });
                    }
                    pos.put(Square::new(file, rank as u8), piece);
                    file += 1;
                }
            }
        }
        // Keep these apart: a short final rank and a placement that ran out of
        // ranks are different mistakes, and reporting the first for the second
        // names a rank that was actually fine.
        if file != 8 {
            return Err(FenError::RankWidth {
                rank: rank.max(0) as u8 + 1,
                width: file as usize,
            });
        }
        if rank != 0 {
            return Err(FenError::NotEnoughRanks {
                found: (8 - rank) as u8,
            });
        }

        pos.side_to_move = match fields.next().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(FenError::BadSideToMove(other.to_owned())),
        };

        let castling = fields.next().unwrap_or("-");
        if castling != "-" {
            for c in castling.chars() {
                let right = CastlingRights::from_char(c)
                    .ok_or_else(|| FenError::BadCastlingRights(castling.to_owned()))?;
                pos.castling.add(right);
            }
        }

        let ep = fields.next().unwrap_or("-");
        pos.ep_square = if ep == "-" {
            None
        } else {
            let sq: Square = ep
                .parse()
                .map_err(|_| FenError::BadEnPassant(ep.to_owned()))?;
            if sq.rank() != 2 && sq.rank() != 5 {
                return Err(FenError::BadEnPassant(ep.to_owned()));
            }
            Some(sq)
        };

        // Both counters are optional: plenty of tools emit four-field FENs.
        if let Some(field) = fields.next() {
            pos.halfmove_clock = field
                .parse()
                .map_err(|_| FenError::BadHalfmoveClock(field.to_owned()))?;
        }
        if let Some(field) = fields.next() {
            pos.fullmove_number = field
                .parse()
                .map_err(|_| FenError::BadFullmoveNumber(field.to_owned()))?;
        }

        Ok(pos)
    }
}

// ---------------------------------------------------------------------------
// FEN serialization and board rendering
// ---------------------------------------------------------------------------

impl fmt::Display for Position {
    /// The FEN for this position.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            let mut gap = 0;
            for file in 0..8 {
                match self.piece_at(Square::new(file, rank)) {
                    Some(piece) => {
                        if gap > 0 {
                            write!(f, "{gap}")?;
                            gap = 0;
                        }
                        write!(f, "{}", piece.to_char())?;
                    }
                    None => gap += 1,
                }
            }
            if gap > 0 {
                write!(f, "{gap}")?;
            }
            if rank > 0 {
                f.write_str("/")?;
            }
        }

        write!(f, " {} {} ", self.side_to_move.to_char(), self.castling)?;
        match self.ep_square {
            Some(sq) => write!(f, "{sq}")?,
            None => f.write_str("-")?,
        }
        write!(f, " {} {}", self.halfmove_clock, self.fullmove_number)
    }
}

impl fmt::Debug for Position {
    /// The board as ASCII art, with the FEN underneath. This is what shows up
    /// when a test assertion fails, so it favors being readable over being
    /// parseable.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "  +------------------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} |", rank + 1)?;
            for file in 0..8 {
                match self.piece_at(Square::new(file, rank)) {
                    Some(piece) => write!(f, " {} ", piece.to_char())?,
                    None => f.write_str(" . ")?,
                }
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +------------------------+")?;
        writeln!(f, "    a  b  c  d  e  f  g  h")?;
        writeln!(f, "{self}")
    }
}

/// Castling rights lost when a piece moves from, or onto, `sq`.
fn rights_touched(sq: Square) -> CastlingRights {
    match sq {
        Square::A1 => CastlingRights::WHITE_QUEEN,
        Square::H1 => CastlingRights::WHITE_KING,
        Square::E1 => CastlingRights::both(Color::White),
        Square::A8 => CastlingRights::BLACK_QUEEN,
        Square::H8 => CastlingRights::BLACK_KING,
        Square::E8 => CastlingRights::both(Color::Black),
        _ => CastlingRights::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of positions covering both en passant colors, full and partial
    /// castling rights, promotion material, and non-default move counters.
    const ROUND_TRIP_FENS: [&str; 8] = [
        START_FEN,
        // Kiwipete: all four castling rights, heavy piece traffic.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // En passant available to White on the sixth rank.
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        // En passant available to Black on the third rank.
        "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3",
        // Partial castling rights: White kingside and Black queenside only.
        "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 4 12",
        // Partial rights again, plus a non-trivial fullmove number.
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        // No rights at all, Black to move, large halfmove clock.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 13 42",
        // Promotion race, both sides one rank away.
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    ];

    #[test]
    fn fen_round_trips() {
        for fen in ROUND_TRIP_FENS {
            let pos: Position = fen.parse().unwrap_or_else(|e| panic!("{fen}: {e}"));
            pos.assert_invariants();
            assert_eq!(pos.to_fen(), fen, "round trip changed the FEN");
            // Parsing the serialized form must reach the same position.
            let again: Position = pos.to_fen().parse().expect("reparse");
            assert_eq!(again, pos);
        }
    }

    #[test]
    fn fen_fields_are_parsed_into_the_right_places() {
        let pos: Position = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 4 12".parse().unwrap();
        assert_eq!(pos.side_to_move(), Color::White);
        assert!(pos.castling().contains(CastlingRights::WHITE_KING));
        assert!(pos.castling().contains(CastlingRights::BLACK_QUEEN));
        assert!(!pos.castling().contains(CastlingRights::WHITE_QUEEN));
        assert!(!pos.castling().contains(CastlingRights::BLACK_KING));
        assert_eq!(pos.ep_square(), None);
        assert_eq!(pos.halfmove_clock(), 4);
        assert_eq!(pos.fullmove_number(), 12);
        assert_eq!(pos.piece_at(Square::E1), Some(Piece::WhiteKing));
        assert_eq!(pos.piece_at(Square::A8), Some(Piece::BlackRook));
        assert_eq!(pos.piece_at(Square::new(4, 4)), None);

        let ep: Position = "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3"
            .parse()
            .unwrap();
        assert_eq!(ep.ep_square(), Square::from_uci("e3"));
        assert_eq!(ep.side_to_move(), Color::Black);
    }

    #[test]
    fn startpos_bitboards_and_mailbox_agree() {
        let pos = Position::startpos();
        pos.assert_invariants();
        assert_eq!(pos.occupied().popcount(), 32);
        assert_eq!(pos.colored(Color::White).popcount(), 16);
        assert_eq!(pos.pieces(Color::White, PieceType::Pawn).popcount(), 8);
        assert_eq!(pos.by_type(PieceType::Knight).popcount(), 4);
        assert_eq!(pos.king_square(Color::White), Some(Square::E1));
        assert_eq!(pos.king_square(Color::Black), Some(Square::E8));
        assert_eq!(pos.castling(), CastlingRights::ALL);
        assert!(!pos.in_check(Color::White));

        // Every occupied square in the mailbox is in the matching bitboard.
        for sq in Square::ALL {
            match pos.piece_at(sq) {
                Some(piece) => {
                    assert!(pos.pieces(piece.color(), piece.piece_type()).contains(sq));
                    assert!(pos.colored(piece.color()).contains(sq));
                    assert!(pos.occupied().contains(sq));
                }
                None => assert!(!pos.occupied().contains(sq)),
            }
        }
    }

    #[test]
    fn empty_position_is_consistent() {
        let pos = Position::empty();
        pos.assert_invariants();
        assert!(pos.occupied().is_empty());
        assert_eq!(pos.king_square(Color::White), None);
        assert_eq!(pos.to_fen(), "8/8/8/8/8/8/8/8 w - - 0 1");
    }

    #[test]
    fn rejects_malformed_fen() {
        use FenError::*;
        let cases: [(&str, FenError); 6] = [
            ("", MissingPlacement),
            ("8/8/8/8/8/8/8 w - - 0 1", NotEnoughRanks { found: 7 }),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w - - 0 1",
                BadPiece('X'),
            ),
            (
                "8/8/8/8/8/8/8/8 x - - 0 1",
                BadSideToMove("x".to_owned()),
            ),
            (
                "8/8/8/8/8/8/8/8 w KQxq - 0 1",
                BadCastlingRights("KQxq".to_owned()),
            ),
            (
                "8/8/8/8/8/8/8/8 w - e4 0 1",
                BadEnPassant("e4".to_owned()),
            ),
        ];
        for (fen, want) in cases {
            assert_eq!(fen.parse::<Position>(), Err(want), "{fen:?}");
        }
        // A rank that overflows eight files.
        assert!(matches!(
            "rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1".parse::<Position>(),
            Err(RankWidth { .. })
        ));
        // Non-numeric counters.
        assert!(matches!(
            "8/8/8/8/8/8/8/8 w - - x 1".parse::<Position>(),
            Err(BadHalfmoveClock(_))
        ));
        assert!(matches!(
            "8/8/8/8/8/8/8/8 w - - 0 x".parse::<Position>(),
            Err(BadFullmoveNumber(_))
        ));
    }

    #[test]
    fn four_field_fens_get_default_counters() {
        let pos: Position = "8/8/8/8/8/8/8/K6k w - -".parse().unwrap();
        pos.assert_invariants();
        assert_eq!(pos.halfmove_clock(), 0);
        assert_eq!(pos.fullmove_number(), 1);
    }

    #[test]
    fn debug_renders_an_ascii_board() {
        let rendered = format!("{:?}", Position::startpos());
        let lines: Vec<&str> = rendered.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines[1], "8 | r  n  b  q  k  b  n  r |");
        assert_eq!(lines[4], "5 | .  .  .  .  .  .  .  . |");
        assert_eq!(lines[8], "1 | R  N  B  Q  K  B  N  R |");
        assert!(lines.last().unwrap().starts_with("rnbqkbnr/"));
    }

    // -- mutation keeps the two representations in step -----------------------

    /// Apply a move and re-check the invariant, returning the new position.
    fn step(pos: &Position, mv: Move) -> Position {
        let next = pos.make_move(mv);
        next.assert_invariants();
        next
    }

    #[test]
    fn double_pawn_push_sets_ep_square() {
        let e2 = Square::from_uci("e2").unwrap();
        let e4 = Square::from_uci("e4").unwrap();
        let next = step(&Position::startpos(), Move::new(e2, e4, Move::DOUBLE_PAWN));
        assert_eq!(next.ep_square(), Square::from_uci("e3"));
        assert_eq!(next.side_to_move(), Color::Black);
        assert_eq!(next.halfmove_clock(), 0);
        assert_eq!(next.piece_at(e2), None);
        assert_eq!(next.piece_at(e4), Some(Piece::WhitePawn));
    }

    #[test]
    fn castling_moves_the_rook_and_clears_rights() {
        let pos: Position = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        let next = step(&pos, Move::new(Square::E1, Square::G1, Move::KING_CASTLE));
        assert_eq!(next.piece_at(Square::G1), Some(Piece::WhiteKing));
        assert_eq!(next.piece_at(Square::F1), Some(Piece::WhiteRook));
        assert_eq!(next.piece_at(Square::H1), None);
        assert_eq!(next.piece_at(Square::E1), None);
        assert_eq!(next.castling(), CastlingRights::both(Color::Black));
    }

    #[test]
    fn capturing_a_rook_on_its_home_square_clears_that_right() {
        let pos: Position = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        let next = step(&pos, Move::new(Square::A1, Square::A8, Move::CAPTURE));
        assert!(!next.castling().contains(CastlingRights::BLACK_QUEEN));
        assert!(!next.castling().contains(CastlingRights::WHITE_QUEEN));
        assert!(next.castling().contains(CastlingRights::BLACK_KING));
        assert_eq!(next.piece_at(Square::A8), Some(Piece::WhiteRook));
        assert_eq!(next.colored(Color::Black).popcount(), 2);
    }

    #[test]
    fn en_passant_capture_removes_the_right_pawn() {
        // Black just played d7d5; White captures with exd6.
        let pos: Position = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
            .parse()
            .unwrap();
        let e5 = Square::from_uci("e5").unwrap();
        let d6 = Square::from_uci("d6").unwrap();
        let d5 = Square::from_uci("d5").unwrap();
        let next = step(&pos, Move::new(e5, d6, Move::EN_PASSANT));
        assert_eq!(next.piece_at(d6), Some(Piece::WhitePawn));
        assert_eq!(next.piece_at(d5), None, "the captured pawn must be gone");
        assert_eq!(next.piece_at(e5), None);
        assert_eq!(next.pieces(Color::Black, PieceType::Pawn).popcount(), 7);
        assert_eq!(next.ep_square(), None);
    }

    #[test]
    fn promotion_replaces_the_pawn() {
        let pos: Position = "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1".parse().unwrap();
        let g2 = Square::from_uci("g2").unwrap();
        let h1 = Square::H1;
        let next = step(&pos, Move::new(g2, h1, Move::PROMO_CAP_Q));
        assert_eq!(next.piece_at(h1), Some(Piece::BlackQueen));
        assert_eq!(next.piece_at(g2), None);
        assert_eq!(next.pieces(Color::Black, PieceType::Pawn).popcount(), 2);
        assert_eq!(next.pieces(Color::Black, PieceType::Queen).popcount(), 1);
        assert_eq!(next.pieces(Color::White, PieceType::Knight).popcount(), 1);
        assert_eq!(next.fullmove_number(), 2, "black moved, so the count ticks");
    }

    #[test]
    fn halfmove_clock_counts_quiet_moves_and_resets_on_pawns_and_captures() {
        let pos: Position = "4k3/8/8/8/8/8/4P3/4K3 w - - 7 30".parse().unwrap();
        let quiet = step(
            &pos,
            Move::new(Square::E1, Square::D1, Move::QUIET),
        );
        assert_eq!(quiet.halfmove_clock(), 8);
        assert_eq!(quiet.fullmove_number(), 30);

        let e2 = Square::from_uci("e2").unwrap();
        let e3 = Square::from_uci("e3").unwrap();
        let pawn = step(&pos, Move::new(e2, e3, Move::QUIET));
        assert_eq!(pawn.halfmove_clock(), 0);
    }
}
