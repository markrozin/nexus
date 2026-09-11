//! Core newtypes: [`Color`], [`PieceType`], [`Piece`], [`Square`], [`Move`],
//! [`CastlingRights`].
//!
//! Board mapping is little-endian rank-file: `A1 = 0`, `B1 = 1`, ..., `H8 = 63`.

use core::fmt;
use core::str::FromStr;

use crate::bitboard::Bitboard;

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// Side to move / piece owner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const COUNT: usize = 2;
    pub const ALL: [Color; Self::COUNT] = [Color::White, Color::Black];

    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Array index for this color. Paired with [`Color::from_index`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Color::White),
            1 => Some(Color::Black),
            _ => None,
        }
    }

    /// Square-index delta for one pawn advance in this color's direction.
    #[inline]
    pub const fn pawn_push(self) -> i8 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }

    /// FEN side-to-move character.
    #[inline]
    pub const fn to_char(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }
}

// ---------------------------------------------------------------------------
// PieceType
// ---------------------------------------------------------------------------

/// Piece kind, ignoring color.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const COUNT: usize = 6;
    pub const ALL: [PieceType; Self::COUNT] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        if i < Self::COUNT {
            Some(Self::ALL[i])
        } else {
            None
        }
    }

    /// Lowercase FEN character.
    #[inline]
    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Piece
// ---------------------------------------------------------------------------

/// A colored piece. The discriminant is `color * 6 + piece_type`, so a `Piece`
/// indexes a flat 12-entry table directly — the shape NNUE feature indexing
/// will want.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Piece {
    WhitePawn = 0,
    WhiteKnight = 1,
    WhiteBishop = 2,
    WhiteRook = 3,
    WhiteQueen = 4,
    WhiteKing = 5,
    BlackPawn = 6,
    BlackKnight = 7,
    BlackBishop = 8,
    BlackRook = 9,
    BlackQueen = 10,
    BlackKing = 11,
}

impl Piece {
    pub const COUNT: usize = 12;
    pub const ALL: [Piece; Self::COUNT] = [
        Piece::WhitePawn,
        Piece::WhiteKnight,
        Piece::WhiteBishop,
        Piece::WhiteRook,
        Piece::WhiteQueen,
        Piece::WhiteKing,
        Piece::BlackPawn,
        Piece::BlackKnight,
        Piece::BlackBishop,
        Piece::BlackRook,
        Piece::BlackQueen,
        Piece::BlackKing,
    ];

    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Self::ALL[color.index() * PieceType::COUNT + piece_type.index()]
    }

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(i: usize) -> Option<Self> {
        if i < Self::COUNT {
            Some(Self::ALL[i])
        } else {
            None
        }
    }

    #[inline]
    pub const fn color(self) -> Color {
        if self.index() < PieceType::COUNT {
            Color::White
        } else {
            Color::Black
        }
    }

    #[inline]
    pub const fn piece_type(self) -> PieceType {
        PieceType::ALL[self.index() % PieceType::COUNT]
    }

    /// FEN character: uppercase for white, lowercase for black.
    #[inline]
    pub const fn to_char(self) -> char {
        let c = self.piece_type().to_char();
        match self.color() {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        match PieceType::from_char(c) {
            Some(piece_type) => Some(Piece::new(color, piece_type)),
            None => None,
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// Square
// ---------------------------------------------------------------------------

const fn all_squares() -> [Square; Square::COUNT] {
    let mut squares = [Square(0); Square::COUNT];
    let mut i = 0;
    while i < Square::COUNT {
        squares[i] = Square(i as u8);
        i += 1;
    }
    squares
}

/// A board square. `A1 = 0`, `H8 = 63`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    pub const COUNT: usize = 64;

    /// Every square, in index order (A1, B1, ..., H8).
    pub const ALL: [Square; Self::COUNT] = all_squares();

    // Only the squares the rules code names directly; everything else comes
    // from `new`, `ALL`, or parsing.
    pub const A1: Self = Self(0);
    pub const C1: Self = Self(2);
    pub const D1: Self = Self(3);
    pub const E1: Self = Self(4);
    pub const F1: Self = Self(5);
    pub const G1: Self = Self(6);
    pub const H1: Self = Self(7);
    pub const A8: Self = Self(56);
    pub const C8: Self = Self(58);
    pub const D8: Self = Self(59);
    pub const E8: Self = Self(60);
    pub const F8: Self = Self(61);
    pub const G8: Self = Self(62);
    pub const H8: Self = Self(63);

    /// # Panics
    /// In debug builds, if `i >= 64`.
    #[inline]
    pub const fn from_index(i: u8) -> Self {
        debug_assert!(i < Self::COUNT as u8);
        Self(i)
    }

    /// `file` and `rank` are both `0..8` (file 0 is the A-file, rank 0 is the
    /// first rank).
    #[inline]
    pub const fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Self(rank * 8 + file)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// A bitboard containing only this square.
    #[inline]
    pub const fn bb(self) -> Bitboard {
        Bitboard::from_bits(1u64 << self.0)
    }

    /// Shift by a square-index delta, `None` if it leaves the board. Callers
    /// must still guard against file wraparound themselves.
    #[inline]
    pub const fn offset(self, delta: i8) -> Option<Self> {
        let i = self.0 as i16 + delta as i16;
        if i < 0 || i > 63 {
            None
        } else {
            Some(Self(i as u8))
        }
    }

    /// Parse coordinate notation, e.g. `"e4"`. Convenience wrapper over
    /// [`FromStr`] for callers that want an `Option`.
    #[inline]
    pub fn from_uci(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// The string was not a square in coordinate notation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParseSquareError;

impl fmt::Display for ParseSquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected a square in coordinate notation, e.g. \"e4\"")
    }
}

impl std::error::Error for ParseSquareError {}

impl FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(ParseSquareError);
        }
        let file = bytes[0].to_ascii_lowercase().wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file > 7 || rank > 7 {
            return Err(ParseSquareError);
        }
        Ok(Self::new(file, rank))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

/// A move, packed as `from | to << 6 | flag << 12`.
///
/// The flag nibble follows the conventional layout: bit 2 (`0b0100`) marks a
/// capture, bit 3 (`0b1000`) marks a promotion, and for promotions the low two
/// bits select knight/bishop/rook/queen.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Move(u16);

impl Move {
    pub const QUIET: u16 = 0b0000;
    pub const DOUBLE_PAWN: u16 = 0b0001;
    pub const KING_CASTLE: u16 = 0b0010;
    pub const QUEEN_CASTLE: u16 = 0b0011;
    pub const CAPTURE: u16 = 0b0100;
    pub const EN_PASSANT: u16 = 0b0101;
    pub const PROMO_N: u16 = 0b1000;
    pub const PROMO_B: u16 = 0b1001;
    pub const PROMO_R: u16 = 0b1010;
    pub const PROMO_Q: u16 = 0b1011;
    pub const PROMO_CAP_N: u16 = 0b1100;
    pub const PROMO_CAP_B: u16 = 0b1101;
    pub const PROMO_CAP_R: u16 = 0b1110;
    pub const PROMO_CAP_Q: u16 = 0b1111;

    /// Sentinel for "no move". Encodes a from == to == A1 quiet move, which is
    /// never generated.
    pub const NONE: Self = Self(0);

    #[inline]
    pub const fn new(from: Square, to: Square, flag: u16) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6) | (flag << 12))
    }

    #[inline]
    pub const fn from(self) -> Square {
        Square((self.0 & 0x3f) as u8)
    }

    #[inline]
    pub const fn to(self) -> Square {
        Square(((self.0 >> 6) & 0x3f) as u8)
    }

    #[inline]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline]
    pub const fn is_capture(self) -> bool {
        self.flag() & 0b0100 != 0
    }

    #[inline]
    pub const fn is_promotion(self) -> bool {
        self.flag() & 0b1000 != 0
    }

    #[inline]
    pub const fn is_en_passant(self) -> bool {
        self.flag() == Self::EN_PASSANT
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        matches!(self.flag(), Self::KING_CASTLE | Self::QUEEN_CASTLE)
    }

    #[inline]
    pub const fn is_double_pawn(self) -> bool {
        self.flag() == Self::DOUBLE_PAWN
    }

    /// The piece a promotion produces, or `None` for non-promotions.
    #[inline]
    pub const fn promotion(self) -> Option<PieceType> {
        if !self.is_promotion() {
            return None;
        }
        Some(match self.flag() & 0b11 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            _ => PieceType::Queen,
        })
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// The raw 16-bit packing. The transposition table stores a move in one
    /// `u16` field, so it needs the bits directly.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Rebuild from [`Move::bits`]. Any `u16` decodes to *some* move, so a
    /// value read back out of a hash table must still be checked for legality
    /// before it is played.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
}

impl fmt::Display for Move {
    /// Long algebraic notation as UCI expects it, e.g. `e2e4`, `e7e8q`.
    /// Castling is written as the king's own from/to (`e1g1`), not Chess960.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(p) = self.promotion() {
            write!(f, "{}", p.to_char())?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

// ---------------------------------------------------------------------------
// CastlingRights
// ---------------------------------------------------------------------------

/// Castling availability, as a 4-bit mask over a `u8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_KING: Self = Self(0b0001);
    pub const WHITE_QUEEN: Self = Self(0b0010);
    pub const BLACK_KING: Self = Self(0b0100);
    pub const BLACK_QUEEN: Self = Self(0b1000);
    pub const ALL: Self = Self(0b1111);

    /// Bits outside the low nibble are not castling rights.
    const MASK: u8 = 0b1111;

    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & Self::MASK)
    }

    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True only if every right in `other` is present.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    pub fn add(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clear the bits set in `other`.
    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    #[inline]
    pub const fn king_side(color: Color) -> Self {
        match color {
            Color::White => Self::WHITE_KING,
            Color::Black => Self::BLACK_KING,
        }
    }

    #[inline]
    pub const fn queen_side(color: Color) -> Self {
        match color {
            Color::White => Self::WHITE_QUEEN,
            Color::Black => Self::BLACK_QUEEN,
        }
    }

    /// Both rights belonging to `color`.
    #[inline]
    pub const fn both(color: Color) -> Self {
        match color {
            Color::White => Self(0b0011),
            Color::Black => Self(0b1100),
        }
    }

    /// The `(right, character)` pairs in FEN order.
    const FEN_ORDER: [(Self, char); 4] = [
        (Self::WHITE_KING, 'K'),
        (Self::WHITE_QUEEN, 'Q'),
        (Self::BLACK_KING, 'k'),
        (Self::BLACK_QUEEN, 'q'),
    ];

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'K' => Some(Self::WHITE_KING),
            'Q' => Some(Self::WHITE_QUEEN),
            'k' => Some(Self::BLACK_KING),
            'q' => Some(Self::BLACK_QUEEN),
            _ => None,
        }
    }
}

impl fmt::Display for CastlingRights {
    /// The FEN castling field, `-` when no rights remain.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("-");
        }
        for (right, c) in Self::FEN_ORDER {
            if self.contains(right) {
                write!(f, "{c}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_index_round_trips_through_color_and_type() {
        for (i, piece) in Piece::ALL.iter().enumerate() {
            assert_eq!(piece.index(), i);
            assert_eq!(Piece::from_index(i), Some(*piece));
            assert_eq!(Piece::new(piece.color(), piece.piece_type()), *piece);
            assert_eq!(Piece::from_char(piece.to_char()), Some(*piece));
        }
        assert_eq!(Piece::from_index(Piece::COUNT), None);
        assert_eq!(Piece::WhiteKing.color(), Color::White);
        assert_eq!(Piece::BlackPawn.color(), Color::Black);
        assert_eq!(Piece::BlackRook.piece_type(), PieceType::Rook);
    }

    #[test]
    fn color_and_piece_type_indices_round_trip() {
        for (i, color) in Color::ALL.iter().enumerate() {
            assert_eq!(color.index(), i);
            assert_eq!(Color::from_index(i), Some(*color));
        }
        for (i, kind) in PieceType::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i);
            assert_eq!(PieceType::from_index(i), Some(*kind));
        }
        assert_eq!(Color::from_index(2), None);
        assert_eq!(PieceType::from_index(6), None);
    }

    #[test]
    fn square_all_is_ordered_and_consistent() {
        assert_eq!(Square::ALL.len(), 64);
        for (i, sq) in Square::ALL.iter().enumerate() {
            assert_eq!(sq.index(), i);
            assert_eq!(Square::new(sq.file(), sq.rank()), *sq);
            assert_eq!(sq.bb().bits(), 1u64 << i);
            // Display and FromStr are inverses over the whole board.
            assert_eq!(sq.to_string().parse::<Square>(), Ok(*sq));
        }
        assert_eq!(Square::ALL[0], Square::A1);
        assert_eq!(Square::ALL[63], Square::H8);
    }

    #[test]
    fn square_parsing_is_strict_but_case_insensitive() {
        assert_eq!("e4".parse::<Square>(), Ok(Square::new(4, 3)));
        assert_eq!("E4".parse::<Square>(), Ok(Square::new(4, 3)));
        assert_eq!(Square::from_uci("a1"), Some(Square::A1));
        for bad in ["", "e", "e44", "i4", "e9", "4e", " e4"] {
            assert_eq!(bad.parse::<Square>(), Err(ParseSquareError), "{bad:?}");
        }
    }

    #[test]
    fn castling_rights_display_in_fen_order() {
        assert_eq!(CastlingRights::ALL.to_string(), "KQkq");
        assert_eq!(CastlingRights::NONE.to_string(), "-");
        let mut rights = CastlingRights::NONE;
        rights.add(CastlingRights::BLACK_QUEEN);
        rights.add(CastlingRights::WHITE_KING);
        assert_eq!(rights.to_string(), "Kq");
        rights.remove(CastlingRights::WHITE_KING);
        assert_eq!(rights.to_string(), "q");
        assert!(!rights.contains(CastlingRights::WHITE_KING));
        assert!(rights.contains(CastlingRights::BLACK_QUEEN));
    }

    #[test]
    fn castling_rights_bits_are_masked() {
        assert_eq!(CastlingRights::from_bits(0xff), CastlingRights::ALL);
        assert_eq!(CastlingRights::ALL.bits(), 0b1111);
        assert_eq!(
            CastlingRights::both(Color::White),
            CastlingRights::from_bits(0b0011)
        );
    }

    #[test]
    fn move_encoding_round_trips() {
        let mv = Move::new(Square::E1, Square::G1, Move::KING_CASTLE);
        assert_eq!(mv.from(), Square::E1);
        assert_eq!(mv.to(), Square::G1);
        assert!(mv.is_castle());
        assert!(!mv.is_capture());
        assert_eq!(mv.to_string(), "e1g1");

        let promo = Move::new(Square::new(4, 6), Square::new(5, 7), Move::PROMO_CAP_Q);
        assert!(promo.is_promotion() && promo.is_capture());
        assert_eq!(promo.promotion(), Some(PieceType::Queen));
        assert_eq!(promo.to_string(), "e7f8q");
        assert_eq!(Move::NONE.to_string(), "0000");
    }
}
