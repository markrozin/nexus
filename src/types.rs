//! Core newtypes: [`Color`], [`PieceType`], [`Piece`], [`Square`], [`Move`],
//! [`CastleRights`].
//!
//! Board mapping is little-endian rank-file: `A1 = 0`, `B1 = 1`, ..., `H8 = 63`.

use core::fmt;

use crate::bitboard::Bitboard;

/// Side to move / piece owner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Direction a pawn of this color advances, as a square-index delta.
    #[inline]
    pub const fn pawn_push(self) -> i8 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }
}

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
    pub const ALL: [PieceType; 6] = [
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

    /// Lowercase FEN character for this kind.
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

/// A colored piece.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceType,
}

impl Piece {
    #[inline]
    pub const fn new(color: Color, kind: PieceType) -> Self {
        Self { color, kind }
    }

    /// FEN character: uppercase for white, lowercase for black.
    pub const fn to_char(self) -> char {
        let c = self.kind.to_char();
        match self.color {
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
            Some(kind) => Some(Piece::new(color, kind)),
            None => None,
        }
    }
}

/// A board square. `A1 = 0`, `H8 = 63`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    pub const COUNT: usize = 64;

    // Only the squares the rules code names directly. Everything else is
    // built with `Square::new` or parsed from text.
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
        debug_assert!(i < 64);
        Self(i)
    }

    /// `file` and `rank` are both `0..8` (file 0 = A-file, rank 0 = first rank).
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

    /// Shift by a square-index delta, returning `None` if it leaves the board.
    /// Callers must still guard against file wraparound themselves.
    #[inline]
    pub const fn offset(self, delta: i8) -> Option<Self> {
        let i = self.0 as i16 + delta as i16;
        if i < 0 || i > 63 {
            None
        } else {
            Some(Self(i as u8))
        }
    }

    /// Parse coordinate notation, e.g. `"e4"`.
    pub fn from_uci(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 2 {
            return None;
        }
        let file = b[0].wrapping_sub(b'a');
        let rank = b[1].wrapping_sub(b'1');
        if file > 7 || rank > 7 {
            return None;
        }
        Some(Self::new(file, rank))
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

/// Castling availability, as a 4-bit mask.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CastleRights(u8);

impl CastleRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_KING: Self = Self(0b0001);
    pub const WHITE_QUEEN: Self = Self(0b0010);
    pub const BLACK_KING: Self = Self(0b0100);
    pub const BLACK_QUEEN: Self = Self(0b1000);
    pub const ALL: Self = Self(0b1111);

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
}

impl fmt::Display for CastleRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "-");
        }
        for (bit, c) in [
            (Self::WHITE_KING, 'K'),
            (Self::WHITE_QUEEN, 'Q'),
            (Self::BLACK_KING, 'k'),
            (Self::BLACK_QUEEN, 'q'),
        ] {
            if self.contains(bit) {
                write!(f, "{c}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for CastleRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
