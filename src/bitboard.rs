//! [`Bitboard`] and attack generation.
//!
//! Little-endian rank-file: bit 0 is A1, bit 63 is H8. North is `<< 8`, east is
//! `<< 1`.
//!
//! Sliding attacks are computed by walking rays at runtime. That is correct but
//! slow; magic bitboards or PEXT replace this once search exists to benchmark
//! against.

use core::fmt;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr,
    ShrAssign, Sub,
};

use crate::types::{Color, Square};

/// A set of squares.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(!0);

    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);
    pub const RANK_1: Self = Self(0x0000_0000_0000_00ff);
    pub const RANK_4: Self = Self(0x0000_0000_ff00_0000);
    pub const RANK_5: Self = Self(0x0000_00ff_0000_0000);
    pub const RANK_8: Self = Self(0xff00_0000_0000_0000);

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.index()) != 0
    }

    /// Number of set squares. Named `popcount` rather than `count` so it does
    /// not shadow [`Iterator::count`], which this type also has.
    #[inline]
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.index();
    }

    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.index());
    }

    /// The least significant set square, or `None` if empty.
    #[inline]
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::from_index(self.0.trailing_zeros() as u8))
        }
    }

    /// Remove and return the least significant set square.
    #[inline]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        let sq = self.lsb()?;
        self.0 &= self.0 - 1;
        Some(sq)
    }

    /// The most significant set square, or `None` if empty.
    #[inline]
    pub const fn msb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::from_index(63 - self.0.leading_zeros() as u8))
        }
    }

    /// Shift every square one rank toward `color`'s promotion end.
    #[inline]
    pub const fn forward(self, color: Color) -> Self {
        match color {
            Color::White => Self(self.0 << 8),
            Color::Black => Self(self.0 >> 8),
        }
    }

    /// Shift east (toward the H-file), dropping anything that would wrap.
    #[inline]
    pub const fn east(self) -> Self {
        Self((self.0 & !Self::FILE_H.0) << 1)
    }

    /// Shift west (toward the A-file), dropping anything that would wrap.
    #[inline]
    pub const fn west(self) -> Self {
        Self((self.0 & !Self::FILE_A.0) >> 1)
    }
}

/// Iterating a `Bitboard` yields its squares from A1 upward, consuming it.
impl Iterator for Bitboard {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        self.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.popcount() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Bitboard {}

macro_rules! impl_bitop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl $trait for Bitboard {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Self(self.0 $op rhs.0)
            }
        }
        impl $assign_trait for Bitboard {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                self.0 = self.0 $op rhs.0;
            }
        }
    };
}

impl_bitop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
impl_bitop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
impl_bitop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bitboard {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

/// Set difference: `a - b` is the squares in `a` but not `b`.
impl Sub for Bitboard {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 & !rhs.0)
    }
}

/// Shifts move squares along the board index: `<< 8` is one rank north, `>> 1`
/// one file west. Neither masks files, so use [`Bitboard::east`] and
/// [`Bitboard::west`] where wraparound matters.
impl Shl<u32> for Bitboard {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: u32) -> Self {
        Self(self.0 << rhs)
    }
}

impl ShlAssign<u32> for Bitboard {
    #[inline]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 <<= rhs;
    }
}

impl Shr<u32> for Bitboard {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: u32) -> Self {
        Self(self.0 >> rhs)
    }
}

impl ShrAssign<u32> for Bitboard {
    #[inline]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 >>= rhs;
    }
}

impl fmt::Debug for Bitboard {
    /// Eight rank lines, rank 8 first, so the output reads like a board.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let c = if self.contains(Square::new(file, rank)) {
                    'x'
                } else {
                    '.'
                };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attack tables
// ---------------------------------------------------------------------------

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

const KING_DELTAS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

const ROOK_DIRS: [(i8, i8); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, -1), (-1, 1)];

/// Build a jump-piece attack table at compile time.
const fn step_table(deltas: &[(i8, i8); 8]) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let mut i = 0;
        while i < 8 {
            let f = file + deltas[i].0;
            let r = rank + deltas[i].1;
            if f >= 0 && f < 8 && r >= 0 && r < 8 {
                table[sq] |= 1u64 << (r * 8 + f);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}

/// Pawn capture targets. `white` selects the advance direction.
const fn pawn_table(white: bool) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let r = if white { rank + 1 } else { rank - 1 };
        if r >= 0 && r < 8 {
            if file > 0 {
                table[sq] |= 1u64 << (r * 8 + file - 1);
            }
            if file < 7 {
                table[sq] |= 1u64 << (r * 8 + file + 1);
            }
        }
        sq += 1;
    }
    table
}

static KNIGHT_ATTACKS: [u64; 64] = step_table(&KNIGHT_DELTAS);
static KING_ATTACKS: [u64; 64] = step_table(&KING_DELTAS);
static PAWN_ATTACKS: [[u64; 64]; 2] = [pawn_table(true), pawn_table(false)];

#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(KNIGHT_ATTACKS[sq.index()])
}

#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(KING_ATTACKS[sq.index()])
}

/// Squares a pawn of `color` standing on `sq` attacks.
#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    Bitboard(PAWN_ATTACKS[color.index()][sq.index()])
}

/// Walk rays from `sq`, stopping on (and including) the first blocker.
fn ray_attacks(sq: Square, occupied: Bitboard, dirs: &[(i8, i8); 4]) -> Bitboard {
    let mut out = Bitboard::EMPTY;
    for &(df, dr) in dirs {
        let mut f = sq.file() as i8;
        let mut r = sq.rank() as i8;
        loop {
            f += df;
            r += dr;
            if !(0..8).contains(&f) || !(0..8).contains(&r) {
                break;
            }
            let target = Square::new(f as u8, r as u8);
            out.set(target);
            if occupied.contains(target) {
                break;
            }
        }
    }
    out
}

#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &ROOK_DIRS)
}

#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &BISHOP_DIRS)
}

#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_bit_mapping() {
        assert_eq!(Square::A1.bb().bits(), 1);
        assert_eq!(Square::H8.bb().bits(), 1u64 << 63);
        assert_eq!(Square::from_uci("e4").unwrap().index(), 28);
    }

    #[test]
    fn knight_on_a1_has_two_moves() {
        assert_eq!(knight_attacks(Square::A1).popcount(), 2);
        assert_eq!(knight_attacks(Square::new(3, 3)).popcount(), 8);
    }

    #[test]
    fn rook_stops_at_blocker() {
        let occ = Square::new(0, 3).bb();
        let attacks = rook_attacks(Square::A1, occ);
        assert!(attacks.contains(Square::new(0, 3)));
        assert!(!attacks.contains(Square::new(0, 4)));
        assert!(attacks.contains(Square::H1));
    }

    #[test]
    fn iteration_yields_each_square_once() {
        let bb = Square::A1.bb() | Square::H8.bb();
        let squares: Vec<_> = bb.collect();
        assert_eq!(squares, vec![Square::A1, Square::H8]);
    }

    #[test]
    fn msb_and_lsb_find_the_extremes() {
        let bb = Square::A1.bb() | Square::new(3, 3).bb() | Square::H8.bb();
        assert_eq!(bb.lsb(), Some(Square::A1));
        assert_eq!(bb.msb(), Some(Square::H8));
        assert_eq!(bb.popcount(), 3);
        assert_eq!(Bitboard::EMPTY.lsb(), None);
        assert_eq!(Bitboard::EMPTY.msb(), None);
        for sq in Square::ALL {
            assert_eq!(sq.bb().lsb(), Some(sq));
            assert_eq!(sq.bb().msb(), Some(sq));
        }
    }

    #[test]
    fn shifts_move_along_the_board_index() {
        // One rank north is eight bits up.
        assert_eq!(Square::A1.bb() << 8, Square::new(0, 1).bb());
        assert_eq!(Square::H8.bb() >> 8, Square::new(7, 6).bb());
        // Shifts deliberately do not mask files: a2 shifted west lands on h1,
        // which is exactly what `west` exists to prevent.
        assert_eq!(Square::new(0, 1).bb() >> 1, Square::H1.bb());
        assert_eq!(Square::new(0, 1).bb().west(), Bitboard::EMPTY);

        let mut bb = Square::A1.bb();
        bb <<= 8;
        assert_eq!(bb, Square::new(0, 1).bb());
        bb >>= 8;
        assert_eq!(bb, Square::A1.bb());
    }
}
