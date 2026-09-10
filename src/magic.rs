//! Magic bitboards: constant-time sliding-piece attacks.
//!
//! A rook on d4 only cares about blockers on the d-file and rank 4, and not even
//! all of those — a piece on d8 stops the ray at d8, and an empty d8 also stops
//! it at d8, so the edge square is irrelevant. That leaves 10 to 12 relevant
//! squares for a rook and 5 to 9 for a bishop.
//!
//! Those relevant bits are scattered across the word, so a magic multiply
//! smears them into the high bits, where a shift makes a dense table index:
//!
//! ```text
//! index = ((occupancy & mask) * magic) >> (64 - bits)
//! ```
//!
//! There is no derivation for the magic constants — multiplication by a random
//! word is chaotic bit mixing, and you find working ones by trying candidates
//! until no two occupancies that need different attack sets collide. That search
//! lives behind the `magicgen` feature and ran once; its output is
//! [`crate::magic_constants`].
//!
//! Correctness is pinned to [`crate::bitboard::slow_rook_attacks`] and its
//! bishop counterpart, which walk rays one square at a time. See
//! `tests::magic_matches_the_slow_reference`.

use std::sync::LazyLock;

use crate::bitboard::{slow_bishop_attacks, slow_rook_attacks, Bitboard};
use crate::magic_constants::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::types::{PieceType, Square};

/// Everything one square needs for a lookup, in a single cache line.
#[derive(Clone, Copy, Debug)]
struct Magic {
    mask: Bitboard,
    magic: u64,
    /// `64 - relevant_bits`.
    shift: u32,
    /// Where this square's block starts in the shared table.
    offset: u32,
}

impl Magic {
    #[inline]
    fn index(&self, occupied: Bitboard) -> usize {
        let relevant = (occupied & self.mask).bits();
        self.offset as usize + (relevant.wrapping_mul(self.magic) >> self.shift) as usize
    }
}

// ---------------------------------------------------------------------------
// Relevant-occupancy masks
// ---------------------------------------------------------------------------

/// Squares whose occupancy changes a rook's attack set from `sq`.
///
/// Excludes `sq` itself and the last square of each ray: a blocker there and an
/// empty square there produce the same attacks.
pub const fn rook_mask(sq: Square) -> Bitboard {
    let file = (sq.index() & 7) as i32;
    let rank = (sq.index() >> 3) as i32;
    let mut mask = 0u64;

    let mut r = rank + 1;
    while r <= 6 {
        mask |= 1u64 << (r * 8 + file);
        r += 1;
    }
    let mut r = rank - 1;
    while r >= 1 {
        mask |= 1u64 << (r * 8 + file);
        r -= 1;
    }
    let mut f = file + 1;
    while f <= 6 {
        mask |= 1u64 << (rank * 8 + f);
        f += 1;
    }
    let mut f = file - 1;
    while f >= 1 {
        mask |= 1u64 << (rank * 8 + f);
        f -= 1;
    }
    Bitboard::from_bits(mask)
}

/// Squares whose occupancy changes a bishop's attack set from `sq`.
pub const fn bishop_mask(sq: Square) -> Bitboard {
    let file = (sq.index() & 7) as i32;
    let rank = (sq.index() >> 3) as i32;
    let mut mask = 0u64;

    let (mut r, mut f) = (rank + 1, file + 1);
    while r <= 6 && f <= 6 {
        mask |= 1u64 << (r * 8 + f);
        r += 1;
        f += 1;
    }
    let (mut r, mut f) = (rank + 1, file - 1);
    while r <= 6 && f >= 1 {
        mask |= 1u64 << (r * 8 + f);
        r += 1;
        f -= 1;
    }
    let (mut r, mut f) = (rank - 1, file + 1);
    while r >= 1 && f <= 6 {
        mask |= 1u64 << (r * 8 + f);
        r -= 1;
        f += 1;
    }
    let (mut r, mut f) = (rank - 1, file - 1);
    while r >= 1 && f >= 1 {
        mask |= 1u64 << (r * 8 + f);
        r -= 1;
        f -= 1;
    }
    Bitboard::from_bits(mask)
}

/// Walk every subset of `mask`, in Carry-Rippler order.
///
/// `(subset - mask) & mask` borrows through the gaps in the mask, so successive
/// values enumerate all `2^popcount` subsets and return to zero at the end.
pub fn subsets(mask: Bitboard) -> impl Iterator<Item = Bitboard> {
    let bits = mask.bits();
    let count = 1usize << mask.popcount();
    let mut subset = 0u64;
    (0..count).map(move |_| {
        let current = subset;
        subset = subset.wrapping_sub(bits) & bits;
        Bitboard::from_bits(current)
    })
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// Fixed-shift ("plain") magics: each square gets its own `2^relevant_bits`
/// block. Rooks total 102,400 entries (800 KB) and bishops 5,248 (41 KB).
/// Fitting the tables tighter by allowing collisions between occupancies that
/// share an attack set would shrink that; not worth the complexity until there
/// is a search to measure it against.
struct Sliding {
    rook_magics: [Magic; Square::COUNT],
    bishop_magics: [Magic; Square::COUNT],
    rook_table: Box<[Bitboard]>,
    bishop_table: Box<[Bitboard]>,
}

/// Built once on first use. The tables are ~840 KB, so they are heap-allocated
/// at init rather than baked into the binary; nothing here allocates afterwards.
static SLIDING: LazyLock<Sliding> = LazyLock::new(Sliding::build);

impl Sliding {
    fn build() -> Self {
        let (rook_magics, rook_table) = Self::build_one(false, &ROOK_MAGICS);
        let (bishop_magics, bishop_table) = Self::build_one(true, &BISHOP_MAGICS);
        Self {
            rook_magics,
            bishop_magics,
            rook_table,
            bishop_table,
        }
    }

    fn build_one(
        is_bishop: bool,
        magics: &[u64; Square::COUNT],
    ) -> ([Magic; Square::COUNT], Box<[Bitboard]>) {
        let mut entries = [Magic {
            mask: Bitboard::EMPTY,
            magic: 0,
            shift: 0,
            offset: 0,
        }; Square::COUNT];

        // Lay out each square's block back to back.
        let mut offset = 0u32;
        for sq in Square::ALL {
            let mask = if is_bishop {
                bishop_mask(sq)
            } else {
                rook_mask(sq)
            };
            let bits = mask.popcount();
            entries[sq.index()] = Magic {
                mask,
                magic: magics[sq.index()],
                shift: 64 - bits,
                offset,
            };
            offset += 1 << bits;
        }

        let mut table = vec![Bitboard::EMPTY; offset as usize];
        for sq in Square::ALL {
            let entry = entries[sq.index()];
            assert_ne!(
                entry.magic, 0,
                "magic constant for {sq} is zero; regenerate src/magic_constants.rs"
            );
            for occupied in subsets(entry.mask) {
                let attacks = reference(is_bishop, sq, occupied);
                let slot = &mut table[entry.index(occupied)];
                // A collision is fine only when both occupancies want the same
                // answer. Anything else means the constant is wrong.
                assert!(
                    *slot == Bitboard::EMPTY || *slot == attacks,
                    "magic for {sq} collides destructively"
                );
                *slot = attacks;
            }
        }
        (entries, table.into_boxed_slice())
    }
}

#[inline]
fn reference(is_bishop: bool, sq: Square, occupied: Bitboard) -> Bitboard {
    if is_bishop {
        slow_bishop_attacks(sq, occupied)
    } else {
        slow_rook_attacks(sq, occupied)
    }
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let tables = &*SLIDING;
    tables.rook_table[tables.rook_magics[sq.index()].index(occupied)]
}

#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let tables = &*SLIDING;
    tables.bishop_table[tables.bishop_magics[sq.index()].index(occupied)]
}

#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

/// Attacks for any piece type. Pawns are color-dependent and so are not covered
/// here; use [`crate::bitboard::pawn_attacks`].
///
/// # Panics
/// If `pt` is [`PieceType::Pawn`].
#[inline]
pub fn attacks(pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
    use crate::bitboard::{king_attacks, knight_attacks};
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occupied),
        PieceType::Rook => rook_attacks(sq, occupied),
        PieceType::Queen => queen_attacks(sq, occupied),
        PieceType::King => king_attacks(sq),
        PieceType::Pawn => panic!("pawn attacks depend on color; use pawn_attacks"),
    }
}

/// Force the tables to be built. Call from engine startup so the first `go` does
/// not pay for it mid-search.
pub fn init() {
    LazyLock::force(&SLIDING);
}

// ---------------------------------------------------------------------------
// Magic search (generation only)
// ---------------------------------------------------------------------------

/// Search for a magic constant for `sq`.
///
/// Behind the `magicgen` feature because it runs once, offline: seconds of
/// searching is an unacceptable startup delay for an engine a GUI is waiting on,
/// so the results are pasted into `magic_constants.rs` and shipped. Kept in tree
/// because changing the mask scheme means regenerating.
#[cfg(feature = "magicgen")]
pub fn find_magic(sq: Square, is_bishop: bool, rng: &mut crate::rng::Rng) -> u64 {
    let mask = if is_bishop {
        bishop_mask(sq)
    } else {
        rook_mask(sq)
    };
    let bits = mask.popcount();
    let shift = 64 - bits;
    let size = 1usize << bits;

    // Every blocker arrangement, with the answer it must produce.
    let occupancies: Vec<Bitboard> = subsets(mask).collect();
    let answers: Vec<Bitboard> = occupancies
        .iter()
        .map(|&occ| reference(is_bishop, sq, occ))
        .collect();

    let mut table = vec![Bitboard::EMPTY; size];
    // Generation stamps let us reuse the buffer instead of clearing it.
    let mut stamped = vec![0u32; size];
    let mut generation = 0u32;

    loop {
        // Sparse candidates spread bits better; ANDing three draws biases toward
        // roughly a dozen set bits.
        let magic = rng.next_u64() & rng.next_u64() & rng.next_u64();

        // Cheap reject: a magic that does not populate the high byte cannot
        // spread the mask across the index either.
        if (mask.bits().wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }

        generation += 1;
        let mut ok = true;
        for (i, &occ) in occupancies.iter().enumerate() {
            let idx = (occ.bits().wrapping_mul(magic) >> shift) as usize;
            if stamped[idx] != generation {
                stamped[idx] = generation;
                table[idx] = answers[i];
            } else if table[idx] != answers[i] {
                ok = false; // destructive collision
                break;
            }
        }
        if ok {
            return magic;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn masks_exclude_the_square_and_the_ray_ends() {
        // A rook in the middle sees 10 relevant squares, in a corner 12.
        let d4 = Square::from_uci("d4").unwrap();
        assert_eq!(rook_mask(d4).popcount(), 10);
        assert_eq!(rook_mask(Square::A1).popcount(), 12);
        assert_eq!(rook_mask(Square::H8).popcount(), 12);
        // Never the square itself, never an edge of its own rays.
        assert!(!rook_mask(d4).contains(d4));
        assert!(!rook_mask(d4).contains(Square::from_uci("d8").unwrap()));
        assert!(!rook_mask(d4).contains(Square::from_uci("d1").unwrap()));
        assert!(!rook_mask(d4).contains(Square::from_uci("a4").unwrap()));
        assert!(rook_mask(d4).contains(Square::from_uci("d7").unwrap()));

        assert_eq!(bishop_mask(d4).popcount(), 9);
        assert_eq!(bishop_mask(Square::A1).popcount(), 6);
        assert!(!bishop_mask(d4).contains(Square::from_uci("a7").unwrap()));
        assert!(bishop_mask(d4).contains(Square::from_uci("b6").unwrap()));
    }

    #[test]
    fn carry_rippler_enumerates_every_subset_once() {
        let mask = rook_mask(Square::from_uci("d4").unwrap());
        let found: Vec<Bitboard> = subsets(mask).collect();
        assert_eq!(found.len(), 1 << mask.popcount());
        let mut sorted: Vec<u64> = found.iter().map(|b| b.bits()).collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), found.len(), "subsets must be distinct");
        assert!(found.iter().all(|s| (*s - mask).is_empty()));
        assert!(found.contains(&Bitboard::EMPTY));
        assert!(found.contains(&mask));
    }

    /// The gate for this milestone: the table lookup must agree with the
    /// ray walker on every square, over a wide spread of occupancies.
    #[test]
    fn magic_matches_the_slow_reference() {
        let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
        for sq in Square::ALL {
            // Exhaustive over the occupancies that can actually matter.
            for occ in subsets(rook_mask(sq)) {
                assert_eq!(rook_attacks(sq, occ), slow_rook_attacks(sq, occ), "rook {sq}");
            }
            for occ in subsets(bishop_mask(sq)) {
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slow_bishop_attacks(sq, occ),
                    "bishop {sq}"
                );
            }
            // Plus random full-board occupancies, which include irrelevant bits
            // the mask must discard.
            for _ in 0..2000 {
                let occ = Bitboard::from_bits(rng.next_u64() & rng.next_u64());
                assert_eq!(rook_attacks(sq, occ), slow_rook_attacks(sq, occ), "rook {sq}");
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slow_bishop_attacks(sq, occ),
                    "bishop {sq}"
                );
                assert_eq!(
                    queen_attacks(sq, occ),
                    slow_rook_attacks(sq, occ) | slow_bishop_attacks(sq, occ),
                    "queen {sq}"
                );
            }
        }
    }

    #[test]
    fn attacks_dispatches_by_piece_type() {
        use crate::bitboard::{king_attacks, knight_attacks};
        let d4 = Square::from_uci("d4").unwrap();
        let occ = Bitboard::from_bits(0x0000_1000_0010_0000);
        assert_eq!(attacks(PieceType::Knight, d4, occ), knight_attacks(d4));
        assert_eq!(attacks(PieceType::King, d4, occ), king_attacks(d4));
        assert_eq!(attacks(PieceType::Rook, d4, occ), rook_attacks(d4, occ));
        assert_eq!(attacks(PieceType::Bishop, d4, occ), bishop_attacks(d4, occ));
        assert_eq!(attacks(PieceType::Queen, d4, occ), queen_attacks(d4, occ));
    }

    #[test]
    #[should_panic(expected = "pawn attacks depend on color")]
    fn attacks_rejects_pawns() {
        attacks(PieceType::Pawn, Square::A1, Bitboard::EMPTY);
    }
}
