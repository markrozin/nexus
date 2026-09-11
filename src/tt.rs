//! Transposition table.
//!
//! Positions recur constantly — 1.e4 e5 2.Nf3 and 1.Nf3 e5 2.e4 reach the same
//! board — so the search caches what it learned about a position under its
//! Zobrist key. A hit either cuts the node off outright or, far more often,
//! hands back the move that was best last time, which is worth a great deal to
//! move ordering on its own.
//!
//! # Concurrency
//!
//! The table is shared and accessed racily on purpose: locking it would cost
//! more than the occasional corrupted entry. Two things make that safe.
//!
//! First, every slot is a pair of [`AtomicU64`] read and written [`Relaxed`].
//! On x86-64 and AArch64 a relaxed 64-bit load or store is a plain move, so
//! this costs nothing over a raw field, and it means the module contains no
//! `unsafe` at all — no `UnsafeCell`, no hand-written `Sync`.
//!
//! Second, a slot is two words that can be written by different threads at
//! different moments, so a reader can see one half of one entry and one half of
//! another. The stored key is therefore `key ^ data`: a torn pair fails the
//! `stored ^ data == key` check and is treated as a miss. A wrong-but-consistent
//! pair is possible and harmless — the search validates the move it gets back.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::search::{MATE_IN_MAX_PLY, MAX_PLY};
use crate::types::Move;

/// Bytes per slot. Two `u64`s, so four slots to a 64-byte cache line.
pub const ENTRY_BYTES: usize = 16;

/// What a stored score means relative to the window it was found in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    /// No entry.
    None = 0,
    /// The search completed inside its window; the score is exact.
    Exact = 1,
    /// A beta cutoff: the true score is at least this.
    Lower = 2,
    /// Nothing beat alpha: the true score is at most this.
    Upper = 3,
}

impl Bound {
    const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            1 => Bound::Exact,
            2 => Bound::Lower,
            3 => Bound::Upper,
            _ => Bound::None,
        }
    }
}

/// What a probe returns.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: u8,
    pub bound: Bound,
}

#[derive(Default)]
struct Slot {
    /// `key ^ data`, so a torn pair fails verification.
    key: AtomicU64,
    data: AtomicU64,
}

fn pack(mv: Move, score: i16, eval: i16, depth: u8, bound: Bound, age: u8) -> u64 {
    (mv.bits() as u64)
        | ((score as u16 as u64) << 16)
        | ((depth as u64) << 32)
        | ((bound as u64) << 40)
        | (((age & 0x3f) as u64) << 42)
        | ((eval as u16 as u64) << 48)
}

fn unpack(data: u64) -> Hit {
    Hit {
        mv: Move::from_bits(data as u16),
        score: (data >> 16) as u16 as i16 as i32,
        eval: (data >> 48) as u16 as i16 as i32,
        depth: (data >> 32) as u8,
        bound: Bound::from_bits((data >> 40) as u8),
    }
}

fn age_of(data: u64) -> u8 {
    ((data >> 42) & 0x3f) as u8
}

pub struct TranspositionTable {
    slots: Box<[Slot]>,
    /// `slots.len() - 1`; the length is a power of two so indexing is a mask.
    mask: usize,
    /// Bumped once per `go`, so entries from earlier moves lose ties.
    age: AtomicU8,
}

impl TranspositionTable {
    /// Allocate a table of about `megabytes` MB, rounded down to a power of two
    /// number of slots. Always at least one slot.
    pub fn new(megabytes: usize) -> Self {
        let wanted = megabytes.max(1) * 1024 * 1024 / ENTRY_BYTES;
        let slots = wanted.next_power_of_two().min(wanted.max(1)).max(1);
        // `next_power_of_two` rounds up, which would overshoot the budget, so
        // step back down unless that lands on zero.
        let slots = if slots > wanted { slots / 2 } else { slots }.max(1);
        Self {
            slots: (0..slots).map(|_| Slot::default()).collect(),
            mask: slots - 1,
            age: AtomicU8::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Forget everything. Called for `ucinewgame`, where positions from the
    /// previous game are noise.
    pub fn clear(&self) {
        for slot in self.slots.iter() {
            slot.key.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }

    /// Start of a new search: entries from previous moves become stale.
    pub fn new_generation(&self) {
        self.age
            .store(self.age.load(Ordering::Relaxed).wrapping_add(1) & 0x3f, Ordering::Relaxed);
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        // The low bits pick the slot and the whole key verifies it, so an
        // unrelated position can only alias, never masquerade.
        (key as usize) & self.mask
    }

    /// Look `key` up. `ply` un-adjusts a stored mate distance.
    pub fn probe(&self, key: u64, ply: usize) -> Option<Hit> {
        let slot = &self.slots[self.index(key)];
        let stored = slot.key.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        if stored ^ data != key {
            return None;
        }
        let mut hit = unpack(data);
        if hit.bound == Bound::None {
            return None;
        }
        hit.score = score_from_tt(hit.score, ply);
        Some(hit)
    }

    /// Record what this node learned.
    ///
    /// Replacement is depth-preferred within a generation: a shallower result
    /// for the same position does not overwrite a deeper one, but anything from
    /// an earlier `go` does get replaced regardless of depth.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        mv: Move,
        score: i32,
        eval: i32,
        depth: u8,
        bound: Bound,
        ply: usize,
    ) {
        let slot = &self.slots[self.index(key)];
        let age = self.age.load(Ordering::Relaxed);

        let existing_data = slot.data.load(Ordering::Relaxed);
        let existing_key = slot.key.load(Ordering::Relaxed);
        let same_position = existing_key ^ existing_data == key;
        if same_position
            && age_of(existing_data) == age
            && unpack(existing_data).depth > depth
            && bound != Bound::Exact
        {
            return;
        }

        // Keep the previous move if this node did not produce one: a move from
        // a fail-low is still better than nothing for ordering.
        let mv = if mv.is_none() && same_position {
            unpack(existing_data).mv
        } else {
            mv
        };

        let data = pack(
            mv,
            score_to_tt(score, ply).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            depth,
            bound,
            age,
        );
        slot.key.store(key ^ data, Ordering::Relaxed);
        slot.data.store(data, Ordering::Relaxed);
    }

    /// Rough fill estimate in permille, over a sample of slots. This is what
    /// UCI `hashfull` wants.
    pub fn permille_full(&self) -> u32 {
        let sample = 1000.min(self.slots.len());
        let used = self.slots[..sample]
            .iter()
            .filter(|s| s.data.load(Ordering::Relaxed) != 0)
            .count();
        (used * 1000 / sample.max(1)) as u32
    }
}

/// A mate score means "mate in N from *this node*", so it has to be stored
/// relative to the node and re-based on the way out. Without this a mate found
/// at ply 8 would be reported as the same distance when the entry is reused at
/// ply 2.
#[inline]
pub fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX_PLY {
        score + ply as i32
    } else if score <= -MATE_IN_MAX_PLY {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
pub fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX_PLY {
        score - ply as i32
    } else if score <= -MATE_IN_MAX_PLY {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::MATE;
    use crate::types::Square;

    fn a_move() -> Move {
        Move::new(Square::E1, Square::G1, Move::KING_CASTLE)
    }

    #[test]
    fn sizing_is_a_power_of_two_within_budget() {
        for mb in [1usize, 2, 7, 16, 64] {
            let tt = TranspositionTable::new(mb);
            assert!(tt.len().is_power_of_two(), "{mb}MB gave {} slots", tt.len());
            assert!(
                tt.len() * ENTRY_BYTES <= mb * 1024 * 1024,
                "{mb}MB overshot its budget"
            );
        }
        // Zero is clamped rather than producing an unindexable table.
        assert!(TranspositionTable::new(0).len() >= 1);
    }

    #[test]
    fn round_trips_a_stored_entry() {
        let tt = TranspositionTable::new(1);
        let key = 0x0123_4567_89ab_cdef;
        tt.store(key, a_move(), 42, -17, 7, Bound::Exact, 0);

        let hit = tt.probe(key, 0).expect("just stored");
        assert_eq!(hit.mv, a_move());
        assert_eq!(hit.score, 42);
        assert_eq!(hit.eval, -17);
        assert_eq!(hit.depth, 7);
        assert_eq!(hit.bound, Bound::Exact);
    }

    #[test]
    fn a_different_key_is_a_miss() {
        let tt = TranspositionTable::new(1);
        tt.store(0xAAAA_AAAA_AAAA_AAAA, a_move(), 1, 0, 1, Bound::Exact, 0);
        // Same slot (low bits), different position: must not masquerade.
        let colliding = 0xAAAA_AAAA_AAAA_AAAAu64 ^ (1 << 40);
        assert!(tt.probe(colliding, 0).is_none());
        assert!(tt.probe(0x1234, 0).is_none());
    }

    #[test]
    fn negative_scores_and_moves_survive_packing() {
        let tt = TranspositionTable::new(1);
        for score in [-30_000i32, -1, 0, 1, 30_000] {
            let key = score as u64 ^ 0x9999;
            tt.store(key, a_move(), score, score, 3, Bound::Lower, 0);
            let hit = tt.probe(key, 0).unwrap();
            assert_eq!(hit.score, score, "score {score} did not survive");
            assert_eq!(hit.eval, score);
            assert_eq!(hit.bound, Bound::Lower);
        }
    }

    #[test]
    fn mate_scores_are_stored_relative_to_the_node() {
        // Mate in one found at ply 6 must read back as mate in one at ply 6,
        // and as a *different* distance if the entry is reused at another ply.
        let tt = TranspositionTable::new(1);
        let key = 0xFEED_FACE_CAFE_BEEF;
        let mate_at_ply_6 = MATE - 7;
        tt.store(key, a_move(), mate_at_ply_6, 0, 5, Bound::Exact, 6);

        assert_eq!(tt.probe(key, 6).unwrap().score, mate_at_ply_6);
        // Two plies shallower: the same mate is now two plies further away.
        assert_eq!(tt.probe(key, 4).unwrap().score, mate_at_ply_6 - 2);

        // Symmetric for being mated.
        tt.store(key, a_move(), -mate_at_ply_6, 0, 5, Bound::Exact, 6);
        assert_eq!(tt.probe(key, 6).unwrap().score, -mate_at_ply_6);
        assert_eq!(tt.probe(key, 4).unwrap().score, -(mate_at_ply_6 - 2));
    }

    #[test]
    fn ordinary_scores_are_not_ply_adjusted() {
        for ply in [0usize, 1, 50, MAX_PLY - 1] {
            assert_eq!(score_to_tt(123, ply), 123);
            assert_eq!(score_from_tt(123, ply), 123);
            assert_eq!(score_to_tt(-123, ply), -123);
        }
    }

    #[test]
    fn a_deeper_entry_is_not_replaced_by_a_shallower_one() {
        let tt = TranspositionTable::new(1);
        let key = 0x5555_5555_5555_5555;
        tt.store(key, a_move(), 100, 0, 10, Bound::Lower, 0);
        tt.store(key, Move::NONE, 200, 0, 2, Bound::Lower, 0);
        assert_eq!(tt.probe(key, 0).unwrap().depth, 10);
        assert_eq!(tt.probe(key, 0).unwrap().score, 100);

        // An exact result replaces regardless: it is strictly better news.
        tt.store(key, a_move(), 250, 0, 2, Bound::Exact, 0);
        assert_eq!(tt.probe(key, 0).unwrap().score, 250);
    }

    #[test]
    fn a_new_generation_lets_shallow_entries_replace_deep_ones() {
        let tt = TranspositionTable::new(1);
        let key = 0x7777_7777_7777_7777;
        tt.store(key, a_move(), 100, 0, 20, Bound::Lower, 0);
        tt.new_generation();
        tt.store(key, a_move(), 5, 0, 1, Bound::Lower, 0);
        assert_eq!(tt.probe(key, 0).unwrap().depth, 1, "stale entry should go");
    }

    #[test]
    fn clear_empties_the_table() {
        let tt = TranspositionTable::new(1);
        tt.store(0xABCD, a_move(), 1, 0, 1, Bound::Exact, 0);
        assert!(tt.probe(0xABCD, 0).is_some());
        tt.clear();
        assert!(tt.probe(0xABCD, 0).is_none());
        assert_eq!(tt.permille_full(), 0);
    }

    #[test]
    fn a_torn_pair_reads_as_a_miss() {
        // Simulate the interleaving the XOR guards against: one entry's key
        // beside another entry's data.
        let tt = TranspositionTable::new(1);
        let key_a = 0x1111_2222_3333_4444;
        tt.store(key_a, a_move(), 10, 0, 4, Bound::Exact, 0);
        let idx = tt.index(key_a);
        let stale_data = tt.slots[idx].data.load(Ordering::Relaxed);

        let key_b = key_a ^ 0xFFFF_0000_0000_0000;
        tt.store(key_b, a_move(), 20, 0, 4, Bound::Exact, 0);
        // Put back the earlier data under the newer key: a torn write.
        tt.slots[tt.index(key_b)]
            .data
            .store(stale_data, Ordering::Relaxed);

        assert!(tt.probe(key_b, 0).is_none(), "torn entry must not verify");
    }
}
