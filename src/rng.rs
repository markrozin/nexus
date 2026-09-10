//! A small self-contained PRNG.
//!
//! xorshift64*: one multiply and three shifts per draw, no dependency, and a
//! period of 2^64 - 1. This is for move selection, Zobrist key generation, and
//! datagen jitter — not for anything cryptographic.

/// xorshift64* generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Fixed-seed constructor. A zero seed is remapped, since xorshift's
    /// all-zero state is absorbing.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    /// Seed from the system clock. Good enough to vary games between runs.
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0);
        // Mix so that near-identical clock readings give distant states.
        Self::new(nanos ^ (nanos << 31) ^ 0xA076_1D64_78BD_642F)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `0..n`. Uses Lemire's multiply-shift, which is fast and has
    /// bias below 2^-64 relative — irrelevant at the sizes used here.
    ///
    /// Returns 0 when `n == 0`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Pick a uniform index into a slice of length `len`, or `None` if empty.
    #[inline]
    pub fn choose(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            None
        } else {
            Some(self.below(len as u64) as usize)
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::from_entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_given_seed() {
        let a: Vec<u64> = (0..8).map(|_| Rng::new(42).next_u64()).collect();
        assert!(a.iter().all(|&x| x == a[0]));

        let mut r1 = Rng::new(7);
        let mut r2 = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn zero_seed_does_not_stick() {
        let mut rng = Rng::new(0);
        assert_ne!(rng.next_u64(), 0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = Rng::new(0xdead_beef);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            let v = rng.below(5);
            assert!(v < 5);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every value should appear");
    }

    #[test]
    fn choose_handles_empty() {
        let mut rng = Rng::new(1);
        assert_eq!(rng.choose(0), None);
        assert_eq!(rng.choose(1), Some(0));
    }
}
