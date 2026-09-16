//! Monotonic time, where the platform has any.
//!
//! `wasm32-unknown-unknown` has no clock: `std::time::Instant::now()` compiles
//! and then panics at runtime. The web build therefore limits its search by
//! node count, and this stub lets the time-management code compile and do
//! nothing. Elapsed time is always zero and a deadline is always in the far
//! future, so [`crate::search::Search::check_abort`] never stops on the clock.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use stub::Instant;

#[cfg(target_arch = "wasm32")]
mod stub {
    use std::ops::Add;
    use std::time::Duration;

    /// Ticks that never advance. `now()` is 0 and any deadline is `u64::MAX`,
    /// which is what makes `now() >= deadline` false forever.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct Instant(u64);

    impl Instant {
        pub fn now() -> Self {
            Self(0)
        }

        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;

        fn add(self, _: Duration) -> Instant {
            Instant(u64::MAX)
        }
    }
}
