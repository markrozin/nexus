//! Offline magic-constant search. Run once, then commit the output:
//!
//! ```text
//! cargo build --release --features magicgen --bin magicgen
//! ./target/release/magicgen > src/magic_constants.rs
//! ```
//!
//! Build before redirecting: the shell truncates the target file first, and the
//! crate needs it to compile.

use nexus::magic::find_magic;
use nexus::rng::Rng;
use nexus::types::Square;

fn main() {
    // Fixed seed so a regeneration is reproducible.
    let mut rng = Rng::new(0x0000_1234_5678_9abc);
    println!("//! Magic constants for `crate::magic`. Generated, do not hand-edit.");
    println!("//!");
    println!("//! Regenerate with:");
    println!("//! `cargo build --release --features magicgen --bin magicgen`");
    println!("//! then `./target/release/magicgen > src/magic_constants.rs`.");
    println!("//!");
    println!("//! Only needed if the mask scheme in `crate::magic` changes.");
    println!();
    emit("ROOK_MAGICS", false, &mut rng);
    println!();
    emit("BISHOP_MAGICS", true, &mut rng);
}

fn emit(name: &str, is_bishop: bool, rng: &mut Rng) {
    println!("#[rustfmt::skip]");
    println!("pub(crate) const {name}: [u64; 64] = [");
    for sq in Square::ALL {
        let magic = find_magic(sq, is_bishop, rng);
        println!("    0x{magic:016x}, // {sq}");
    }
    println!("];");
}
