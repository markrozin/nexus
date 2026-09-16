//! The engine behind a C ABI, for the browser.
//!
//! No wasm-bindgen: the whole surface is a handful of `extern "C"` functions
//! passing UTF-8 through the module's own linear memory, which keeps the
//! dependency list empty and the JavaScript glue small.
//!
//! The protocol, for every entry point: JavaScript allocates buffers with
//! [`alloc`], writes UTF-8 into them, and passes pointer/length pairs. Each
//! function writes UTF-8 into the caller's output buffer and returns the number
//! of bytes written, or `-1` when an argument is not something this engine
//! understands.
//!
//! The browser has no clock (see `nexus::clock`), so [`best_move`] is limited
//! by node count rather than by time.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use nexus::board::Position;
use nexus::movegen::generate_legal;
use nexus::nnue::Network;
use nexus::search::{
    is_insufficient_material, Evaluator, Search, SearchLimits, MATE, MATE_IN_MAX_PLY,
};

/// The network the page plays with: trained on 191.5M Lichess positions, and
/// far stronger than the handcrafted evaluation.
static NET_BYTES: &[u8] = include_bytes!("../../networks/candidates/lichess-wdl0-40.bin");

thread_local! {
    /// One search, reused across moves so its transposition table stays warm.
    /// WebAssembly here is single-threaded, so a thread local is the whole story.
    static SEARCH: RefCell<Search> = RefCell::new(new_search());
}

fn new_search() -> Search {
    nexus::magic::init();
    let net = Network::from_bytes(NET_BYTES).expect("the embedded network matches HIDDEN");
    let mut search = Search::new(Arc::new(AtomicBool::new(false)));
    // Leaked once: the search borrows the network for the life of the page.
    search.set_evaluator(Evaluator::Nnue(Box::leak(Box::new(net))));
    search
}

/// Reserve `len` bytes for JavaScript to write into. Released by [`dealloc`].
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Release a buffer from [`alloc`].
///
/// # Safety
///
/// `ptr` must come from [`alloc`] with this same `len`, and must not be used
/// afterwards.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    // SAFETY: the caller guarantees ptr/len came from `alloc`, which built the
    // allocation with exactly this capacity and a length of zero.
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

/// Read a UTF-8 argument.
///
/// # Safety
///
/// `ptr` must point to `len` initialised bytes that stay valid for this call.
unsafe fn text<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    // SAFETY: the caller guarantees the pointer and length describe live bytes.
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).ok()
}

/// Write `value` into the caller's buffer, or return -1 if it does not fit.
///
/// # Safety
///
/// `out` must point to `cap` writable bytes.
unsafe fn write(value: &str, out: *mut u8, cap: usize) -> i32 {
    let bytes = value.as_bytes();
    if bytes.len() > cap {
        return -1;
    }
    // SAFETY: the caller guarantees `cap` writable bytes, and the length fits.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
    bytes.len() as i32
}

/// Parse the FEN argument, run `body`, and write what it returns.
///
/// # Safety
///
/// The pointers must describe live buffers, as in [`text`] and [`write`].
unsafe fn with_position<F>(fen: *const u8, fen_len: usize, out: *mut u8, cap: usize, body: F) -> i32
where
    F: FnOnce(Position) -> String,
{
    // SAFETY: forwarded from this function's own contract.
    let Some(fen) = (unsafe { text(fen, fen_len) }) else {
        return -1;
    };
    let Ok(position) = fen.parse::<Position>() else {
        return -1;
    };
    // SAFETY: forwarded from this function's own contract.
    unsafe { write(&body(position), out, cap) }
}

/// Every legal move in the position, space-separated, in UCI notation.
///
/// # Safety
///
/// See [`with_position`].
#[no_mangle]
pub unsafe extern "C" fn legal_moves(
    fen: *const u8,
    fen_len: usize,
    out: *mut u8,
    cap: usize,
) -> i32 {
    // SAFETY: forwarded to the caller.
    unsafe {
        with_position(fen, fen_len, out, cap, |position| {
            generate_legal(&position)
                .iter()
                .map(|mv| mv.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
    }
}

/// Play `mv` and return the resulting FEN. `-1` if that move is not legal here.
///
/// # Safety
///
/// See [`with_position`]; `mv` must also describe live bytes.
#[no_mangle]
pub unsafe extern "C" fn apply_move(
    fen: *const u8,
    fen_len: usize,
    mv: *const u8,
    mv_len: usize,
    out: *mut u8,
    cap: usize,
) -> i32 {
    // SAFETY: forwarded to the caller.
    let Some(wanted) = (unsafe { text(mv, mv_len) }) else {
        return -1;
    };
    let mut legal = false;
    // SAFETY: forwarded to the caller.
    let written = unsafe {
        with_position(fen, fen_len, out, cap, |position| {
            match generate_legal(&position)
                .iter()
                .find(|m| m.to_string().eq_ignore_ascii_case(wanted))
            {
                Some(&mv) => {
                    legal = true;
                    position.make_move(mv).to_fen()
                }
                None => String::new(),
            }
        })
    };
    if legal {
        written
    } else {
        -1
    }
}

/// One word for how the game stands: `checkmate`, `stalemate`, `insufficient`,
/// `fifty`, `check` or `ok`.
///
/// # Safety
///
/// See [`with_position`].
#[no_mangle]
pub unsafe extern "C" fn status(fen: *const u8, fen_len: usize, out: *mut u8, cap: usize) -> i32 {
    // SAFETY: forwarded to the caller.
    unsafe {
        with_position(fen, fen_len, out, cap, |position| {
            let in_check = position.in_check(position.side_to_move());
            if generate_legal(&position).is_empty() {
                return if in_check { "checkmate" } else { "stalemate" }.to_string();
            }
            if is_insufficient_material(&position) {
                return "insufficient".to_string();
            }
            if position.halfmove_clock() >= 100 {
                return "fifty".to_string();
            }
            if in_check { "check" } else { "ok" }.to_string()
        })
    }
}

/// Search `nodes` nodes and report `<move> <score> <depth> <nodes>`, the score
/// in centipawns from the side to move, or `mate <n>`.
///
/// # Safety
///
/// See [`with_position`].
#[no_mangle]
pub unsafe extern "C" fn best_move(
    fen: *const u8,
    fen_len: usize,
    nodes: u64,
    out: *mut u8,
    cap: usize,
) -> i32 {
    // SAFETY: forwarded to the caller.
    unsafe {
        with_position(fen, fen_len, out, cap, |position| {
            SEARCH.with(|search| {
                let mut search = search.borrow_mut();
                let result = search.run(
                    &position,
                    SearchLimits {
                        max_nodes: Some(nodes),
                        ..Default::default()
                    },
                    &mut |_| {},
                );
                let score = if result.score.abs() >= MATE_IN_MAX_PLY {
                    let plies = MATE - result.score.abs();
                    let moves = (plies + 1) / 2 * result.score.signum();
                    format!("mate {moves}")
                } else {
                    result.score.to_string()
                };
                format!(
                    "{} {} {} {}",
                    result.best_move, score, result.depth, result.nodes
                )
            })
        })
    }
}
