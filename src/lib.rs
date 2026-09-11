//! newchessbot — a chess engine.
//!
//! Layered bottom-up:
//!
//! - [`types`] — newtypes for squares, pieces, moves, castling rights.
//! - [`bitboard`] — the square-set type and attack generation.
//! - [`board`] — [`board::Position`], FEN I/O, and move application.
//! - [`magic`] — magic bitboards for sliding attacks.
//! - [`movegen`] — legal move generation, plus `perft` for verifying it.
//! - [`rng`] — a small self-contained PRNG.
//! - [`uci`] — the protocol handler and its search worker thread.
//!
//! Search, evaluation, and NNUE are not implemented yet: `go` currently
//! answers with a random legal move.
//!
//! Conventions (board mapping, score units, allocation and `unsafe` rules) are
//! documented in `CLAUDE.md` at the repository root.

pub mod bitboard;
pub mod board;
pub mod eval;
pub mod magic;
mod magic_constants;
pub mod movegen;
pub mod rng;
pub mod search;
pub mod types;
pub mod uci;
pub mod zobrist;
