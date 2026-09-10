use std::io;

use newchessbot::uci::Uci;

/// Read UCI commands from stdin on this thread. `go` dispatches to a worker,
/// so the loop stays responsive to `stop` and `quit`.
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    Uci::stdout().run(stdin.lock())
}
