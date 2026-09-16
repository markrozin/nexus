use std::io;

use nexus::uci::Uci;

/// Read UCI commands from stdin on this thread. `go` dispatches to a worker,
/// so the loop stays responsive to `stop` and `quit`.
fn main() -> io::Result<()> {
    nexus::magic::init();
    let stdin = io::stdin();
    Uci::stdout().run(stdin.lock())
}
