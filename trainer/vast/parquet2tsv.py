"""Stream a parquet shard of Lichess evaluations as TSV for `lichesseval --tsv`.

    python parquet2tsv.py train-00000.parquet | lichesseval --tsv - shard0.txt

Reads the deduplicated Hugging Face mirror
(mateuszgrzyb/lichess-stockfish-normalized: fen, depth, cp, mate) in record
batches, so a 700 MB shard never has to fit in memory, and writes
`fen \\t depth \\t cp \\t mate` with empty fields for nulls.

Runs on the rental, which installs pyarrow for it. It exits cleanly when the
reader stops early -- a MAX_POSITIONS limit closes the pipe on purpose.
"""

import os
import sys

import pyarrow.parquet as pq

COLUMNS = ["fen", "depth", "cp", "mate"]


def field(value):
    return "" if value is None else str(value)


def main():
    if len(sys.argv) != 2:
        print("usage: parquet2tsv.py SHARD.parquet", file=sys.stderr)
        return 2
    shard = pq.ParquetFile(sys.argv[1])
    out = sys.stdout
    try:
        for batch in shard.iter_batches(batch_size=65536, columns=COLUMNS):
            cols = batch.to_pydict()
            out.write(
                "".join(
                    f"{fen}\t{field(depth)}\t{field(cp)}\t{field(mate)}\n"
                    for fen, depth, cp, mate in zip(cols["fen"], cols["depth"], cols["cp"], cols["mate"])
                )
            )
        out.flush()
    except BrokenPipeError:
        # The reader hung up. Point stdout at devnull so the interpreter's own
        # flush at exit does not raise a second time and turn this into a failure.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
