import re
import sys
from pathlib import Path

import click

from chimera_llama_grpc.entrypoint import DEFAULT_CKPT_DIR, DEFAULT_TOKENIZER_PATH
from chimera_llama_grpc.tools import load_entry_point

_HERE = Path(__file__).parent
ENTRYPOINT = _HERE / "entrypoint.py"


@click.command()
@click.option("--port", default=50051)
@click.option("--ckpt_dir", default=DEFAULT_CKPT_DIR)
@click.option("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
@click.option("--max_seq_len", default=None)
@click.option("--max_batch_size", default=None)
def start(port, ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])

    sys.argv = [sys.argv[0]]
    sys.argv.extend(["--nproc-per-node", "1"])

    sys.argv.extend([ENTRYPOINT.resolve().as_posix()])
    ckpt_dir = Path(ckpt_dir).resolve().as_posix()
    tokenizer_path = Path(tokenizer_path).resolve().as_posix()
    sys.argv.extend(["--port", f"{port}"])
    sys.argv.extend(["--ckpt_dir", f"{ckpt_dir}"])
    sys.argv.extend(["--tokenizer_path", f"{tokenizer_path}"])
    if max_seq_len:
        sys.argv.extend(["--max_seq_len", f"{max_seq_len}"])
    if max_batch_size:
        sys.argv.extend(["--max_batch_size", f"{max_batch_size}"])

    sys.exit(load_entry_point("torch", "console_scripts", "torchrun")())


@click.group()
def cli():
    pass


cli.add_command(start)

if __name__ == "__main__":
    cli()
