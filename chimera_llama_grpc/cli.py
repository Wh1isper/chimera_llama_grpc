import re
import sys
from pathlib import Path

import click

from chimera_llama_grpc.entrypoint import DEFAULT_CKPT_DIR, DEFAULT_TOKENIZER_PATH
from chimera_llama_grpc.tools import load_entry_point

_HERE = Path(__file__).parent
ENTRYPOINT = _HERE / "entrypoint.py"


@click.command()
# TorchRun options
@click.option("--nnodes", default=None)
@click.option("--nproc-per-node", default=None)
@click.option("--node-rank", default=None)
@click.option("--master-addr", default=None)
@click.option("--master-port", default=None)
@click.option("--local-addr", default=None)
# Chimera-llama options
@click.option("--port", default=50051)
@click.option("--ckpt-dir", default=DEFAULT_CKPT_DIR)
@click.option("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
@click.option("--max-seq-len", default=None)
@click.option("--max-batch-size", default=None)
def start(
    nnodes,
    nproc_per_node,
    node_rank,
    master_addr,
    master_port,
    local_addr,
    port,
    ckpt_dir,
    tokenizer_path,
    max_seq_len,
    max_batch_size,
):
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])

    # torch.distributed.launch
    sys.argv = [sys.argv[0]]
    if nnodes:
        sys.argv.extend(["--nnodes", str(nnodes)])
    if nproc_per_node:
        sys.argv.extend(["--nproc-per-node", str(nproc_per_node)])
    if node_rank:
        sys.argv.extend(["--node-rank", str(node_rank)])
    if master_addr:
        sys.argv.extend(["--master-addr", str(master_addr)])
    if master_port:
        sys.argv.extend(["--master-port", str(master_port)])
    if local_addr:
        sys.argv.extend(["--local-addr", str(local_addr)])

    # chimera_llama_grpc entrypoint
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
