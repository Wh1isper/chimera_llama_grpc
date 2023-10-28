import asyncio
from typing import Optional

import fire
import grpc
from chimera_llm_proto import chimera_llm_pb2_grpc

from chimera_llama_grpc.log import logger
from chimera_llama_grpc.service import LlamaServicer

DEFAULT_CKPT_DIR = "./ckpt/"
DEFAULT_TOKENIZER_PATH = "./ckpt/tokenizer.model"


async def serve(
    port: int = 50051,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> None:
    server = grpc.aio.server()
    chimera_llm_pb2_grpc.add_LLMServicer_to_server(
        LlamaServicer(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Starting server on port {port}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(fire.Fire(serve))
    finally:
        loop.close()
