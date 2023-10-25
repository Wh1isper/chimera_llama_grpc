import asyncio

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc


async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chimera_llm_pb2_grpc.LLMStub(channel)
        response = await stub.Completion(
            chimera_llm_pb2.CompletionRequest(
                request_id="1",
                prompt="I believe the meaning of life is",
                common_args=chimera_llm_pb2.CommonArgs(
                    temperature=0.6,
                    top_p=0.9,
                ),
            )
        )
        print(response.generation)


if __name__ == "__main__":
    asyncio.run(run())
