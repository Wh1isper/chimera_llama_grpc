import asyncio

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc


async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chimera_llm_pb2_grpc.LLMStub(channel)
        async for r in stub.Inspect(chimera_llm_pb2.InspectRequest(report_duration=1)):
            print(r)
            break


if __name__ == "__main__":
    asyncio.run(run())
