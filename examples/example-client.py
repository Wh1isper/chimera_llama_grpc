import asyncio

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc


async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chimera_llm_pb2_grpc.LLMStub(channel)
        async for r in stub.Inspect(chimera_llm_pb2.InspectRequest()):
            print(r)
            break
        response = await stub.LoadModel(chimera_llm_pb2.LoadModelRequest(model_id=1))
        print(response)
        response = await stub.LoadModel(chimera_llm_pb2.LoadModelRequest(model_id=2))
        print(response)


if __name__ == "__main__":
    asyncio.run(run())
