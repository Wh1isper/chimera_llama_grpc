import asyncio
import json

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc


async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chimera_llm_pb2_grpc.LLMStub(channel)
        dialog = [
            chimera_llm_pb2.ChatMessage(
                role=chimera_llm_pb2.Role.SYSTEM,
                content="You will play the role of an ARPC host and the user will start an adventure as an adventurer."
                "You will need to either provide a backstory for the user when they first start the dialogue, or use the backstory provided by the user."
                "The game is option-driven, offering the user a few fixed options, but also allowing the user to make actions outside of those options"
                "Once the user's character dies, the game is over.",
            ),
            chimera_llm_pb2.ChatMessage(
                role=chimera_llm_pb2.Role.USER,
                content="Hi, let's start a game",
            ),
        ]
        for d in dialog:
            if d.role == chimera_llm_pb2.Role.USER:
                print(f"User: {d.content.strip()}")
            elif d.role == chimera_llm_pb2.Role.ASSISTANT:
                print(f"LLAMA: {d.content.strip()}")
        while True:
            response = await stub.Chat(
                chimera_llm_pb2.ChatRequest(
                    request_id="1",
                    messages=dialog,
                    inference_args=chimera_llm_pb2.InferenceArgs(
                        temperature=0.6,
                        top_p=0.9,
                    ),
                )
            )
            dialog.append(
                chimera_llm_pb2.ChatMessage(
                    role=chimera_llm_pb2.Role.ASSISTANT,
                    content=response.message.content,
                )
            )

            print(f"LLAMA: {response.message.content.strip()}")
            content = input("User: ")
            dialog.append(
                chimera_llm_pb2.ChatMessage(
                    role=chimera_llm_pb2.Role.USER,
                    content=content,
                )
            )


if __name__ == "__main__":
    asyncio.run(run())
