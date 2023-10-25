from functools import wraps
from typing import Optional

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc
from chimera_llm_proto.tools import get_inference_args, get_uuid

from chimera_llama_grpc.llama import Llama
from chimera_llama_grpc.log import logger
from chimera_llama_grpc.tools import run_in_threadpool


def log_exception(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise

    return wrapper


pb_role_map = {
    chimera_llm_pb2.SYSTEM: "system",
    chimera_llm_pb2.USER: "user",
    chimera_llm_pb2.ASSISTANT: "assistant",
}
role_pb_map = {v: k for k, v in pb_role_map.items()}


def role_to_str(role: chimera_llm_pb2.Role) -> str:
    if role not in pb_role_map:
        raise ValueError(f"Cannot convert role to str: {role}")
    return pb_role_map[role]


def role_to_pb(role: str) -> chimera_llm_pb2.Role:
    if role not in role_pb_map:
        raise ValueError(f"Cannot convert role to pb: {role}")
    return role_pb_map[role]


class LlamaServicer(chimera_llm_pb2_grpc.LLMServicer):
    def __init__(
        self,
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        if not max_seq_len:
            max_seq_len = 2048
        if not max_batch_size:
            max_batch_size = 8

        logger.info("Initializing Llama model...")
        logger.debug(
            "\n"
            f"ckpt_dir: {ckpt_dir} \n"
            f"tokenizer_path: {tokenizer_path} \n"
            f"max_seq_len: {max_seq_len} \n"
            f"max_batch_size: {max_batch_size} "
        )

        self.model: Llama = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        logger.info("Llama model initialized.")

    @log_exception
    async def Completion(
        self,
        request: chimera_llm_pb2.CompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> chimera_llm_pb2.CompletionPrediction:
        kwargs = {}
        if not request.prompt:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("prompt must not be empty")
            raise ValueError("prompt must not be empty")
        kwargs = {
            "prompts": [request.prompt],
        }
        kwargs.update(get_inference_args(request.inference_args))

        logger.debug(f"Text completion request: {kwargs}")
        predictions = await run_in_threadpool(self.model.text_completion, **kwargs)
        return chimera_llm_pb2.CompletionPrediction(
            request_id=request.request_id,
            response_id=get_uuid(),
            generation=predictions[0]["generation"],
        )

    @log_exception
    async def Chat(
        self,
        request: chimera_llm_pb2.ChatRequest,
        context: grpc.aio.ServicerContext,
    ) -> chimera_llm_pb2.ChatPrediction:
        if not request.messages:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("messages must not be empty")
            raise ValueError("messages must not be empty")
        dialog = []
        for messages in request.messages:
            dialog.append(
                {
                    "role": role_to_str(messages.role),
                    "content": messages.content,
                }
            )

        kwargs = {
            "dialogs": [dialog],
        }
        kwargs.update(get_inference_args(request.inference_args))

        logger.debug(f"Chat request: {kwargs}")
        predictions = await run_in_threadpool(self.model.chat_completion, **kwargs)
        return chimera_llm_pb2.ChatPrediction(
            request_id=request.request_id,
            response_id=get_uuid(),
            message=chimera_llm_pb2.ChatMessage(
                role=role_to_pb(predictions[0]["generation"]["role"]),
                content=predictions[0]["generation"]["content"],
            ),
        )
