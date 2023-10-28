import asyncio
from functools import wraps
from typing import Optional

import grpc
from chimera_llm_proto import chimera_llm_pb2, chimera_llm_pb2_grpc
from chimera_llm_proto.tools import get_inference_args, get_uuid

from chimera_llama_grpc.llama import Llama
from chimera_llama_grpc.log import logger
from chimera_llama_grpc.model_manager import ModelManager
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


def log_stream_exception(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            async for r in f(*args, **kwargs):
                yield r
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
        prefer_model_tag: chimera_llm_pb2.ModelTag = chimera_llm_pb2.CHAT,
        *,
        report_duration: int = 5,
    ) -> None:
        if not max_seq_len:
            max_seq_len = 2048
        if not max_batch_size:
            max_batch_size = 8

        self.model_manager = ModelManager(
            ckpt_dir,
            tokenizer_path,
            {
                "max_seq_len": max_seq_len,
                "max_batch_size": max_batch_size,
            },
            prefer_model_tag=prefer_model_tag,
            init_model_when_construct=True,
        )
        self.report_duration = report_duration

    @property
    def model(self) -> Llama:
        return self.model_manager.model

    @log_stream_exception
    async def Inspect(
        self,
        request: chimera_llm_pb2.InspectRequest,
        context: grpc.aio.ServicerContext,
    ) -> chimera_llm_pb2.InspectResponse:
        duration = request.report_duration or self.report_duration
        while True:
            self.model_manager.refresh_avaliable_models()
            yield chimera_llm_pb2.InspectResponse(
                avaliable_models=self.model_manager.avaliable_model_list,
                current_status=self.model_manager.status,
                current_model=self.model_manager.current_model,
            )
            await asyncio.sleep(duration)

    # @log_exception
    # async def LoadModel(self, request, context):
    # TODO: implement LoadModel

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
