import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chimera_llm_proto.chimera_llm_pb2 import AvaliableModel, ModelTag

from chimera_llama_grpc.exceptions import NoSuchModel
from chimera_llama_grpc.llama import Llama
from chimera_llama_grpc.log import logger

STATUS_NOT_READY = 0
STATUS_INITLIZING = 1
STATUS_READY = 2


def lock_acquire(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.mutex:
            return func(self, *args, **kwargs)

    return wrapper


class StatusfulModel:
    def __init__(self):
        self.status = STATUS_NOT_READY
        self.model: Optional[Llama] = None
        self.current_model: Optional[AvaliableModel] = None


class ModelManager:
    model_description = "Chimera Llama Backend: https://github.com/Wh1isper/chimera_llama_grpc"

    def __init__(
        self,
        ckpt_dir: Union[Path, str],
        tokenizer_path: Union[Path, str],
        model_params: Dict[str, Any],
        *,
        prefer_model_tag: int = ModelTag.CHAT,
    ) -> None:
        if not isinstance(ckpt_dir, Path):
            ckpt_dir = Path(ckpt_dir).resolve()
        self.ckpt_dir = ckpt_dir

        if not isinstance(tokenizer_path, Path):
            tokenizer_path = Path(tokenizer_path).resolve()

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Cannot find tokenizer_path: {self.tokenizer_path}")
        self.tokenizer_path = tokenizer_path

        self.model_params = model_params
        self.available_models: Dict[Path, AvaliableModel] = self.retrieve_available_models()

        if not self.available_models:
            raise FileNotFoundError(f"No available models found in ckpt_dir: {self.ckpt_dir}")
        self.model_host = StatusfulModel()
        self.prefer_model_tag = prefer_model_tag
        self.mutex = threading.Lock()

    @property
    def model(self):
        if not self.model_host.model:
            self.change_model()
        return self.model_host.model

    @property
    def status(self):
        return self.model_host.status

    @property
    def current_model(self):
        return self.model_host.current_model

    @property
    def avaliable_model_list(self) -> List[AvaliableModel]:
        return list(self.available_models.values())

    def refresh_avaliable_models(self) -> None:
        self.available_models = self.retrieve_available_models()

    def _initialize_model(self, model_id: str) -> Llama:
        path = self._get_path_from_model_id(model_id)
        if not path:
            raise ValueError(f"Cannot find model path for model_id: {model_id}")
        logger.info(f"Initializing model from path: {path}, params: {self.model_params}")
        m = Llama.build(
            path.as_posix(),
            self.tokenizer_path.as_posix(),
            **self.model_params,
        )
        logger.info(f"Model initialized: {m}")
        return m

    @lock_acquire
    def change_model(
        self,
        model_id: Optional[str] = None,
    ) -> None:
        self.model_host.status = STATUS_INITLIZING
        if not model_id:
            prefer_models = self.filter_models_by_tag(self.prefer_model_tag)
            if prefer_models:
                model_id = prefer_models[0].model_id
            else:
                model_id = self.avaliable_model_list[0].model_id
        try:
            current_model = self.get_model_by_id(model_id)
            if not current_model:
                raise NoSuchModel(f"Model not found via model_id: {model_id}")
            logger.info(f"Changing model to {current_model.model_name}")
            if self.model_host.model:
                self.model_params["model_parallel_size"] = int(os.environ.get("WORLD_SIZE", 1))
            del self.model_host.model
            self.model_host.model = self._initialize_model(current_model.model_id)
            self.model_host.current_model = current_model
        except Exception as e:
            logger.exception(e)

            self.model_host.status = STATUS_NOT_READY
            self.model_host.model = None
            self.model_current_model = None

            raise
        else:
            self.model_host.status = STATUS_READY

    def _get_path_from_model_id(self, model_id: str) -> Path:
        for path, model in self.available_models.items():
            if model.model_id == model_id:
                return path

    def filter_models_by_tag(self, tag: ModelTag) -> List[AvaliableModel]:
        return [model for model in self.avaliable_model_list if tag in model.model_tagas]

    def retrieve_available_models(self) -> Dict[Path, AvaliableModel]:
        """
        Retrieve available models from ckpt_dir.

        Returns:
            Dict[Path, AvaliableModel]: Model path: model info.

        Example:
            ckpt_dir/
            ├── llama-2-7b
            │   ├── checklist.chk
            │   ├── consolidated.00.pth
            │   └── params.json
            ├── llama-2-7b-chat
            │   ├── checklist.chk
            │   ├── consolidated.00.pth
            │   └── params.json
            └── tokenizer.model

        Returns:
            List[AvaliableModel]: List of available models.
        """

        avaliable_models: Dict[Path, AvaliableModel] = {}
        index = 1

        for ckpt_model_dir in self.ckpt_dir.iterdir():
            if not ckpt_model_dir.is_dir():
                logger.debug(f"Skip non-dir file: {ckpt_model_dir}")
                continue

            logger.debug(f"Found model dir: {ckpt_model_dir}")

            # Check consolidated.*.pth and params.json
            if not list(ckpt_model_dir.glob("consolidated.*.pth")):
                logger.debug(f"Skip model dir without consolidated.*.pth: {ckpt_model_dir}")
                continue
            if not list(ckpt_model_dir.glob("params.json")):
                logger.debug(f"Skip model dir without params.json: {ckpt_model_dir}")
                continue

            model_name = ckpt_model_dir.name
            model_tags = self._get_tags_from_dir_name(ckpt_model_dir.name)
            model_size = self._get_model_size_from_dir_name(ckpt_model_dir.name)
            avaliable_models[ckpt_model_dir] = AvaliableModel(
                model_id=index,
                model_name=model_name,
                model_description=self.model_description,
                model_tagas=model_tags,
                size=model_size,
            )
            index += 1
        logger.debug(f"Found {len(avaliable_models)} models in {self.ckpt_dir}")
        return avaliable_models

    def _get_tags_from_dir_name(self, dir_name: str) -> List[int]:
        tags = [ModelTag.TEXT]
        if "chat" in dir_name:
            tags.append(ModelTag.CHAT)
        if "code" in dir_name:
            tags.append(ModelTag.CODE)
        return tags

    def _get_model_size_from_dir_name(self, dir_name: str) -> str:
        """
        Get model size from dir_name.

        Example:
            >>> manager = ModelManager(ckpt_dir, tokenizer_path, model_params)
            >>> manager._get_model_size_from_dir_name("llama-2-7b")
            7b
        """
        try:
            return dir_name.split("-")[2]
        except IndexError:
            return "unknown"

    def get_model_by_id(self, model_id: str) -> Optional[AvaliableModel]:
        for model in self.avaliable_model_list:
            if model.model_id == model_id:
                return model
        return None
