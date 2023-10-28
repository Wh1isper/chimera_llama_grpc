import pytest

from chimera_llama_grpc.model_manager import ModelManager


@pytest.fixture
def model_manager(llama_ckpt_dir, llama_tokenizer_path):
    model_manager = ModelManager(
        llama_ckpt_dir, llama_tokenizer_path, {}, init_model_when_construct=False
    )
    return model_manager


def test_model_manager(model_manager: ModelManager):
    assert len(model_manager.available_models) == 2


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
