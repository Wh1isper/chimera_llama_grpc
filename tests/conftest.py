import pytest


@pytest.fixture
def llama_ckpt_dir(tmp_path):
    """
    Generate a temporary directory mock for llama ckpt_dir

    /tmpdir/
    ├── llama-2-7b
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    ├── llama-2-7b-chat
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json

    """
    ckpt_dir = tmp_path / "ckpt_dir"
    ckpt_dir.mkdir()
    model_1 = ckpt_dir / "llama-2-7b"
    model_1.mkdir()
    model_1_chk = model_1 / "checklist.chk"
    model_1_chk.touch()
    model_1_pth = model_1 / "consolidated.00.pth"
    model_1_pth.touch()
    model_1_json = model_1 / "params.json"
    model_1_json.touch()

    model_2 = ckpt_dir / "llama-2-7b-chat"
    model_2.mkdir()
    model_2_chk = model_2 / "checklist.chk"
    model_2_chk.touch()
    model_2_pth = model_2 / "consolidated.00.pth"
    model_2_pth.touch()
    model_2_json = model_2 / "params.json"
    model_2_json.touch()

    return ckpt_dir


@pytest.fixture
def llama_tokenizer_path(tmp_path):
    """
    Generate a temporary directory mock for llama tokenizer_path

    /tmpdir/
    ├── tokenizer.model
    """

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.touch()
    return tokenizer_path
