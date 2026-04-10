"""Thin wrapper around the custom BPE tokenizer with a tiktoken-compatible API."""
from pathlib import Path
from tokenizers import Tokenizer as _Tokenizer

TOKENIZER_PATH = Path(__file__).parent / "data" / "tokenizer" / "tokenizer.json"
HF_TOKENIZER_REPO = "tsuberim/merlin-tokenizer-v0"


def _resolve_path(path: Path) -> Path:
    """Return path if it exists, otherwise download from HuggingFace."""
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download
    import os
    print(f"[tok] tokenizer not found locally, downloading from {HF_TOKENIZER_REPO} ...")
    downloaded = hf_hub_download(
        repo_id=HF_TOKENIZER_REPO,
        filename="tokenizer.json",
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(downloaded)


class Tokenizer:
    def __init__(self, path: str | Path = TOKENIZER_PATH):
        self._tok = _Tokenizer.from_file(str(_resolve_path(Path(path))))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)


def load() -> Tokenizer:
    return Tokenizer()
