"""Thin wrapper around the custom BPE tokenizer with a tiktoken-compatible API."""
from pathlib import Path
from tokenizers import Tokenizer as _Tokenizer

TOKENIZER_PATH = Path(__file__).parent / "data" / "tokenizer" / "tokenizer.json"


class Tokenizer:
    def __init__(self, path: str | Path = TOKENIZER_PATH):
        self._tok = _Tokenizer.from_file(str(path))

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
