"""Pluggable ONNX embedder."""

import math
from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Generate normalized vectors with FastEmbed's CPU ONNX runtime."""

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        fastembed_name = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        }.get(model_name, model_name)
        self.model = TextEmbedding(model_name=fastembed_name, threads=1)
        self.dim = len(next(self.model.embed(["dimension probe"])))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [
            self._normalize(vector)
            for vector in self.model.embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self.model.embed([EMBED_QUERY_PREFIX + text], batch_size=1))
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector) -> list[float]:
        values = vector.tolist()
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Load the embedder once per process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = FastEmbedder()
    return _EMBEDDER
