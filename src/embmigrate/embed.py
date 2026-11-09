"""Local sentence-transformers embedding with per-model, per-role prefixes and a float16 cache."""
from pathlib import Path

import numpy as np

from .data import ROOT

MODELS = {
    "bge": dict(name="BAAI/bge-base-en-v1.5", max_len=512, query="Represent this sentence for searching relevant passages: ", passage=""),
    "e5": dict(name="intfloat/e5-base-v2", max_len=512, query="query: ", passage="passage: "),
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", max_len=256, query="", passage=""),
}
_loaded = {}


def embed(texts, model, role, dataset):
    path = ROOT / "emb" / dataset / f"{model}_{role}.npy"
    if path.exists():
        x = np.load(path)
        assert len(x) == len(texts), f"stale cache {path}"
        return x
    from sentence_transformers import SentenceTransformer

    cfg = MODELS[model]
    if model not in _loaded:
        m = SentenceTransformer(cfg["name"], device="mps")
        m.max_seq_length = cfg["max_len"]
        _loaded[model] = m
    x = _loaded[model].encode([cfg[role] + t for t in texts], batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x.astype(np.float16))
    return x.astype(np.float16)
