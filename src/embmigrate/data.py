"""BEIR datasets from the Hugging Face mirrors (BeIR/*), plus the MS MARCO fit pool."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2] / "data"


def _parquet(repo, sub):
    return pd.read_parquet(hf_hub_download(repo, f"{sub}/{sub}-00000-of-00001.parquet", repo_type="dataset", cache_dir=ROOT / "raw"))


def _doc_text(df):
    # BEIR convention: title + " " + text
    return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip().tolist()


def load_beir(name, split="test"):
    """Returns (doc_ids, doc_texts, query_ids, query_texts, qrels) restricted to queries with qrels."""
    corpus = _parquet(f"BeIR/{name}", "corpus")
    queries = _parquet(f"BeIR/{name}", "queries")
    q = pd.read_csv(hf_hub_download(f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset", cache_dir=ROOT / "raw"), sep="\t", dtype=str)
    qrels = {}
    for qid, did, s in zip(q["query-id"], q["corpus-id"], q["score"].astype(int)):
        qrels.setdefault(qid, {})[did] = s
    queries = queries[queries["_id"].astype(str).isin(qrels)]
    return corpus["_id"].astype(str).tolist(), _doc_text(corpus), queries["_id"].astype(str).tolist(), queries["text"].tolist(), qrels


def msmarco_fit_pool(n=200_000, seed=0):
    """Random MS MARCO passages; sampled IDs persisted to data/splits/msmarco_fit_pool.json."""
    corpus = _parquet("BeIR/msmarco", "corpus")
    split = ROOT / "splits" / "msmarco_fit_pool.json"
    if split.exists():
        ids = set(json.loads(split.read_text()))
        sub = corpus[corpus["_id"].astype(str).isin(ids)]
    else:
        sub = corpus.iloc[np.sort(np.random.default_rng(seed).choice(len(corpus), n, replace=False))]
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text(json.dumps(sub["_id"].astype(str).tolist()))
    return sub["_id"].astype(str).tolist(), _doc_text(sub)
