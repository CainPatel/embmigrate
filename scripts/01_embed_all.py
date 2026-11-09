"""Embed MS MARCO fit pool (passages) and BEIR OOD sets (passages + queries) with every local model."""
import time

from embmigrate.data import load_beir, msmarco_fit_pool
from embmigrate.embed import MODELS, embed

OOD = ["scifact", "nfcorpus", "fiqa"]

jobs = [("msmarco_fit", msmarco_fit_pool()[1], "passage")]
for ds in OOD:
    _, docs, _, qs, _ = load_beir(ds)
    jobs += [(ds, docs, "passage"), (ds, qs, "query")]

for model in MODELS:
    for ds, texts, role in jobs:
        t = time.time()
        embed(texts, model, role, ds)
        print(f"{model:7s} {ds:12s} {role:8s} n={len(texts):7d} {time.time() - t:6.0f}s", flush=True)
