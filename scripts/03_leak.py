"""Leak test: can stored B vectors be read back as text by mapping them into gtr-t5-base and running vec2text?

Attacker model: has leaked B passage vectors, query access to B (open model / API) and public text (MS MARCO).
They embed public text with B and gtr, fit Procrustes B->gtr, map the leaked vectors and invert them.
Compare to inverting native gtr vectors (the upper bound) and to unrelated text (the floor).
vec2text's gtr-base inverter was trained on 32-token NQ passages, so all texts are truncated to 32 gtr tokens.
"""
import numpy as np
import pandas as pd
import torch
import vec2text
from vec2text.models.model_utils import device

from embmigrate.data import ROOT
from embmigrate.embed import embed
from embmigrate.leak import gtr, score, texts as leak_texts
from embmigrate.maps import fit

KS, STEPS = [5_000, 50_000], 20
OUT = ROOT.parent / "results"
fit_texts, evals, _ = leak_texts()

g_fit = gtr(fit_texts)
g_norm = float(np.linalg.norm(g_fit, axis=1).mean())  # inverter expects unnormalized mean-pooled gtr vectors
corrector = vec2text.load_pretrained_corrector("gtr-base")


def invert(vecs, bs=50):
    out = []
    for i in range(0, len(vecs), bs):
        out += vec2text.invert_embeddings(torch.from_numpy(vecs[i : i + bs]).float().to(device), corrector, num_steps=STEPS)
    return out


rows, examples = [], []
for ds, texts in evals.items():
    g_true = gtr(texts)
    conds = {"native gtr (upper bound)": g_true}
    for B in ["bge", "e5", "minilm"]:
        b_fit = embed(fit_texts, B, "passage", "leak_fit")
        b_eval = embed(texts, B, "passage", f"leak_eval_{ds}")
        for k in KS:
            mp = fit(b_fit[:k], g_fit[:k], "procrustes")
            conds[f"{B} -> gtr, k={k}"] = mp.apply(b_eval) * g_norm
    for name, vecs in conds.items():
        hyps = invert(vecs)
        cos = float((vecs * g_true).sum(1).mean() / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(g_true, axis=1)).mean())
        rows.append(dict(dataset=ds, condition=name, cos_to_true_gtr=cos, **score(texts, hyps)))
        examples += [dict(dataset=ds, condition=name, original=r, reconstructed=h) for r, h in list(zip(texts, hyps))[:5]]
        if name.startswith("native"):  # floor: same reconstructions scored against unrelated originals
            rows.append(dict(dataset=ds, condition="floor (unrelated text)", **score(texts[1:] + texts[:1], hyps)))
        print(rows[-1] if not name.startswith("native") else rows[-2], flush=True)

pd.DataFrame(rows).to_csv(OUT / "leak.csv", index=False)
pd.DataFrame(examples).to_csv(OUT / "leak_examples.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
