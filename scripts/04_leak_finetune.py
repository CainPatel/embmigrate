"""Leak test, stronger attacker: fine-tune vec2text's pretrained gtr inverter on *mapped* B vectors.

Same threat model as 03_leak.py (leaked B vectors + query access to B + public text). The attacker fits
Procrustes B->gtr on 50k public pairs, maps ~52k public B vectors into gtr space, and fine-tunes the inverter
to decode mapped vectors into their text. Then they invert the held-out "leaked" vectors in one shot.
Control ("direct"): fine-tune the same inverter on raw bge vectors, no mapping. If that recovers far more text,
the mapping, not the inverter, is what loses the wording.
SMOKE=1 runs a tiny version to check timing.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import vec2text
from vec2text.models.model_utils import device

from embmigrate.data import ROOT
from embmigrate.embed import embed
from embmigrate.leak import gtr, score, texts, trunc
from embmigrate.maps import fit

SMOKE = os.environ.get("SMOKE") == "1"
N_EXTRA, EPOCHS, BS, LR = (200, 1, 64, 1e-4) if SMOKE else (2_000, 1, 64, 1e-4)  # ponytail: 52k x 1 epoch; loss plateaus within it on mapped vectors
CONDS = [("bge", "mapped"), ("minilm", "mapped"), ("bge", "direct")]  # e5 behaved like bge in the first leak test
GEN = dict(num_beams=1, do_sample=False, min_length=1, max_length=64, no_repeat_ngram_size=0, early_stopping=False)
OUT = ROOT.parent / "results"

fit_texts, evals, rest = texts()
extra = trunc(rest[:N_EXTRA])
g_fit = gtr(fit_texts)
g_norm = float(np.linalg.norm(g_fit, axis=1).mean())
g_true = {ds: gtr(t) for ds, t in evals.items()}


def load():
    torch.manual_seed(0)
    return vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32").to(device)


def invert(model, vecs, bs=100):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(vecs), bs):
            ids = model.generate({"frozen_embeddings": torch.from_numpy(vecs[i : i + bs]).float().to(device)}, GEN)
            out += model.tokenizer.batch_decode(ids, skip_special_tokens=True)
    return out


def finetune(model, vecs, txts):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lab = model.tokenizer(txts, max_length=48, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
    lab[lab == model.tokenizer.pad_token_id] = -100
    X = torch.from_numpy(vecs).float()
    rng = np.random.default_rng(0)
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        for b in np.array_split(rng.permutation(len(X)), len(X) // BS):
            loss = model(None, None, labels=lab[b].to(device), frozen_embeddings=X[b].to(device)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0 or step == 1:
                print(f"  epoch {ep} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return model


rows, examples = [], []


def record(ds, cond, hyps):
    rows.append(dict(dataset=ds, condition=cond, **score(evals[ds], hyps)))
    examples.extend(dict(dataset=ds, condition=cond, original=r, reconstructed=h) for r, h in list(zip(evals[ds], hyps))[:5])
    print(rows[-1], flush=True)


base = load()
for ds in evals:
    record(ds, "native gtr, off-the-shelf, one-shot (reference)", invert(base, g_true[ds]))

norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
for B, how in CONDS:
    if how == "mapped":
        mp = fit(embed(fit_texts, B, "passage", "leak_fit"), g_fit, "procrustes")
        tf = lambda x: mp.apply(x) * g_norm
    else:  # bge is 768-d, same as gtr, so the inverter's input layer fits
        tf = lambda x: norm(x.astype(np.float32)) * g_norm
    test = {ds: tf(embed(t, B, "passage", f"leak_eval_{ds}")) for ds, t in evals.items()}
    if how == "mapped":
        for ds in evals:
            record(ds, f"{B} -> gtr, off-the-shelf, one-shot", invert(base, test[ds]))
    train_vecs = tf(np.concatenate([embed(fit_texts, B, "passage", "leak_fit"), embed(extra, B, "passage", f"leak_extra_{N_EXTRA}")]))
    print(f"fine-tuning on {len(train_vecs)} {how} {B} vectors", flush=True)
    tuned = finetune(load(), train_vecs, fit_texts + extra)
    for ds in evals:
        record(ds, f"{B} {'-> gtr' if how == 'mapped' else 'raw (no map)'}, fine-tuned inverter, one-shot", invert(tuned, test[ds]))
    del tuned

if not SMOKE:
    pd.DataFrame(rows).to_csv(OUT / "leak_finetune.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT / "leak_finetune_examples.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
