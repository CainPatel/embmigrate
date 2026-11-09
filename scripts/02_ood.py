"""Decisive experiment: maps fit on public MS MARCO pairs, evaluated out-of-domain (k sweep, equal-k in-domain comparison, self-match).

Index = native A passage embeddings of the OOD corpus. Queries = B-embedded, mapped into A.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embmigrate.data import ROOT, load_beir
from embmigrate.eval import bootstrap_ci, per_query, retrieve
from embmigrate.maps import fit

OOD = ["scifact", "nfcorpus", "fiqa"]
DIRS = [("bge", "e5"), ("e5", "bge"), ("minilm", "bge"), ("bge", "minilm")]  # (B = new query model, A = existing index)
KINDS = ["identity", "procrustes", "ridge", "mlp"]
KS = [300, 1000, 2000, 3000, 10000, 30000, 180000]
FULL, EQ_K = 180000, 2000
OUT = ROOT.parent / "results"

emb = lambda ds, m, role: np.load(ROOT / "emb" / ds / f"{m}_{role}.npy").astype(np.float32)
fitpool = {m: emb("msmarco_fit", m, "passage") for m in ["bge", "e5", "minilm"]}
perm = np.random.default_rng(0).permutation(len(next(iter(fitpool.values()))))
hold, train = perm[:20000], perm[20000:]  # holdout used only for the self-match check

data = {ds: load_beir(ds) for ds in OOD}
rows, perq = [], {}


def evaluate(ds, qvec, index_model):
    doc_ids, _, qids, _, qrels = data[ds]
    n, r = per_query(retrieve(qvec, emb(ds, index_model, "passage"), doc_ids, qids), qrels)
    return n, r


native = {}
for ds in OOD:
    for m in ["bge", "e5", "minilm"]:
        native[ds, m] = evaluate(ds, emb(ds, m, "query"), m)
        rows.append(dict(src=m, dst=m, map="native", k=0, fit="-", dataset=ds, ndcg10=native[ds, m][0].mean(), recall100=native[ds, m][1].mean()))
        print(f"native {m:7s} {ds:9s} nDCG@10={native[ds, m][0].mean():.4f}", flush=True)

for B, A in DIRS:
    for kind in KINDS:
        for k in [0] if kind == "identity" else KS:
            idx = train[:k] if k else train[:1]
            mp = fit(fitpool[B][idx], fitpool[A][idx], kind)
            for ds in OOD:
                n, r = evaluate(ds, mp.apply(emb(ds, B, "query")), A)
                perq[B, A, kind, k, "msmarco", ds] = n
                rows.append(dict(src=B, dst=A, map=kind, k=k, fit="msmarco", dataset=ds, ndcg10=n.mean(), recall100=r.mean(), lam=getattr(mp, "lam", None)))
            if k == FULL or kind == "identity":
                xb, xa = mp.apply(fitpool[B][hold[:10000]]), fitpool[A][hold[:10000]]
                xa = xa / np.linalg.norm(xa, axis=1, keepdims=True)
                r1 = float((np.argmax(xb @ xa.T, axis=1) == np.arange(len(xb))).mean())
                rows.append(dict(src=B, dst=A, map=kind, k=k, fit="msmarco", dataset="E2_selfmatch_R@1", ndcg10=r1))
            print(f"{B}->{A} {kind:10s} k={k:6d} " + " ".join(f"{ds}={perq[B, A, kind, k, 'msmarco', ds].mean():.4f}" for ds in OOD), flush=True)
        # equal-k in-domain comparison: fit on 2k docs of the target corpus itself (docs only, no queries)
        if kind != "identity":
            for ds in OOD:
                xb, xa = emb(ds, B, "passage"), emb(ds, A, "passage")
                sel = np.random.default_rng(0).choice(len(xb), EQ_K, replace=False)
                mp = fit(xb[sel], xa[sel], kind)
                n, r = evaluate(ds, mp.apply(emb(ds, B, "query")), A)
                perq[B, A, kind, EQ_K, "indomain", ds] = n
                rows.append(dict(src=B, dst=A, map=kind, k=EQ_K, fit="indomain", dataset=ds, ndcg10=n.mean(), recall100=r.mean()))

df = pd.DataFrame(rows)
for (ds, m), (n, _) in native.items():
    sel = (df.dataset == ds) & (df.dst == m)
    df.loc[sel, "frac_native_A"] = df.loc[sel, "ndcg10"] / n.mean()
df.to_csv(OUT / "ood.csv", index=False)

# significance: procrustes vs ridge/mlp at full k, and transfer gaps at equal k
sig = []
for B, A in DIRS:
    for ds in OOD:
        p = perq[B, A, "procrustes", FULL, "msmarco", ds]
        for other in ["ridge", "mlp"]:
            o = perq[B, A, other, FULL, "msmarco", ds]
            sig.append(dict(src=B, dst=A, dataset=ds, test=f"procrustes - {other} @k=180k", diff=(p - o).mean(), ci=bootstrap_ci(p, o)))
            gap = lambda kind: perq[B, A, kind, EQ_K, "indomain", ds] - perq[B, A, kind, EQ_K, "msmarco", ds]
            sig.append(dict(src=B, dst=A, dataset=ds, test=f"gap(procrustes) - gap({other}) @k=2k", diff=(gap("procrustes") - gap(other)).mean(), ci=bootstrap_ci(gap("procrustes"), gap(other))))
pd.DataFrame(sig).to_csv(OUT / "ood_significance.csv", index=False)

# k-sweep plot: fraction of native A, averaged over OOD sets
fig, axes = plt.subplots(1, len(DIRS), figsize=(5 * len(DIRS), 4), sharey=True)
ms = df[(df.fit == "msmarco") & df.dataset.isin(OOD)]
for ax, (B, A) in zip(axes, DIRS):
    for kind in KINDS[1:]:
        s = ms[(ms.src == B) & (ms.dst == A) & (ms["map"] == kind)].groupby("k").frac_native_A.mean()
        ax.plot(s.index, s.values, marker="o", label=kind)
    ax.axhline(ms[(ms.src == B) & (ms.dst == A) & (ms["map"] == "identity")].frac_native_A.mean(), ls=":", c="gray", label="identity")
    for d in {768, 384} & {fitpool[B].shape[1], fitpool[A].shape[1]}:
        ax.axvline(d, ls="--", c="k", lw=0.7)
    ax.axhline(0.9, c="g", lw=0.7); ax.axhline(0.7, c="r", lw=0.7)
    ax.set(xscale="log", title=f"{B} queries -> {A} index", xlabel="k fit pairs (MS MARCO)")
axes[0].set_ylabel("OOD nDCG@10 / native A (mean of 3 sets)"); axes[0].legend()
fig.tight_layout(); fig.savefig(OUT / "plots" / "ood_ksweep.png", dpi=120)
print("done")
