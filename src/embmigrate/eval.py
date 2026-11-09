"""Exact inner-product retrieval (equivalent to FAISS IndexFlatIP) + pytrec_eval metrics."""
import numpy as np
import pytrec_eval
import torch


def retrieve(q, d, doc_ids, query_ids, k=100):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    D = torch.from_numpy(d.astype(np.float32)).to(dev)
    run = {}
    for i in range(0, len(q), 1024):
        s, idx = (torch.from_numpy(q[i : i + 1024].astype(np.float32)).to(dev) @ D.T).topk(k, dim=1)
        for qi, (row_s, row_i) in enumerate(zip(s.cpu().numpy(), idx.cpu().numpy())):
            run[query_ids[i + qi]] = {doc_ids[j]: float(v) for j, v in zip(row_i, row_s)}
    return run


def per_query(run, qrels):
    """Per-query nDCG@10 and Recall@100, aligned to sorted query ids."""
    res = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"}).evaluate(run)
    qs = sorted(res)
    return np.array([res[q]["ndcg_cut_10"] for q in qs]), np.array([res[q]["recall_100"] for q in qs])


def bootstrap_ci(a, b, n=10_000, seed=0):
    """95% CI of mean(a - b) by paired bootstrap over queries."""
    diff = a - b
    idx = np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
