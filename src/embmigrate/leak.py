"""Shared pieces of the leak tests: gtr-t5-base embedding (vec2text's format), texts, and scoring."""
import re
from collections import Counter
from functools import cache

import numpy as np
import sacrebleu
import torch

from .data import load_beir, msmarco_fit_pool

N_FIT, N_EVAL = 50_000, 100


@cache
def _gtr():
    from transformers import AutoModel, AutoTokenizer
    from vec2text.models.model_utils import device

    tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    return tok, AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device).eval()


def trunc(texts, n=32):
    """vec2text's gtr inverter was trained on 32-token passages."""
    tok, _ = _gtr()
    return tok.batch_decode([ids[:n] for ids in tok(texts, add_special_tokens=False)["input_ids"]], skip_special_tokens=True)


def gtr(texts, bs=256):
    """Unnormalized mean-pooled gtr-t5-base encoder output, as vec2text expects."""
    from vec2text.models.model_utils import device, mean_pool

    tok, enc = _gtr()
    out = []
    for i in range(0, len(texts), bs):
        x = tok(texts[i : i + bs], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        with torch.no_grad():
            out.append(mean_pool(enc(**x).last_hidden_state, x["attention_mask"]).cpu().numpy())
    return np.concatenate(out)


def texts():
    """(map-fit texts, eval sets, remaining MS MARCO texts). Same seeds as the original leak test."""
    _, pool = msmarco_fit_pool()
    perm = np.random.default_rng(1).permutation(len(pool))
    _, fiqa_docs, _, _, _ = load_beir("fiqa")
    evals = {
        "msmarco_heldout": trunc([pool[i] for i in perm[N_FIT : N_FIT + N_EVAL]]),
        "fiqa": trunc([fiqa_docs[i] for i in np.random.default_rng(1).choice(len(fiqa_docs), N_EVAL, replace=False)]),
    }
    return trunc([pool[i] for i in perm[:N_FIT]]), evals, [pool[i] for i in perm[N_FIT + N_EVAL :]]


def _words(s):
    return re.findall(r"\w+", s.lower())


def token_f1(a, b):
    a, b = Counter(_words(a)), Counter(_words(b))
    common = sum((a & b).values())
    return 0.0 if not common else 2 * common / (sum(a.values()) + sum(b.values()))


def score(refs, hyps):
    f1 = np.array([token_f1(h, r) for h, r in zip(hyps, refs)])
    bleu = np.array([sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)])
    return dict(bleu=bleu.mean(), token_f1=f1.mean(), frac_f1_ge_0_5=(f1 >= 0.5).mean(), exact=np.mean([h.strip() == r.strip() for h, r in zip(hyps, refs)]))
