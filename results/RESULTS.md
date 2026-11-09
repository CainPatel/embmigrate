# Detailed results

## 1. Migration (2026-11-08)

The setup is the realistic "vectors only" migration. Maps are fit on **public MS MARCO passage pairs only**, then evaluated **out of domain** on SciFact, NFCorpus and FiQA (BEIR test qrels). The index holds native A passage vectors; queries are embedded with B and mapped into A. Metric: nDCG@10 as a fraction of native A→A, averaged over the three sets. All three models run locally.

Native baselines match published BEIR numbers (bge SciFact .740 / NFCorpus .373 / FiQA .406; MiniLM .645 / .316 / .369).

### Fraction of native nDCG@10 (out of domain, fit on MS MARCO)

| B queries → A index | identity (no map) | Procrustes k=1k | k=10k | k=180k | best other @180k | in-domain Procrustes, k=2k |
|---|---|---|---|---|---|---|
| bge → e5 (768→768)      | 0.83 | 0.89 | **0.93** | 0.93 | ridge 0.93 | 0.94 |
| e5 → bge (768→768)      | 0.49 | 0.82 | 0.88 | 0.89 | mlp 0.88 | 0.92 |
| MiniLM → bge (384→768)  | 0.01 | 0.50 | 0.65 | 0.67 | mlp 0.65 | 0.74 |
| bge → MiniLM (768→384)  | 0.01 | 0.57 | 0.74 | 0.77 | mlp 0.80 | 0.89 |

Plot: `plots/ood_ksweep.png`. Raw data: `ood.csv`. Paired-bootstrap CIs: `ood_significance.csv`.

### Findings

1. **It works, but it never matches native.** For related models, the best result is 89–93% of native. For dissimilar ones (MiniLM ↔ bge) it is 67–80%. No mapped direction beats *either* model used natively. For example, MiniLM queries mapped into the bge index score 0.67× bge, while native MiniLM scores 0.88× bge.
2. **bge ↔ e5 is an easy case, not a clean test of "rotation".** Both are fine-tuned from `bert-base-uncased`, so their coordinates are already partly aligned. With no map at all, bge → e5 reaches 0.83 of native, and the fitted Procrustes matrix has a mean diagonal of 0.48.
3. **Procrustes is the best or tied map across the k sweep.** It is significantly better than ridge for MiniLM → bge at k=180k on all three sets (+2.5 to +4.6 nDCG points, CIs exclude 0). It is clearly ahead at small k (≤ 3k) in every direction. The MLP catches up or wins only at k=180k on the 768↔384 pair.
4. **The constrained map does *not* transfer with a smaller gap.** We expected Procrustes to lose less than ridge or the MLP when moving from in-domain to out-of-domain fitting, measured at equal k = 2k. Instead its gap is *larger*, because Procrustes gains the most from in-domain fitting. Most CIs include 0. It is still the best performer in absolute out-of-domain terms.
5. **Sample efficiency.** Quality jumps once k passes the embedding dimension (dashed lines on the plot) and saturates by roughly 10k–30k pairs.
6. **2k in-domain docs beat 180k public pairs.** If you have any of your own text, a small in-domain fit is the best option (0.74–0.94).
7. **Self-match is saturated.** Retrieving each mapped document's own A vector among 10k held-out documents gives R@1 ≥ 0.996 for every fitted map, so it doesn't separate the methods.

### Against the targets set before running

- **Failure threshold** (nothing above 0.70 of native): not hit. The related-model pairs reach about 0.9.
- **Target ≥ 0.90 at k=10k:** met out of domain for bge → e5 only, the shared-initialization pair. e5 → bge reaches 0.88.
- **Constrained map transfers better:** not confirmed (see finding 4).

### Not run

Commercial API models, an in-domain MS MARCO evaluation split, multiple MLP seeds (1 run, not 3), PCA whitening, neighbor-overlap and pair-cosine metrics, a mixed old/new index, and pair-free alignment.

## 2. Leak test (2026-11-08)

**Setup.** An attacker has leaked B passage vectors, access to model B, and public text. They embed 5k or 50k public MS MARCO passages with both B and `gtr-t5-base`, fit Procrustes B → gtr, map the leaked vectors, and invert them with vec2text's pretrained gtr inverter (20 correction steps, no beam). Texts are truncated to 32 tokens to match the inverter. Eval: 100 held-out MS MARCO passages + 100 FiQA passages. Script: `scripts/03_leak.py`. Data: `leak.csv`, `leak_examples.csv`.

| Condition | MS MARCO token F1 | FiQA token F1 | BLEU (MS MARCO) | cos to true gtr vector |
|---|---|---|---|---|
| Native gtr vectors (upper bound) | **0.86** | **0.72** | 57.7 | 1.00 |
| bge → gtr, k=50k | 0.28 | 0.29 | 3.9 | 0.62 |
| e5 → gtr, k=50k | 0.28 | 0.28 | 4.6 | 0.63 |
| MiniLM → gtr, k=50k | 0.27 | 0.26 | 3.6 | 0.58 |
| bge/e5/MiniLM → gtr, k=5k | 0.24–0.25 | 0.24–0.27 | 3.3–4.2 | 0.56–0.61 |
| Floor (unrelated text) | 0.08 | 0.11 | 1.8 | — |

**Finding.** Inverting native vectors recovers text almost verbatim (97% of MS MARCO passages at token F1 ≥ 0.5). After mapping through a different model, inversion recovers the **topic and some keywords** but not the text: token F1 is about 0.27, against a floor of 0.08–0.11, and at most 3% of passages reach F1 ≥ 0.5. Fidelity is the limit. Mapped vectors reach only about 0.6 cosine to the true gtr vector (gtr is T5-based, a different family from the BERT models), and the off-the-shelf inverter needs much closer inputs. 5k public pairs already give most of the effect.

Environment note: vec2text 0.0.13 needs `transformers<4.47`, so the project is pinned to transformers 4.46.3 / sentence-transformers 5.7.0. The migration experiment's embeddings were cached earlier under sentence-transformers 6.2; the leak tests' embeddings were computed under the pinned versions.

## 3. Stronger attacker: fine-tuned inverter (2026-11-09)

**Setup.** Same threat model. The attacker fine-tunes vec2text's pretrained gtr inverter (`jxm/gtr__nq__32`) on 52k public MS MARCO texts paired with their *mapped* B vectors (1 epoch, AdamW lr 1e-4, batch 64). They then invert the held-out leaked vectors in one shot (greedy, no correction steps). Control: the same fine-tune on **raw** bge vectors, with no map. Script: `scripts/04_leak_finetune.py`. Data: `leak_finetune.csv`, `leak_finetune_examples.csv`. An earlier identical run gave the same numbers to ±0.01.

| Inverted vectors (one-shot) | Token F1, MS MARCO / FiQA | F1 ≥ 0.5 |
|---|---|---|
| Native gtr, off-the-shelf (reference) | 0.54 / 0.49 | 63% / 45% |
| bge → gtr, off-the-shelf | 0.27 / 0.29 | 5% / 2% |
| bge → gtr, **fine-tuned** | **0.30 / 0.32** | 10% / 7% |
| MiniLM → gtr, off-the-shelf | 0.27 / 0.26 | 4% / 2% |
| MiniLM → gtr, **fine-tuned** | 0.29 / 0.28 | 6% / 3% |
| bge raw (no map), fine-tuned (control) | 0.13 / 0.14 | 0% / 0% |

**Findings.**
1. **Fine-tuning helps only a little.** Token F1 rises about 0.02–0.04 and passages with F1 ≥ 0.5 roughly double (to 6–10%), but reconstructions stay topic-level. For example, a passage about renting yurts for weddings and festivals comes back as "We have rented thousands of yogi for weddings… and other special events". Training loss flattened by mid-epoch (≈3.2 for bge, 3.4 for MiniLM).
2. **The rotation is a cheap head start for the attacker.** With the same 52k-example budget, training on raw bge vectors gets less than half as far (F1 0.13, loss still 4.15 at the end). Mapping into the space of an existing inverter does most of the work.
3. **Still unknown: what a well-resourced attacker reaches.** vec2text's own inverters were trained on millions of examples over many epochs. Our control shows the 52k budget is too small to learn a new space from scratch, so it can't settle whether the mapping or the training budget is the ceiling. Untested as well: a target model from the same family as the victim's, where mapping fidelity is likely higher.

Note: runs on MPS slowed sharply over time (the control's 800 steps took about 3.2 h, against 42 min for the first fine-tune). Probably thermal or memory pressure; it doesn't affect results.
