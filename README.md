# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business record, find all Source 2 / Source 3 records of the same real-world
business (noisy names + addresses; US, India, and France — France appears only in test).
Metric: macro F0.5 per Source 1 entity. No external data or APIs are used; every model is
trained here on the provided training files.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/prepare.py` | TSV → parquet (int row ids), ground-truth pairs, EDA facts, VAL / LOCO splits |
| 2 | `src/normalize.py` | NFKD/accent/case folding; transliteration dictionary, address aliases and noise tokens **learned from FIT training pairs**; address parts |
| 3 | `src/block.py` | per-country candidate generation: name char TF-IDF, name+address word TF-IDF, address char TF-IDF (sparse top-n), exact keys → learned re-ranker → top-K with probability floor |
| 4 | `src/features.py` | rapidfuzz string similarities, address agreement, blocking features, owner-competition features |
| 5 | `src/train.py`, `src/predict.py` | GPU XGBoost (LightGBM fallback), 3-fold GroupKFold by S1, isotonic calibration on out-of-fold scores |
| 6 | `src/decide.py` | owner exclusivity + expected-F0.5 / threshold rule tuned on VAL; writes the two TSVs |
| – | `src/score.py` | the competition metric (self-tests: `python src/score.py`) |

## Current model and validation status

Validation = 33,172 held-out Source-1 entities (US 19,968, India 13,204; 115,068 true pairs), macro F0.5, decision rule re-chosen on one half and scored on the other over 20 pinned split-halves. France has no labels, so no France validation score exists.

| Config | Val F0.5 | Held-out mean / worst | Blocking recall | Cand./S1 |
|---|---|---|---|---|
| A: TF-IDF candidates, stacked GBDT (saved) | 0.98108 | 0.98111 / 0.98042 | 0.9713 | 11.05 |
| B: + reverse dense (e5, k=1) candidates | 0.98562 | 0.98555 / 0.98505 | 0.9861 | 11.47 |
| C: B + owner competition over the full union | 0.98641 | 0.98645 / 0.98584 | 0.9861 | 11.47 |
| E: C + dense features (current best) | 0.98643 | 0.98633 / 0.98585 | 0.9861 | 11.47 |
| F: E + forward dense k=4 candidates (**frozen FWD4, `exp/union_fwd4_ctl1`**) | 0.98673 | 0.98673 / 0.98614 | 0.9871 | 14.68 |

**Leaderboard:** the frozen FWD4 model's test predictions (`src/exp_fwd_test_ctl1e.py`; validator PASS, audit PASS, checksums in `experiments.md`) scored **0.979392** on the public leaderboard, against 0.98673 validation (gap ~0.0073). Retrain noise on the same GPU is about +/-0.0001 full-validation (control #2 = 0.98660); the 0-based-rank variant scored 0.98656 (not an improvement). Test differs from train in candidate-pool density (S2+S3 records per S1: 4.68 train vs 5.75 test) and adds France (15% of test S1, no labels), which behaves differently from US/India in the test diagnostics (more candidates per S1, more owner competition, lower-confidence kept matches).

Most of the gain is candidate expansion (+0.0045); dense features add ~0 once full-union competition is present. Owner-competition features have a train/validation feature-distribution difference (validation rows see extra competitor rows that fit rows do not); no label or fold contamination was found.

Remaining validation errors (model E): 4,113 false negatives (1,602 never retrieved, 1,669 rejected by the decision rule, 461 owner-contested, 381 outranked) and 258 false positives. The per-category oracle gains are upper bounds, not achievable gains (`experiments.md`).

Experiment scripts (Colab, GPU): `src/exp_union.py` (union model), `src/exp_abl_eval2.py` (A-E table), `src/exp_eb_union2.py` + `src/exp_eb_pattern3.py` (error budget), `src/exp_rank2.py` (ranking-feature experiments on the pinned splits).

## Reproduce (Linux, ~50 GB RAM, 12 CPU; ~3 h end to end)

```bash
pip install -r requirements.txt
export PYTHONIOENCODING=utf-8
# config.yaml: data_dir = folder holding the 7 challenge TSVs (flat), cache_dir, output_dir
python src/prepare.py
python src/normalize.py
python src/block.py --split FIT --frac 0.01 --raw     # re-ranker training rows
python src/block.py --fit-rerank                      # re-ranker (cache/rerank.pkl)
K=10 FLOOR=0.003 DATA=/path/to/tsvs ./run_all.sh      # TRAIN+TEST blocking, features, model, decision, validator
```
Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.
Every script prints timings and peak RAM; `experiments.md` logs every measured result.

## Models and licences
| Component | Licence | Size |
|---|---|---|
| XGBoost (final matcher, GPU) — LightGBM fallback on CPU | Apache-2.0 / MIT | gradient-boosted trees, trained here |
| scikit-learn HistGradientBoosting (blocking re-ranker), TF-IDF, isotonic | BSD-3 | trained here |
| rapidfuzz, polars, sparse_dot_topn, indic-transliteration | MIT / MIT / Apache-2.0 / MIT | libraries |

No pretrained neural model is used, so the ≤ 8B-parameter rule is trivially met.
