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
| 5 | `src/train.py` | LightGBM binary classifier + isotonic calibration |
| 6 | `src/decide.py` | owner exclusivity + expected-F0.5 / threshold rule tuned on VAL; writes the two TSVs |
| – | `src/score.py` | the competition metric (self-tests: `python src/score.py`) |

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
| LightGBM (final matcher) | MIT | gradient-boosted trees, trained here |
| scikit-learn HistGradientBoosting (blocking re-ranker), TF-IDF, isotonic | BSD-3 | trained here |
| rapidfuzz, polars, sparse_dot_topn, indic-transliteration | MIT / MIT / Apache-2.0 / MIT | libraries |

No pretrained neural model is used, so the ≤ 8B-parameter rule is trivially met.
