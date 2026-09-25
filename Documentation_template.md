# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TODO
**Team Members:** TODO
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A three-stage, fully local pipeline: (1) **normalisation whose dictionaries are learned from the
training pairs** (Indic→Latin transliteration, address aliases, noise/legal tokens); (2) **multi-channel
sparse blocking with a learned re-ranker**, which keeps ~11 candidates per Source-1 entity at 97.4 %
pair recall; (3) a **LightGBM matcher with owner-competition features** followed by a decision layer
tuned directly for macro F0.5 (owner exclusivity + per-entity expected-F0.5 selection).

---

## 2. Methodology

### 2.1 Problem Analysis (measured on the training data)
- 2.2 M Source-1 (S1) entities; 5.6 % singletons; 3.46 true matches per S1 on average (1.67 S2, 1.79 S3).
- **Every S2/S3 record has exactly one S1 owner** (0 of 7.64 M matched records has two owners) and
  **every true pair shares the country string** → per-country blocking and one-owner exclusivity are safe.
- ~26 % of S2/S3 records match nothing (pure distractors).
- Names: legal-suffix churn, typos inside words and suffixes ("prinvae"), DBA/website names,
  word reordering, ~9–11 % of S2/S3 names in Indic scripts (Devanagari, Telugu, Tamil, Kannada,
  Gujarati, Bengali, Malayalam, Gurmukhi, Odia).
- Addresses: ~3 % empty in S2/S3, component reordering, abbreviations, native-script state names,
  wrong/noisy city names; postcodes almost absent (< 1.5 %).
- Test adds France (15 % of test S1) with no training labels: generic names ("École", "Amicale"),
  French abbreviations (R, AV, BD, N°).

### 2.2 Solution Strategy
**Approach type:** Blocking + learned re-ranker + gradient-boosted pair classifier + metric-aware decision layer.
**Core innovations:**
1. Normalisation maps learned from aligned training pairs rather than hand-written lists
   (1,312 transliterated tokens cover 93.4 % of Indic test tokens; 296 address aliases; 74 noise tokens).
2. A learned blocking re-ranker that lifts recall@5 per source from 93.9 % (cosine sum) to 96.4 %.
3. Owner-competition features computed with **all** S1 entities queried (TRAIN mirrors TEST), plus
   exclusivity at decision time.
4. The decision rule optimises the exact competition metric (macro F0.5 incl. singletons).

---

## 3. Candidate Generation (Blocking)
Blocking is done **within each country string** (open set; France handled with the same code) and
separately against S2 and S3. Channels:

| Channel | Representation | Retrieval |
|---|---|---|
| name | char_wb 3–4-gram TF-IDF of the core name (legal/noise tokens stripped, Indic transliterated) | sparse top-30 (sparse_dot_topn) |
| name+address words | word TF-IDF of core name + house number + street tokens + city | sparse top-30 |
| address | char_wb 3–4-gram TF-IDF of the normalised address | sparse top-30 |
| exact keys | core name; house no + street token + city; postcode + first name token (blocks ≤ 200) | hash join |

N-grams occurring in > 1 % of records are dropped (8× faster, negligible signal loss). The union
(~97 candidates per S1 and source) gets three exact cosines and rank/max features; a
HistGradientBoosting re-ranker trained on 1 % of FIT entities orders it; we keep the top 10 per
source whose re-ranker probability ≥ 0.003.

- **Blocking keys used:** character/word TF-IDF neighbours of name and address, exact keys above.
- **Candidate pairs generated (test):** TODO total; France 16.6 per S1, VAL 11.0 per S1.
- **Recall (VAL, 21,991 S1):** 97.4 % of true pairs at 11 candidates per S1 (union ceiling 98.5 %).
  Top-5/source: 96.6 % (US 97.0 %, India 96.0 %, Indic-script names 96.4 %, empty-address records 74.6 %).
- **How true matches were not lost:** three complementary channels (name, address, mixed) +
  exact keys; misses were diagnosed (95 % were ranking losses, not retrieval misses), which motivated
  the address channel and the learned re-ranker.

## 4. Matching Model
**Features used:**
- Name: rapidfuzz ratio / token-set / token-sort / partial ratio, Jaro-Winkler, token Jaccard,
  exact core, first-token equality, length ratio, full-name (with suffix) similarity,
  DBA/alternate-name similarity, legal-suffix agreement/conflict, candidate-in-Indic-script flag.
- Address: token-set and ratio of normalised addresses, street-token similarity, house-number
  exact / numeric-part / conflict, city similarity, state agreement, city/state set overlap, empty-address flag.
- Blocking: the three cosines, channel flags, per-S1 maxima and ranks, re-ranker probability, candidate count.
- Competition: best score of the record over all S1, number of S1 claiming it, owner rank,
  gap to the best owner, gap to the S1's best candidate.
- Country is **not** a feature (France is unseen).

**Model type:** LightGBM (binary, 127 leaves, lr 0.05, early stopping) + isotonic calibration.
**Threshold selection:** grid search on VAL for macro F0.5 over (a) owner exclusivity weight,
(b) plain threshold vs (c) per-entity expected-F0.5 prefix selection
(E[F](k) = 1.25·Σp₁..ₖ / (0.25·(Σp + miss) + k), empty list if γ·Π(1−p) is larger).

## 5. Results & Error Analysis
- **Baseline (re-ranker probability, threshold 0.7):** VAL macro F0.5 0.837 (precision 0.876, recall 0.773, singleton accuracy 0.506).
- **F0.5 Score (macro, VAL):** TODO (LightGBM + decision layer)
- **Common false positives:** TODO
- **Common false negatives:** empty-address records with heavily corrupted names; renamed businesses (DBA) with different addresses.

## 6. Conclusion
TODO

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/`: `src/prepare.py → normalize.py → block.py → features.py →
train.py → decide.py`, orchestrated by `run_all.sh` (see README.md). `src/score.py` implements the metric.
### B. Additional Results
See `experiments.md` in the repository for every measured run.
