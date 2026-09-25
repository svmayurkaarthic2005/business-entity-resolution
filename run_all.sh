#!/bin/bash
# End-to-end: blocking -> features -> LightGBM -> decision tuning (VAL) -> TEST outputs -> validator.
# Assumes prepare.py + normalize.py + the re-ranker (block.py --fit-rerank) already ran.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
K=${K:-10}                     # candidates kept per (S1, source)
DATA=${DATA:-/content/data}    # folder with the challenge TSVs
python src/block.py --split TRAIN --k "$K"
python src/features.py --split TRAIN
python src/train.py
python src/decide.py
python src/block.py --split TEST --k "$K"
python src/features.py --split TEST
python src/train.py --predict TEST
python src/decide.py --split TEST
python "${VALIDATOR:-utils/validate_submission.py}" --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir "$DATA"
