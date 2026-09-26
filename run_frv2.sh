#!/bin/bash
# France location-fix experiment on Colab (detached, survives terminal/cell disconnects).
cd /content/ber || exit 1
cp /content/exp_fr_v2.py src/
md5sum output_full/*.tsv > /content/full_md5_before.txt
L=/content/exp_fr_v2.log
PYTHONIOENCODING=utf-8 python -u src/exp_fr_v2.py > $L 2>&1 &&
python /content/validate_submission.py --matching output_france_model_v2/matching_results.tsv \
  --candidate output_france_model_v2/candidate_pairs.tsv --test-dir /content/data --check-ids >> $L 2>&1
md5sum output_full/*.tsv | cmp -s - /content/full_md5_before.txt && echo OUTPUT_FULL_UNCHANGED >> $L
echo DONE >> $L
