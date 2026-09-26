#!/bin/bash
# Error budget + cap check, country-aware model, adaptive-K experiment (all VAL/read-only; output_full untouched).
cd /content/ber || exit 1
export PYTHONIOENCODING=utf-8 RETR_DF=0.005
cp /content/exp_budget.py /content/exp_country.py /content/exp_k.py src/
mkdir -p exp
md5sum output_full/*.tsv > exp/full_md5_before.txt
S=/content/exp3_status.log; : > $S
run() { local n=$1; shift; echo "$(date +%T) start $n" >> $S; python -u "$@" > /content/$n.log 2>&1; local rc=$?; echo "$(date +%T) end $n rc=$rc" >> $S; }
run exp_budget src/exp_budget.py
run exp_country src/exp_country.py
[ -f cache/candidates_VAL.parquet ] || run block_val_k30 src/block.py --split VAL --k 30 --floor 0.003
run exp_k src/exp_k.py
md5sum output_full/*.tsv | cmp -s - exp/full_md5_before.txt && echo OUTPUT_FULL_UNCHANGED >> $S
echo DONE >> $S
