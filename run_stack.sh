#!/bin/bash
# Stacked-feature experiment (VAL only; output_full untouched).
cd /content/ber || exit 1
export PYTHONIOENCODING=utf-8
cp /content/exp_stack.py src/
md5sum output_full/*.tsv > exp/full_md5_before_stack.txt
python -u src/exp_stack.py > /content/exp_stack.log 2>&1; echo "rc=$?" >> /content/exp_stack.log
md5sum output_full/*.tsv | cmp -s - exp/full_md5_before_stack.txt && echo OUTPUT_FULL_UNCHANGED >> /content/exp_stack.log
echo DONE >> /content/exp_stack.log
bash /content/run_backup_now.sh
