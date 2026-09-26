#!/bin/bash
# 1) TEST build of the stage-2+number+qualifier model + validator, 2) feature-group ablation, each followed by a Drive backup.
cp /content/exp_stack2_test.py /content/exp_ablate.py /content/ber/src/
cd /content/ber
CK=/content/drive/MyDrive/ber_ckpt
md5sum output_full/*.tsv > /content/full_md5_before2.txt
PYTHONIOENCODING=utf-8 python -u src/exp_stack2_test.py > /content/exp_stack2_test.log 2>&1; echo "rc=$?" >> /content/exp_stack2_test.log
python /content/validate_submission.py --matching exp/stack2_test/matching_results.tsv --candidate exp/stack2_test/candidate_pairs.tsv \
  --test-dir /content/data --check-ids >> /content/exp_stack2_test.log 2>&1
md5sum output_full/*.tsv | cmp -s - /content/full_md5_before2.txt && echo OUTPUT_FULL_UNCHANGED >> /content/exp_stack2_test.log
rsync -a exp/ $CK/exp/; cp /content/*.log $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack2_test.log
PYTHONIOENCODING=utf-8 python -u src/exp_ablate.py > /content/exp_ablate.log 2>&1; echo "rc=$?" >> /content/exp_ablate.log
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_ablate.log
