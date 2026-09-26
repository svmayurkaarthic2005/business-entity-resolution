#!/bin/bash
# Fresh runtime: restore from Drive, build the stacked TEST submission in exp/stack_test/, validate, back up.
S=/content/evening_status.log; : > $S
echo "$(date +%T) restore start" >> $S
bash /content/drive/MyDrive/ber_ckpt/logs/restore_from_drive.sh > /content/restore.log 2>&1 || { echo "restore FAILED" >> $S; exit 1; }
echo "$(date +%T) restore done: $(tail -1 /content/restore.log)" >> $S
cp /content/exp_stack.py /content/exp_stack_test.py /content/ber/src/
cd /content/ber
md5sum output_full/*.tsv > /content/full_md5_before.txt
echo "$(date +%T) stack_test start" >> $S
PYTHONIOENCODING=utf-8 python -u src/exp_stack_test.py > /content/exp_stack_test.log 2>&1; echo "rc=$?" >> /content/exp_stack_test.log
python /content/validate_submission.py --matching exp/stack_test/matching_results.tsv --candidate exp/stack_test/candidate_pairs.tsv \
  --test-dir /content/data --check-ids >> /content/exp_stack_test.log 2>&1
md5sum output_full/*.tsv | cmp -s - /content/full_md5_before.txt && echo OUTPUT_FULL_UNCHANGED >> /content/exp_stack_test.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo "$(date +%T) DONE" >> $S
