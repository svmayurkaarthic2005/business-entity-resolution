#!/bin/bash
# Phase 1 blocking-ceiling experiment; waits for the stacked TEST build, pauses the queued stack2 job, then resumes it.
until grep -q DONE /content/evening_status.log 2>/dev/null; do sleep 20; done
pkill -f run_stack2.sh
cp /content/exp_block2.py /content/ber/src/
cd /content/ber
PYTHONIOENCODING=utf-8 python -u src/exp_block2.py > /content/exp_block2.log 2>&1; echo "rc=$?" >> /content/exp_block2.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_block2.log
