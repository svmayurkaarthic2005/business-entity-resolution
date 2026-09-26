#!/bin/bash
# Waits for exp_stack3 (which caches exp/stack3/feats_TRAIN.parquet), then runs the ambiguity-feature + stage-3 experiment.
until grep -q DONE /content/exp_stack3.log 2>/dev/null; do sleep 30; done
cp /content/exp_stack4.py /content/ber/src/
cd /content/ber
PYTHONIOENCODING=utf-8 python -u src/exp_stack4.py > /content/exp_stack4.log 2>&1; echo "rc=$?" >> /content/exp_stack4.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack4.log
