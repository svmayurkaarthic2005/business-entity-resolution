#!/bin/bash
# Waits for the ablation run, then runs the NEW3 features + stage-3 experiment; backs up to Drive.
until grep -q DONE /content/exp_ablate.log 2>/dev/null; do sleep 30; done
cp /content/exp_stack3.py /content/ber/src/
cd /content/ber
PYTHONIOENCODING=utf-8 python -u src/exp_stack3.py > /content/exp_stack3.log 2>&1; echo "rc=$?" >> /content/exp_stack3.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack3.log
