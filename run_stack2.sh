#!/bin/bash
# Queued after run_evening.sh: number-set + qualifier features experiment (VAL only), then Drive backup.
until grep -q DONE /content/evening_status.log 2>/dev/null; do sleep 30; done
cp /content/exp_stack.py /content/exp_stack2.py /content/ber/src/
cd /content/ber
python -u src/exp_stack2.py > /content/exp_stack2.log 2>&1; echo "rc=$?" >> /content/exp_stack2.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack2.log
