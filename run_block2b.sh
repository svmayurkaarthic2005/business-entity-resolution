#!/bin/bash
# Empty-address top-k sweep (~2 min), then the paused number-set/qualifier feature experiment, then Drive backup.
cp /content/exp_block2b.py /content/exp_block2.py /content/exp_stack.py /content/exp_stack2.py /content/ber/src/
cd /content/ber
PYTHONIOENCODING=utf-8 python -u src/exp_block2b.py > /content/exp_block2b.log 2>&1; echo "rc=$?" >> /content/exp_block2b.log
echo DONE >> /content/exp_block2b.log
PYTHONIOENCODING=utf-8 python -u src/exp_stack2.py > /content/exp_stack2.log 2>&1; echo "rc=$?" >> /content/exp_stack2.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack2.log
