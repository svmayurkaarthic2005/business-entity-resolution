#!/bin/bash
# stack4 (fixed) -> final ablation -> error budget of the best config is run manually afterwards.
cp /content/exp_stack4.py /content/exp_final.py /content/exp_errbudget.py /content/ber/src/
cd /content/ber
PYTHONIOENCODING=utf-8 python -u src/exp_stack4.py > /content/exp_stack4.log 2>&1; echo "rc=$?" >> /content/exp_stack4.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_stack4.log
