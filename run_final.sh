#!/bin/bash
# Final ablation table (waits for exp_stack4, which waits for exp_stack3); backs up to Drive.
until grep -q DONE /content/exp_stack4.log 2>/dev/null; do sleep 30; done
cp /content/exp_final.py /content/ber/src/
cd /content/ber
md5sum output_full/*.tsv > /content/full_md5_before3.txt
PYTHONIOENCODING=utf-8 python -u src/exp_final.py > /content/exp_final.log 2>&1; echo "rc=$?" >> /content/exp_final.log
md5sum output_full/*.tsv | cmp -s - /content/full_md5_before3.txt && echo OUTPUT_FULL_UNCHANGED >> /content/exp_final.log
CK=/content/drive/MyDrive/ber_ckpt
rsync -a exp/ $CK/exp/; cp src/*.py $CK/src_latest/; cp /content/*.log /content/*.sh $CK/logs/ 2>/dev/null
echo DONE >> /content/exp_final.log
