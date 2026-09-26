#!/bin/bash
# Wait for the stacked experiment, then copy all experiment results to Google Drive (survives runtime loss).
CK=/content/drive/MyDrive/ber_ckpt
until grep -q DONE /content/exp_stack.log 2>/dev/null; do sleep 60; done
mkdir -p $CK/exp $CK/logs $CK/cache_extra
cd /content/ber
rsync -a exp/ $CK/exp/
for d in output_france_loose output_france_model_v2; do [ -d $d ] && rsync -a $d $CK/; done
cp /content/*.log /content/*.py /content/*.sh $CK/logs/ 2>/dev/null
cp cache/decide_france_loose.json cache/pred_TEST_frv2.parquet cache/candidates_VAL.parquet $CK/cache_extra/ 2>/dev/null
echo "BACKUP_DONE $(date)" > $CK/logs/backup_done.txt
