#!/bin/bash
# Immediate copy of all current work to Google Drive (run_backup.sh repeats this after the stack experiment ends).
CK=/content/drive/MyDrive/ber_ckpt
mkdir -p $CK/exp $CK/logs $CK/cache_extra $CK/src_latest
cd /content/ber
rsync -a exp/ $CK/exp/
for d in output_full output_france_loose output_france_model_v2; do [ -d $d ] && rsync -a $d $CK/; done
rsync -a src/ $CK/src_latest/
cp /content/*.log /content/*.py /content/*.sh /content/*.md $CK/logs/ 2>/dev/null
cp cache/decide.json cache/decide_france_loose.json cache/pred_TEST_frv2.parquet cache/candidates_VAL.parquet $CK/cache_extra/ 2>/dev/null
ls -la $CK $CK/logs | head -60 > /content/backup_now.log
echo "BACKUP_NOW_DONE $(date)" | tee -a /content/backup_now.log > $CK/logs/backup_now_done.txt
