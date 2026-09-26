#!/bin/bash
# Evening restore: rebuild /content/ber from Google Drive WITHOUT running any pipeline stage
# (colab_recover.sh re-runs decide/predict and rewrites output_full -- do NOT use it for resuming).
# Prereq: Drive mounted at /content/drive (run the notebook's drive.mount cell, click Allow).
# Usage (notebook cell):  !bash /content/drive/MyDrive/ber_ckpt/logs/restore_from_drive.sh
set -e
CK=/content/drive/MyDrive/ber_ckpt
B=/content/ber
mkdir -p $B/src $B/cache $B/exp /content/data
rsync -a $CK/cache/ $B/cache/
cp $CK/cache_extra/*.parquet $CK/cache_extra/*.json $B/cache/ 2>/dev/null || true
for d in output_full output_baseline output_france_loose output_france_model_v2; do [ -d $CK/$d ] && rsync -a $CK/$d $B/; done
rsync -a $CK/exp/ $B/exp/
cp $CK/src_latest/*.py $B/src/
cp $CK/logs/*.sh $CK/logs/validate_submission.py /content/ 2>/dev/null || true
cat > $B/config.yaml <<EOF
data_dir: /content/data
cache_dir: $B/cache
output_dir: $B/output
seed: 42
val_frac: 0.05
EOF
pip -q install polars==1.44.2 rapidfuzz==3.14.6 indic-transliteration==2.3.82 sparse_dot_topn==1.2.0 psutil pyyaml
if [ ! -f /content/data/test_source3.tsv ]; then
  wget -q -O /content/data.zip https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip
  unzip -q -o -j /content/data.zip 'student_resource/dataset/*/*.tsv' -d /content/data
fi
[ -f /content/validate_submission.py ] || unzip -q -o -j /content/data.zip 'student_resource/utils/validate_submission.py' -d /content
cd $B && md5sum output_full/matching_results.tsv
echo "RESTORED: $(ls cache | wc -l) cache files, exp: $(ls exp | tr '\n' ' ')"
