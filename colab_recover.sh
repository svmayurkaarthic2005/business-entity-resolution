#!/bin/bash
# Colab driver with Google-Drive checkpoints: every stage is skipped when its output already
# exists (locally or restored from Drive), and each finished stage is copied back to Drive.
# Usage (Colab terminal, after mounting Drive and uploading src/*.py + this file to /content):
#   nohup bash /content/colab_recover.sh > /content/recover.log 2>&1 &
set -e
export PYTHONIOENCODING=utf-8 RETR_DF=${RETR_DF:-0.005}
CK=/content/drive/MyDrive/ber_ckpt
B=/content/ber
URL=https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip
mkdir -p "$CK" "$B/src" "$B/cache" /content/data
save() { echo "[ckpt] $(date +%T) saving: $*"; mkdir -p "$CK/cache"; rsync -a "$B/cache/" "$CK/cache/"; \
         for d in output output_baseline output_full; do [ -d "$B/$d" ] && rsync -a "$B/$d" "$CK/"; done; true; }
stage() { echo; echo "===== $(date +%T) $*"; }

cp /content/*.py "$B/src/" 2>/dev/null || true   # code uploaded to /content via the Files panel
cat > "$B/config.yaml" <<EOF
data_dir: /content/data
cache_dir: $B/cache
output_dir: $B/output
seed: 42
val_frac: 0.05
EOF
[ -d "$CK/cache" ] && { stage "restore from Drive"; rsync -a "$CK/cache/" "$B/cache/"; }
for d in output output_baseline output_full; do [ -d "$CK/$d" ] && rsync -a "$CK/$d" "$B/"; done
pip -q install polars==1.44.2 rapidfuzz==3.14.6 indic-transliteration==2.3.82 sparse_dot_topn==1.2.0 psutil pyyaml
cd "$B"

if [ ! -f /content/data/test_source3.tsv ]; then
  stage "download official data"
  wget -q -O /content/data.zip "$URL"
  unzip -q -o -j /content/data.zip 'student_resource/dataset/*/*.tsv' -d /content/data
fi
[ -f /content/validate_submission.py ] || unzip -q -o -j /content/data.zip 'student_resource/utils/validate_submission.py' -d /content

[ -f cache/splits.parquet ] || { stage prepare; python src/prepare.py | tail -3; save prepare; }
[ -f cache/norm/test_source3.parquet ] || { stage normalize; python src/normalize.py 2>&1 | grep -v Warn | tail -4; save normalize; }
if [ ! -f cache/rerank.pkl ]; then
  stage "re-ranker"
  python src/block.py --split FIT --frac 0.01 --raw 2>&1 | grep -v Warn | tail -3
  python src/block.py --fit-rerank 2>&1 | grep -v Warn | tail -4
  save rerank
fi
[ -f cache/candidates_TEST.parquet ] || { stage "TEST blocking"; python src/block.py --split TEST --k 10 --floor 0.003 2>&1 | grep -v Warn; save test-blocking; }
if [ ! -f output_baseline/matching_results.tsv ]; then   # safety-net file (not to be submitted unless asked)
  stage "baseline output"
  echo '{"cfg": {"rule": "thr", "alpha": 0.5, "t": 0.7}}' > cache/decide.json
  python src/decide.py --split TEST --from-candidates candidates_TEST.parquet 2>&1 | grep -v Warn
  mkdir -p output_baseline && cp output/*.tsv output_baseline/ && save baseline
fi
if [ ! -f cache/candidates_TRAIN.parquet ]; then
  stage "TRAIN blocking (30% of train S1)"
  python src/block.py --split TRAIN --frac 0.3 --k 10 --floor 0.003 2>&1 | grep -v Warn
  mv cache/candidates_TRAIN_frac.parquet cache/candidates_TRAIN.parquet; save train-blocking
fi
[ -f cache/features_TRAIN.parquet ] || { stage "features TRAIN"; python src/features.py --split TRAIN --fit-frac 0.5 2>&1 | grep -v Warn; save features-train; }
if [ ! -f cache/pred_TRAIN.parquet ]; then
  stage "train (GPU XGBoost, 3-fold)"; T0=$(date +%s)
  python src/train.py 2>&1 | grep -v Warn
  echo "TRAINING_SECONDS $(( $(date +%s) - T0 ))"; save train
fi
stage "decision tuning on VAL"; python src/decide.py 2>&1 | grep -v Warn; save decide
[ -f cache/features_TEST.parquet ] || { stage "features TEST"; python src/features.py --split TEST 2>&1 | grep -v Warn; save features-test; }
stage "predict TEST"; python src/predict.py TEST 2>&1 | grep -v Warn
stage "write TEST outputs"; python src/decide.py --split TEST 2>&1 | grep -v Warn
python /content/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir /content/data --check-ids
mkdir -p output_full && cp output/*.tsv output_full/ && save full
echo FULL_DONE
