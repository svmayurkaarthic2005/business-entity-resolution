set -e
cd /content/ber; export PYTHONIOENCODING=utf-8
python src/block.py --split TEST --k 10 --floor 0.003
test -f cache/candidates_TEST.parquet
cp cache/decide_baseline.json cache/decide.json
python src/decide.py --split TEST --from-candidates candidates_TEST.parquet
python /content/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir /content/data
mkdir -p output_baseline && cp output/*.tsv output_baseline/
echo BASELINE_DONE
python src/block.py --split TRAIN --frac 0.3 --k 10 --floor 0.003
mv cache/candidates_TRAIN_frac.parquet cache/candidates_TRAIN.parquet
python src/features.py --split TRAIN --fit-frac 0.5
python src/train.py
python src/decide.py
python src/features.py --split TEST
python src/predict.py TEST
python src/decide.py --split TEST
python /content/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir /content/data
mkdir -p output_full && cp output/*.tsv output_full/
echo FULL_DONE
