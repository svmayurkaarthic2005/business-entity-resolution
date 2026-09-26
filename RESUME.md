# Resume here (evening, 26 Sep 2026) — deadline 27 Sep 23:59 IST

## State
- Submitted once: `output_full/matching_results.tsv` (md5 4b6a9320...0b40) -> leaderboard **0.951537**. VAL 0.96303.
- **Stacked stage-2 model works** (src/exp_stack.py, models in `exp/stack/`):
  VAL 0.96300 -> **0.96810** (stage2) / **0.96886** (stage2+is_india); held-out gain +0.0052 / +0.0060
  (worst of 20 split-halves +0.0041 / +0.0054). Precision and recall both up.
- Dead ends (don't retry): decision-rule re-tuning, K>10 / adaptive K, France-only loose rule, France location fix.
- Full log of every experiment: `experiments.md`.

## Where things are
- PC: `C:\__MACOSX\ber` (src/, experiments.md, scripts).
- Google Drive `MyDrive/ber_ckpt/`: cache/ (14 GB), cache_extra/, exp/ (stack, country, adaptive_k),
  output_full/, output_france_loose/, output_france_model_v2/, src_latest/ (latest code), logs/ (all logs + scripts).

## Restore a fresh Colab runtime (L4)
1. Open `ber_run.ipynb`, Runtime -> L4 GPU. Run the `drive.mount` cell (last cell) and click Allow.
2. New cell: `!bash /content/drive/MyDrive/ber_ckpt/logs/restore_from_drive.sh`  (~10-15 min, copies 14 GB)
   - Do NOT run colab_recover.sh — it re-runs predict/decide and rewrites output_full.

## Next step
Build the TEST version of the stacked model -> `exp/stack_test/`:
1. stage-1 p1 for TEST = `cache/pred_TEST.parquet` (fold-mean, same as VAL treatment).
2. stage-2 features with the same code as exp_stack.stage2_features but test norm files
   (`test_source{1,2,3}`), rows from `cache/features_TEST.parquet`.
3. Predict with `exp/stack/stage2_fold{0,1,2}.json` and calibrate with `exp/stack/stage2_calib.pkl`
   (dict iso/cols/engine/folds; same layout as cache/calib.pkl).
4. France: use plain stage2 (is_india variant would treat France as US).
5. Decide with production rule (0.5, 0.6, 2.0, 0.15); write matching_results.tsv + candidate_pairs.tsv;
   run `python /content/validate_submission.py --matching ... --candidate ... --test-dir /content/data --check-ids`.
6. Report France pred/S1 & empty rate vs 2.862 / 7.19%; user decides whether to submit (user uploads manually).
