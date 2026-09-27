#!/bin/bash
# Diagnostic-only debug rerun of the frozen FWD4 ctl1 TEST build (--from-cache). New log names; stderr (full traceback) goes to its own file; stops at the first failure.
cd /content/ber; L=/content/ctl1e
cp /content/exp_fwd_test_ctl1e.py /content/audit_test_build.py src/
sha256sum src/exp_fwd_test_ctl1e.py > ${L}_script_sha256.txt
( cd exp/union_fwd4_ctl1 && sha256sum * ) > ${L}_model_before.txt
sha256sum exp/dense/*test* cache/candidates_TEST.parquet cache/features_TEST.parquet > ${L}_cache_before.txt
{ md5sum output_full/*.tsv; sha256sum output_full/*.tsv; } > ${L}_full_before.txt
cmp -s ${L}_full_before.txt /content/ctl1c_full_before.txt && echo "output_full pre-run identical to earlier pre-run record" > ${L}_status.txt || echo "output_full DIFFERS from earlier pre-run record" > ${L}_status.txt
cmp -s ${L}_model_before.txt /content/ctl1c_model_before.txt && echo "model hashes identical to earlier pre-run record" >> ${L}_status.txt || echo "model hashes DIFFER from earlier record" >> ${L}_status.txt
cmp -s ${L}_cache_before.txt /content/ctl1c_cache_before.txt 2>/dev/null && echo "cache hashes identical to earlier pre-run record" >> ${L}_status.txt || echo "cache hashes: no/different earlier record" >> ${L}_status.txt
stop() { echo "STOPPED at $1" >> ${L}_status.txt; cp ${L}_* /content/drive/MyDrive/ber_ckpt/logs/ 2>/dev/null; echo CTL1E_DONE > /content/ctl1e_done.txt; exit 1; }
PYTHONIOENCODING=utf-8 python3 -u src/exp_fwd_test_ctl1e.py --from-cache > ${L}_test.log 2> ${L}_error.txt || { echo "build rc=$?" >> ${L}_status.txt; stop build; }
echo "build OK" >> ${L}_status.txt
( cd exp/union_fwd4_ctl1 && sha256sum * ) | cmp -s - ${L}_model_before.txt && echo CTL1_MODEL_UNMODIFIED >> ${L}_status.txt || { echo CTL1_MODEL_CHANGED >> ${L}_status.txt; stop model; }
sha256sum exp/dense/*test* cache/candidates_TEST.parquet cache/features_TEST.parquet | cmp -s - ${L}_cache_before.txt && echo CACHE_UNMODIFIED >> ${L}_status.txt || { echo CACHE_CHANGED >> ${L}_status.txt; stop cache; }
python3 /content/validate_submission.py --matching exp/fwd4_test/matching_results.tsv --candidate exp/fwd4_test/candidate_pairs.tsv --test-dir /content/data --check-ids > ${L}_validator.log 2> ${L}_validator_err.txt; echo "validator rc=$?" >> ${L}_validator.log
grep -q "validator rc=0" ${L}_validator.log && echo VALIDATOR_PASS >> ${L}_status.txt || { echo VALIDATOR_FAIL >> ${L}_status.txt; stop validator; }
( cd src && PYTHONIOENCODING=utf-8 python3 -u audit_test_build.py > ${L}_audit.log 2> ${L}_audit_err.txt; echo "audit rc=$?" >> ${L}_audit.log )
grep -q "audit rc=0" ${L}_audit.log && echo AUDIT_PASS >> ${L}_status.txt || { echo AUDIT_FAIL >> ${L}_status.txt; stop audit; }
{ md5sum output_full/*.tsv; sha256sum output_full/*.tsv; } | cmp -s - ${L}_full_before.txt && echo OUTPUT_FULL_UNCHANGED >> ${L}_status.txt || { echo OUTPUT_FULL_CHANGED >> ${L}_status.txt; stop output_full; }
( cd exp/fwd4_test && md5sum matching_results.tsv candidate_pairs.tsv && sha256sum matching_results.tsv candidate_pairs.tsv && ls -l matching_results.tsv candidate_pairs.tsv ) > ${L}_final_checksums.txt
mkdir -p /content/drive/MyDrive/ber_ckpt/exp; rsync -a exp/fwd4_test/ /content/drive/MyDrive/ber_ckpt/exp/fwd4_test/ 2>/dev/null
cp ${L}_* /content/drive/MyDrive/ber_ckpt/logs/ 2>/dev/null
echo ALL_CHECKS_PASSED >> ${L}_status.txt; echo CTL1E_DONE > /content/ctl1e_done.txt
