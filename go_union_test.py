import hashlib
import subprocess
import time


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


src = open("/content/exp_union_test.py").read()
assert "union_frmap_test" in src and "fit_tables" in src, "exp_union_test.py on disk is not the TEST-build version"
print(time.strftime("%H:%M:%S"), "UTC | exp_union_test.py", len(src), "bytes sha256", hashlib.sha256(src.encode()).hexdigest())
print(sh("cp /content/exp_union_test.py /content/ber/src/ && cd /content/ber && python3 -m py_compile src/exp_union_test.py && echo COMPILED"))
print("inputs:", sh("cd /content/ber && ls exp/union/u_union_fold0.json exp/union/u_union_calib.pkl exp/union/fold_map.parquet exp/union/allf_union.parquet "
                    "exp/dense/rev_test.parquet exp/dense/emb_test_s1.npy cache/candidates_TEST.parquet cache/features_TEST.parquet 2>&1"))
print("python jobs:", sh("ps aux | grep 'src/exp_' | grep -v grep | wc -l"), "| output_full:",
      sh("cd /content/ber && md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv | cut -c1-32").replace("\n", " "))
V = "python3 /content/validate_submission.py --test-dir /content/data --check-ids"
lines = [
    "cd /content/ber",
    "md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv cache/candidates_TEST.parquet > /content/ut_md5_before.txt",
    "python3 -u src/exp_union_test.py > /content/exp_union_test.log 2>&1; echo EXIT=$? >> /content/exp_union_test.log",
    f"{V} --matching exp/union_test/matching_results.tsv --candidate exp/union_test/candidate_pairs.tsv >> /content/exp_union_test.log 2>&1",
    "python3 -u src/exp_union_test.py --from-cache --fr-map > /content/exp_union_frmap_test.log 2>&1; echo EXIT=$? >> /content/exp_union_frmap_test.log",
    f"{V} --matching exp/union_frmap_test/matching_results.tsv --candidate exp/union_frmap_test/candidate_pairs.tsv >> /content/exp_union_frmap_test.log 2>&1",
    "md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv cache/candidates_TEST.parquet > /content/ut_md5_after.txt",
    "diff /content/ut_md5_before.txt /content/ut_md5_after.txt && echo OUTPUT_FULL_UNCHANGED > /content/ut_done.txt || echo OUTPUT_FULL_CHANGED > /content/ut_done.txt",
    "mkdir -p /content/drive/MyDrive/ber_ckpt/logs /content/drive/MyDrive/ber_ckpt/exp",
    "cp /content/exp_union_test.log /content/exp_union_frmap_test.log /content/ut_md5_*.txt /content/ut_done.txt /content/drive/MyDrive/ber_ckpt/logs/",
    "rsync -a --exclude allf_union_test.parquet /content/ber/exp/union_test /content/ber/exp/union_frmap_test /content/drive/MyDrive/ber_ckpt/exp/",
    "rsync -a /content/ber/exp/dense/rev_test.parquet /content/ber/exp/dense/fwd_test.parquet /content/drive/MyDrive/ber_ckpt/exp/dense/ 2>/dev/null",
    "echo ALL_DONE >> /content/ut_done.txt",
]
open("/content/run_union_test.sh", "w").write("\n".join(lines) + "\n")
subprocess.Popen(["bash", "/content/run_union_test.sh"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)
print("LAUNCHED union TEST build (plain, then --fr-map from cache); jobs now:", sh("ps aux | grep 'src/exp_union_test' | grep -v grep | wc -l"))
