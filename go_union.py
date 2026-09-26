import os
import subprocess
import time


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


print(time.strftime("%H:%M:%S"), "UTC")
print(sh("cp /content/exp_union.py /content/ber/src/ && cd /content/ber && python3 -m py_compile src/exp_union.py && echo COMPILED exp_union.py"))
CACHE = sh("cd /content/ber/src && python3 -c 'from common import CACHE; print(CACHE)'").splitlines()[-1]
EXP = os.path.join(os.path.dirname(CACHE), "exp")
print("CACHE", CACHE, "| EXP", EXP)
need = [f"{EXP}/final/pred_TRAIN__NEW3_NEW4.parquet", f"{CACHE}/rerank.pkl", f"{CACHE}/calib.pkl", f"{CACHE}/decide.json",
        f"{CACHE}/candidates_TRAIN.parquet", f"{CACHE}/features_TRAIN.parquet", f"{CACHE}/gt_pairs.parquet", f"{CACHE}/splits.parquet",
        f"{EXP}/dense/rev_train.parquet", f"{EXP}/dense/emb_train_s1.npy", f"{EXP}/dense/emb_train_s2.npy", f"{EXP}/dense/emb_train_s3.npy"]
missing = [p for p in need if not os.path.exists(p)]
for p in need:
    print("  ok " if os.path.exists(p) else "  MISSING ", p, os.path.getsize(p) // 10**6 if os.path.exists(p) else "", "MB")
print("tfidf cache:", sh(f"ls {CACHE}/tfidf | tr '\\n' ' '"))
print(sh("df -h /content | tail -1"), "|", sh("free -g | sed -n 2p"))
print("running exp_ jobs:", sh("ps aux | grep 'src/exp_' | grep -v grep | wc -l"))
print("output_full md5:", sh("cd /content/ber && md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv | cut -c1-40"))
if missing:
    print("NOT LAUNCHED: missing", missing)
else:
    lines = [
        "cd /content/ber",
        f"md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv src/exp_union.py src/exp_final.py src/exp_stack.py src/exp_stack2.py src/exp_stack3.py src/exp_stack4.py {CACHE}/decide.json {EXP}/final/pred_TRAIN__NEW3_NEW4.parquet {EXP}/dense/rev_train.parquet > /content/union_md5_before.txt",
        "python3 -u src/exp_union.py > /content/exp_union.log 2>&1",
        "echo EXIT=$? >> /content/exp_union.log",
        "md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv > /content/union_md5_after.txt",
        "head -2 /content/union_md5_before.txt | diff - /content/union_md5_after.txt && echo OUTPUT_FULL_UNCHANGED >> /content/exp_union.log",
        "mkdir -p /content/drive/MyDrive/ber_ckpt/logs /content/drive/MyDrive/ber_ckpt/exp",
        "cp /content/exp_union.log /content/union_md5_before.txt /content/union_md5_after.txt /content/drive/MyDrive/ber_ckpt/logs/",
        f"rsync -a --exclude allf_union.parquet {EXP}/union /content/drive/MyDrive/ber_ckpt/exp/",
        "echo SYNCED >> /content/exp_union.log",
    ]
    open("/content/run_union.sh", "w").write("\n".join(lines) + "\n")
    subprocess.Popen(["bash", "/content/run_union.sh"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    print("LAUNCHED run_union.sh; exp jobs now:", sh("ps aux | grep 'src/exp_union' | grep -v grep | wc -l"))
