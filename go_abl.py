import hashlib
import os
import subprocess
import time


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


src = open("/content/exp_union_abl.py").read()
assert "ABL_TAG" in src and "ProductionFolds" in src, "exp_union_abl.py on disk is not the ablation version"
print(time.strftime("%H:%M:%S"), "UTC | exp_union_abl.py", len(src), "bytes sha256", hashlib.sha256(src.encode()).hexdigest())
print(sh("cp /content/exp_union_abl.py /content/ber/src/ && cd /content/ber && python3 -m py_compile src/exp_union_abl.py && echo COMPILED"))
print("python jobs:", sh("ps aux | grep 'src/exp_' | grep -v grep | wc -l"), "| output_full:", sh("cd /content/ber && md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv | cut -c1-32").replace("\n", " "))
lines = [
    "cd /content/ber",
    "md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv src/exp_union_abl.py src/exp_union.py exp/final/pred_TRAIN__NEW3_NEW4.parquet exp/union/pred_TRAIN_union_rev_k1.parquet > /content/abl_md5_before.txt",
    "(ABL_TAG=B ABL_UNIVERSE=queried ABL_DENSE=0 python3 -u src/exp_union_abl.py > /content/exp_abl_B.log 2>&1; echo EXIT=$? >> /content/exp_abl_B.log;"
    " ABL_TAG=D ABL_UNIVERSE=queried ABL_DENSE=1 python3 -u src/exp_union_abl.py > /content/exp_abl_D.log 2>&1; echo EXIT=$? >> /content/exp_abl_D.log) &",
    "(ABL_TAG=C ABL_UNIVERSE=full ABL_DENSE=0 python3 -u src/exp_union_abl.py > /content/exp_abl_C.log 2>&1; echo EXIT=$? >> /content/exp_abl_C.log) &",
    "wait",
    "md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv > /content/abl_md5_after.txt",
    "head -2 /content/abl_md5_before.txt | diff - /content/abl_md5_after.txt && echo OUTPUT_FULL_UNCHANGED > /content/abl_done.txt || echo OUTPUT_FULL_CHANGED > /content/abl_done.txt",
    "mkdir -p /content/drive/MyDrive/ber_ckpt/logs /content/drive/MyDrive/ber_ckpt/exp",
    "cp /content/exp_abl_*.log /content/abl_md5_*.txt /content/abl_done.txt /content/drive/MyDrive/ber_ckpt/logs/",
    "for t in B C D; do rsync -a --exclude allf_union.parquet /content/ber/exp/union_abl_$t /content/drive/MyDrive/ber_ckpt/exp/; done",
    "echo ALL_DONE >> /content/abl_done.txt",
]
open("/content/run_abl.sh", "w").write("\n".join(lines) + "\n")
subprocess.Popen(["bash", "/content/run_abl.sh"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4)
print("LAUNCHED B,D (sequential) and C (parallel); exp jobs now:", sh("ps aux | grep 'src/exp_union_abl' | grep -v grep | wc -l"))
