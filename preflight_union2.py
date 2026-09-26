import hashlib
import os
import re
import subprocess
import time

import polars as pl


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


LOCAL = "c812207ed91f37a669b3da5e536202002d2741ae4c0c04664b7cbe10bdb7f11f"
B = "/content/ber"
print("=== PRE-FLIGHT", time.strftime("%H:%M:%S"), "UTC")
print("1. exp_union.py")
for p in ("/content/exp_union.py", f"{B}/src/exp_union.py"):
    print(f"   {p}: {os.path.getsize(p)} bytes, sha256 {sha(p)}  matches local: {sha(p) == LOCAL}")
src = open(f"{B}/src/exp_union.py").read()
print("2. pinned production folds:", all(k in src for k in ("ProductionFolds", "fold_map.parquet", "S.GroupKFold = ProductionFolds")),
      "| GroupKFold on fit rows of the ORIGINAL set only:", "fit0 = allf_old.filter" in src)
print("   lines:", [l.strip()[:70] for l in src.splitlines() if "ProductionFolds" in l or "fold_map" in l][:4])
print("3. reference / output_full checksums")
ref = f"{B}/exp/final/pred_TRAIN__NEW3_NEW4.parquet"
print("   reference preds md5:", sh(f"md5sum {ref}").split()[0], "| Drive copy md5:", sh("md5sum /content/drive/MyDrive/ber_ckpt/exp/final/pred_TRAIN__NEW3_NEW4.parquet 2>&1").split()[0])
print("   output_full now   :", sh(f"cd {B} && md5sum output_full/matching_results.tsv output_full/candidate_pairs.tsv").replace("\n", " | ").replace("  output_full/", " "))
print("   output_full before:", sh("head -2 /content/union_md5_before.txt").replace("\n", " | ").replace("  output_full/", " "))
print("4. python jobs:")
print("  ", sh("ps aux | grep -E 'python' | grep -v grep | awk '{print $2, $3\"%cpu\", $4\"%mem\", $11, $12, $13, $14}'").replace("\n", "\n   "))
print("   log tail:", sh("tail -c 500 /content/exp_union.log").replace("\n", " ⏎ ")[-500:])
print("5/6. union exactness (label-free key counts) ...")
old = pl.read_parquet(f"{B}/cache/candidates_TRAIN.parquet", columns=["s1_id", "cand_id", "source"])
rev = pl.read_parquet(f"{B}/exp/dense/rev_train.parquet").with_columns([pl.col(c).cast(old.schema[c]) for c in ("s1_id", "cand_id", "source")])
k1 = rev.filter(pl.col("rrank") == 0).select("s1_id", "cand_id", "source")
new = k1.join(old, on=["s1_id", "cand_id", "source"], how="anti")
n_s1 = old["s1_id"].n_unique()
print(f"   current rows {old.height:,} ({old.height / n_s1:.2f}/S1 over {n_s1:,} train S1) | reverse-dense k1 rows {k1.height:,} (unique {k1.unique().height:,}) | new rows {new.height:,}")
print(f"   EXPECTED union rows {old.height + new.height:,} = {(old.height + new.height) / n_s1:.3f} candidates/S1 | no duplicates: {pl.concat([old, new]).unique().height == old.height + new.height}")
val = pl.read_parquet(f"{B}/cache/splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id")
gt = pl.read_parquet(f"{B}/cache/gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source").join(val, on="s1_id", how="semi")
ov, nv = old.join(val, on="s1_id", how="semi"), new.join(val, on="s1_id", how="semi")
gt = gt.with_columns([pl.col(c).cast(old.schema[c]) for c in ("s1_id", "cand_id", "source")])
hit = gt.join(pl.concat([ov, nv]), on=["s1_id", "cand_id", "source"], how="semi").height
nq = ov["s1_id"].n_unique()
print(f"   VAL: {ov.height:,} current + {nv.height:,} new = {ov.height + nv.height:,} rows over {nq:,} queried S1 -> {(ov.height + nv.height) / nq:.2f} candidates/S1 (sweep 11.47) | recall {hit / gt.height:.4f} (sweep 0.9861)")
print("   (script re-asserts all of this on the real union rows and aborts on any mismatch)")
