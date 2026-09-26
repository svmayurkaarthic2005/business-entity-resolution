import polars as pl

B = "/content/ber"
K = ["s1_id", "cand_id", "source"]
old = pl.read_parquet(f"{B}/cache/candidates_TRAIN.parquet", columns=K)
rev = pl.read_parquet(f"{B}/exp/dense/rev_train.parquet").with_columns([pl.col(c).cast(old.schema[c]) for c in K])
k1 = rev.filter(pl.col("rrank") == 0).select(K)
new = k1.join(old, on=K, how="anti")
q = pl.read_parquet(f"{B}/cache/queried_TRAIN.parquet").select("s1_id").with_columns(pl.col("s1_id").cast(old.schema["s1_id"]))
val = pl.read_parquet(f"{B}/cache/splits.parquet").filter(pl.col("rand") == "VAL").select("s1_id").with_columns(pl.col("s1_id").cast(old.schema["s1_id"]))
vq = val.join(q, on="s1_id", how="semi")
print(f"train S1 in candidates_TRAIN ({q.height:,} queried S1); VAL S1 total {val.height:,}, VAL S1 that are queried {vq.height:,}")
new_q = new.join(q, on="s1_id", how="semi")
print(f"new rev-k1 rows: {new.height:,} in total | for QUERIED S1 {new_q.height:,} | for S1 that are NOT queried {new.height - new_q.height:,}  (these enter only the owner-competition features)")
print(f"queried-S1 union: {old.height:,} + {new_q.height:,} = {old.height + new_q.height:,} rows = {(old.height + new_q.height) / q.height:.3f} candidates/S1 (current {old.height / q.height:.3f})")
gt = (pl.read_parquet(f"{B}/cache/gt_pairs.parquet").select("s1_id", cand_id="match_id", source="match_source")
      .with_columns([pl.col(c).cast(old.schema[c]) for c in K]).join(vq, on="s1_id", how="semi"))
ov, nv = old.join(vq, on="s1_id", how="semi"), new.join(vq, on="s1_id", how="semi")
hit0, hit1 = gt.join(ov, on=K, how="semi").height, gt.join(pl.concat([ov, nv]), on=K, how="semi").height
print(f"VAL (queried, {vq.height:,} S1): current {ov.height:,} rows {ov.height / vq.height:.2f}/S1 recall {hit0 / gt.height:.4f} | + new {nv.height:,} -> {(ov.height + nv.height) / vq.height:.2f}/S1 recall {hit1 / gt.height:.4f}   (sweep: 11.46 -> 11.47 / 0.9713 -> 0.9861)")
