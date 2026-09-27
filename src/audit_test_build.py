"""Read-only audit of the frozen-FWD4 TEST build: coverage, ids, duplicates, contamination, schema, checksums, provenance."""
import hashlib, sys
from pathlib import Path
import polars as pl
from common import CACHE
N = CACHE / "norm"; EXP = CACHE.parent / "exp"
D = EXP / "fwd4_test"; M, C = D / "matching_results.tsv", D / "candidate_pairs.tsv"
ok = True
def chk(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

# --- schema / format (raw bytes)
for f, col in ((M, "matched_entity_ids"), (C, "candidate_entity_ids")):
    b = f.read_bytes(); lines = b.split(b"\n")
    chk(f"{f.name} header", lines[0] == f"source1_entity_id\t{col}".encode(), lines[0][:80])
    chk(f"{f.name} ends with newline, no CR, no quotes, ascii", b.endswith(b"\n") and b"\r" not in b and b'"' not in b and b.isascii())
    chk(f"{f.name} every row exactly 2 tab fields", all(l.count(b"\t") == 1 for l in lines[:-1]))
    print(f"CHECKSUM {f.name} size {len(b):,} sha256 {hashlib.sha256(b).hexdigest()} md5 {hashlib.md5(b).hexdigest()}", flush=True)

t1 = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"])
tr1 = pl.read_parquet(CACHE / "train_source1.parquet", columns=["entity_id"])
te = {s: pl.read_parquet(CACHE / f"test_source{s}.parquet", columns=["entity_id"])["entity_id"] for s in (2, 3)}
tr = {s: pl.read_parquet(CACHE / f"train_source{s}.parquet", columns=["entity_id"])["entity_id"] for s in (2, 3)}
m = pl.read_csv(M, separator="\t", quote_char=None, infer_schema_length=0)
c = pl.read_csv(C, separator="\t", quote_char=None, infer_schema_length=0)

# --- S1 coverage exactly once
for nm, x in (("matching", m), ("candidate", c)):
    chk(f"{nm}: rows == test S1 ({t1.height:,})", x.height == t1.height, x.height)
    chk(f"{nm}: S1 ids unique", x["source1_entity_id"].n_unique() == x.height)
    chk(f"{nm}: S1 id set == test_source1", set(x["source1_entity_id"]) == set(t1["entity_id"]))
    chk(f"{nm}: same order as test_source1", x["source1_entity_id"].to_list() == t1["entity_id"].to_list())

def explode(x, col):
    return (x.with_columns(pl.col(col).fill_null("")).filter(pl.col(col) != "")
            .with_columns(pl.col(col).str.split(",")).explode(col).rename({col: "eid"}))
me, ce = explode(m, "matched_entity_ids"), explode(c, "candidate_entity_ids")
print(f"pairs: matching {me.height:,} | candidates {ce.height:,}")
# --- valid ids
allte = set(te[2]) | set(te[3])
for nm, x in (("matching", me), ("candidate", ce)):
    bad = x.filter(~pl.col("eid").is_in(list(allte)))
    chk(f"{nm}: all ids are test S2/S3 entity ids", bad.height == 0, bad.height)
    chk(f"{nm}: ids start with S2-/S3-", x["eid"].str.contains(r"^S[23]-").all())
    # --- duplicates
    d = x.group_by("source1_entity_id", "eid").len().filter(pl.col("len") > 1)
    chk(f"{nm}: no duplicate (S1, id) pair", d.height == 0, d.height)
# --- candidate coverage: matches subset of candidates
j = me.join(ce.with_columns(inc=pl.lit(True)), on=["source1_entity_id", "eid"], how="left")
chk("matching pairs all present in candidate_pairs", j["inc"].fill_null(False).all(), j.filter(pl.col("inc").is_null()).height)
# owner exclusivity (train fact) - informational
own = me.group_by("eid").agg(pl.col("source1_entity_id").n_unique().alias("n"))
print(f"INFO records matched to >1 S1 in matching: {own.filter(pl.col('n') > 1).height:,} of {own.height:,}")
# --- contamination
chk("test S1 ids disjoint from train S1 ids", not set(t1["entity_id"]) & set(tr1["entity_id"]))
for s in (2, 3):
    chk(f"test S{s} ids disjoint from train S{s} ids", not set(te[s]) & set(tr[s]))
trall = set(tr[2]) | set(tr[3])
chk("no train S2/S3 id anywhere in outputs", not (set(me["eid"]) | set(ce["eid"])) & trall)
fit = pl.read_parquet(EXP / "union_fwd4_ctl1" / "fold_map.parquet")
print("INFO fold_map columns", fit.columns, "rows", fit.height)
# S1 ids in training tables are train S1 row ids: verify fold_map s1 ids < train S1 count
chk("model fit rows come from TRAIN S1 only (fold_map ids within train S1 range)", fit.select(pl.col(fit.columns[0]).max()).item() < tr1.height, fit.select(pl.col(fit.columns[0]).max()).item())
# --- per-country predicted counts
cc = m.join(t1, left_on="source1_entity_id", right_on="entity_id").with_columns(n=pl.col("matched_entity_ids").fill_null("").str.count_matches(",") + (pl.col("matched_entity_ids").fill_null("") != "").cast(pl.Int32))
for r in cc.group_by("country").agg(rows=pl.len(), pred=pl.col("n").sum(), empty=(pl.col("n") == 0).mean()).sort("country").iter_rows():
    print(f"COUNTRY {r[0]:7s} S1 {r[1]:,} matches {r[2]:,} ({r[2]/r[1]:.3f}/S1) empty-S1 {r[3]:.2%}")
print("AUDIT", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
