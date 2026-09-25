"""Phase 1: TSV -> parquet, ground-truth pairs, EDA facts, validation splits.

Usage:  python src/prepare.py            # full data
        python src/prepare.py --sample 50000   # first N rows per file (smoke test)
Outputs (in cache/, or cache/sample/ with --sample):
  <file>.parquet   id(int32 row index), entity_id, name, address, country
                   -> also the id <-> entity_id mapping
  gt_pairs.parquet s1_id, match_id, match_source (2 or 3)   (ids = row index in that file)
  gt_s1.parquet    s1_id, n_matches                           (all train S1, incl. singletons)
  splits.parquet   s1_id, country, rand ('FIT'/'VAL'), loco ('FIT'/'VAL'/'')
  eda.json         facts printed below
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from common import CACHE, CFG, DATA, FILES, log, read_tsv

DEVANAGARI = re.compile("[ऀ-ॿ]")


def convert(name, out, nrows):
    """Read one TSV and write it as parquet with an int32 row id."""
    df = read_tsv(DATA / f"{name}.tsv", nrows=nrows)
    df.columns = ["entity_id", "name", "address", "country"]
    df.insert(0, "id", np.arange(len(df), dtype=np.int32))
    df.to_parquet(out / f"{name}.parquet", index=False)
    log(f"{name}: {len(df):,} rows")
    return df


def parse_gt(s1, s2, s3, nrows):
    """Explode ground truth into (s1_id, match_id, match_source) using row ids."""
    gt = read_tsv(DATA / "train_ground_truth.tsv", nrows=nrows)
    gt.columns = ["s1", "matches"]
    s1_map = pd.Series(s1.id.values, index=s1.entity_id)
    gt["s1_id"] = gt.s1.map(s1_map)
    gt_s1 = gt.assign(n_matches=gt.matches.str.count(",") + (gt.matches != "")) \
              [["s1_id", "n_matches"]].dropna()
    ex = gt[gt.matches != ""].assign(m=lambda d: d.matches.str.split(",")).explode("m")
    ex["match_source"] = np.where(ex.m.str.startswith("S2-"), 2, 3).astype(np.int8)
    m2 = pd.Series(s2.id.values, index=s2.entity_id)
    m3 = pd.Series(s3.id.values, index=s3.entity_id)
    ex["match_id"] = np.where(ex.match_source == 2, ex.m.map(m2), ex.m.map(m3))
    missing = int(ex.s1_id.isna().sum() + ex.match_id.isna().sum())  # only >0 with --sample
    pairs = ex.dropna(subset=["s1_id", "match_id"])[["s1_id", "match_id", "match_source"]] \
              .astype({"s1_id": np.int32, "match_id": np.int32})
    return pairs.reset_index(drop=True), gt_s1.astype(np.int32), missing


def eda(dfs, pairs, gt_s1):
    """Collect the facts the later phases depend on."""
    s1, s2, s3 = dfs["train_source1"], dfs["train_source2"], dfs["train_source3"]
    facts = {"rows": {k: len(v) for k, v in dfs.items()},
             "rows_per_country": {k: v.country.value_counts().to_dict() for k, v in dfs.items()}}
    n = gt_s1.n_matches
    facts["matches_per_s1"] = {"dist": n.value_counts().sort_index().to_dict(),
                               "mean": float(n.mean()), "singleton_rate": float((n == 0).mean())}
    per = pairs.groupby(["s1_id", "match_source"]).size().unstack(fill_value=0) \
               .reindex(gt_s1.s1_id, fill_value=0)
    facts["s2_vs_s3_per_s1"] = {
        "mean_s2": float(per.get(2, pd.Series(0)).mean()),
        "mean_s3": float(per.get(3, pd.Series(0)).mean()),
        "s2_count_dist": per.get(2, pd.Series(dtype=int)).value_counts().sort_index().to_dict(),
        "s3_count_dist": per.get(3, pd.Series(dtype=int)).value_counts().sort_index().to_dict()}

    # IMPORTANT CHECK 1: can one S2/S3 record belong to more than one S1?
    owners = pairs.groupby(["match_source", "match_id"]).size()
    facts["check_multi_owner"] = {
        "records_with_1_owner": int((owners == 1).sum()),
        "records_with_2plus_owners": int((owners > 1).sum()),
        "max_owners": int(owners.max()) if len(owners) else 0}
    # share of S2/S3 records that are matched to anything (rest are pure distractors)
    ms = owners.reset_index().groupby("match_source").size()
    facts["matched_share"] = {"s2": float(ms.get(2, 0) / len(s2)), "s3": float(ms.get(3, 0) / len(s3))}

    # IMPORTANT CHECK 2: do matched pairs always share the same country?
    c1 = s1.country.values[pairs.s1_id.values]
    is2 = pairs.match_source.values == 2
    cm = np.empty(len(pairs), dtype=object)
    cm[is2] = s2.country.values[pairs.match_id.values[is2]]
    cm[~is2] = s3.country.values[pairs.match_id.values[~is2]]
    same = c1 == cm
    facts["check_same_country"] = {"pairs": int(len(same)), "same": int(same.sum()),
                                   "different": int((~same).sum())}
    if (~same).any():
        facts["check_same_country"]["examples"] = pd.Series(
            [f"{a}->{b}" for a, b in zip(c1[~same], cm[~same])]).value_counts().head(10).to_dict()

    facts["empty_address_pct"] = {k: float((v.address == "").mean() * 100) for k, v in dfs.items()}
    facts["devanagari_name_pct"] = {k: float(v.name.str.contains(DEVANAGARI).mean() * 100)
                                    for k, v in dfs.items()}
    return facts


def show_examples(dfs, pairs, gt_s1, seed, k=20):
    """Print k random train S1 entities with all their matches."""
    s1, src = dfs["train_source1"], {2: dfs["train_source2"], 3: dfs["train_source3"]}
    by_s1 = pairs.groupby("s1_id")
    for sid in gt_s1.s1_id.sample(k, random_state=seed):
        r = s1.iloc[sid]
        print(f"\n### {r.entity_id} [{r.country}] {r['name']} | {r.address}")
        if sid not in by_s1.groups:
            print("    (singleton)")
            continue
        for m in by_s1.get_group(sid).itertuples():
            x = src[m.match_source].iloc[m.match_id]
            print(f"    {x.entity_id:14} [{x.country}] {x['name']} | {x.address}")


def make_splits(s1, gt_s1, seed, frac):
    """Random 5% VAL / 95% FIT, plus leave-one-country-out (FIT=US, VAL=India sample)."""
    sp = pd.DataFrame({"s1_id": gt_s1.s1_id.values})
    sp["country"] = s1.country.values[sp.s1_id.values]
    rng = np.random.default_rng(seed)
    sp["rand"] = np.where(rng.random(len(sp)) < frac, "VAL", "FIT")
    india = sp.country == "India"
    sp["loco"] = np.where(sp.country == "US", "FIT", "")
    sp.loc[india & (rng.random(len(sp)) < frac), "loco"] = "VAL"
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="read only first N rows per file")
    a = ap.parse_args()
    out = CACHE / "sample" if a.sample else CACHE
    out.mkdir(parents=True, exist_ok=True)
    seed = CFG["seed"]

    dfs = {f: convert(f, out, a.sample) for f in FILES}
    pairs, gt_s1, missing = parse_gt(dfs["train_source1"], dfs["train_source2"],
                                     dfs["train_source3"], a.sample)
    pairs.to_parquet(out / "gt_pairs.parquet", index=False)
    gt_s1.to_parquet(out / "gt_s1.parquet", index=False)
    log(f"gt: {len(gt_s1):,} S1, {len(pairs):,} pairs, {missing} unresolved ids")
    if missing and not a.sample:
        raise SystemExit("ground truth references ids not found in source files")

    facts = eda(dfs, pairs, gt_s1)
    facts["unresolved_gt_ids"] = missing
    sp = make_splits(dfs["train_source1"], gt_s1, seed, CFG["val_frac"])
    sp.to_parquet(out / "splits.parquet", index=False)
    facts["splits"] = {"rand": sp.rand.value_counts().to_dict(),
                       "loco": sp.loco.replace("", "unused").value_counts().to_dict()}
    json.dump(facts, open(out / "eda.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(facts, indent=1, default=str))
    show_examples(dfs, pairs, gt_s1, seed)
    log("done")


if __name__ == "__main__":
    main()
