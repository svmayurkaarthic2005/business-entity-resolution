"""France only: pairs the French token map REMOVES / ADDS vs stack2 (label-free).  Usage: python src/exp_frdiff.py
Prints the exp_diff table (house-number mismatch, exact core name) and 25 random removed + 15 added pairs to inspect by eye.
"""
import json

import polars as pl

from common import CACHE
from decide import apply
from exp_diff import KEY, E, N, table


def main():
    cfg = json.load(open(CACHE / "decide.json"))["cfg"]
    a = apply(pl.read_parquet(E / "stack2_test" / "pred_TEST_stack2.parquet"), cfg)
    b = apply(pl.read_parquet(E / "stack2fr_test" / "pred_TEST_stack2.parquet"), cfg)
    table("TEST: stack2fr (mapped) vs stack2", a, b, "test")
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["id", "country", "name_raw", "addr_norm"]).rename(
        {"id": "s1_id", "name_raw": "n1", "addr_norm": "a1"}).filter(pl.col("country") == "France")
    c = pl.concat([pl.read_parquet(N / f"test_source{s}.parquet", columns=["id", "name_raw", "addr_norm"]).with_columns(source=pl.lit(s, pl.Int8))
                   for s in (2, 3)]).rename({"id": "cand_id", "name_raw": "n2", "addr_norm": "a2"})
    pl.Config.set_fmt_str_lengths(70)
    for tag, d in (("REMOVED by map", a.join(b, on=KEY, how="anti")), ("ADDED by map", b.join(a, on=KEY, how="anti"))):
        d = d.select(KEY).join(s1, on="s1_id").join(c, on=["cand_id", "source"])
        print(f"\n--- France {tag}: {d.height:,}")
        for r in d.sample(min(25 if tag.startswith("R") else 15, d.height), seed=1).iter_rows(named=True):
            print(f"  {r['n1']} | {r['a1']}\n     -> {r['n2']} | {r['a2']}")


if __name__ == "__main__":
    main()
