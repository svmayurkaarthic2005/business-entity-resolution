"""Diagnostic reference only: assemble the frozen FWD4 (ctl1) training matrix with the production stage-1 models (no training) -> exp/fwd4_ref/du.parquet."""
import sys; sys.path.insert(0, "/content/ber/src")
import numpy as np, polars as pl
import exp_stack as S
from exp_final import assemble
from common import CACHE, log
EX = CACHE.parent / "exp"; SRC = EX / "union_fwd4_ctl1"; OUT = EX / "fwd4_ref"; OUT.mkdir(parents=True, exist_ok=True)
allf = pl.read_parquet(SRC / "allf_union.parquet")
fmap = dict(pl.read_parquet(SRC / "fold_map.parquet").iter_rows())
class ProductionFolds:
    def __init__(self, n): self.n = n
    def split(self, X_, y, groups):
        f = np.array([fmap[g] for g in groups])
        for i in range(self.n):
            yield np.flatnonzero(f != i), np.flatnonzero(f == i)
S.GroupKFold = ProductionFolds
du, folds, fit_ids = assemble(allf, "fwd4ref")
du.write_parquet(OUT / "du.parquet")
log(f"wrote {OUT / 'du.parquet'} {du.shape}")
