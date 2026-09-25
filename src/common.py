"""Shared helpers: config, TSV reading, timing/RAM logging."""
import csv
import os
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DATA = Path(CFG["data_dir"])
CACHE = Path(CFG["cache_dir"])
OUT = Path(CFG["output_dir"])
FILES = ["train_source1", "train_source2", "train_source3",
         "test_source1", "test_source2", "test_source3"]


def read_tsv(path, **kw):
    """Read a challenge TSV exactly as the rules require (no quoting, no NaN)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8", **kw)


_T0 = time.time()


def log(msg):
    """Print a message with elapsed seconds, current and peak process RAM."""
    mi = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mi, "peak_wset", mi.rss) / 2**30  # peak_wset exists on Windows
    print(f"[{time.time() - _T0:7.1f}s | ram {mi.rss / 2**30:5.2f} GB | peak {peak:5.2f} GB] {msg}",
          flush=True)
