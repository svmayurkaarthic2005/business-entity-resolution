"""Phase 2: normalise names and addresses for all 6 source files.

Usage:  python src/normalize.py              # all rows -> cache/norm/<file>.parquet
        python src/normalize.py --sample 200000   # first N rows per file -> cache/sample_norm/

Everything is learned from the provided data only (FIT split pairs for supervised maps):
  * translit : Indic-script name token -> Latin token, aligned from S1(Latin)/match(Indic) pairs;
               unseen tokens fall back to rule-based transliteration (indic-transliteration, MIT)
  * alias    : address component variant -> canonical (TX<->Texas, native-script state names)
  * noise    : name tokens that true pairs often disagree on (legal forms, "services", ...)
               + a seed list of legal forms incl. French ones (France has no training labels)
Output columns: id, name_raw, name_norm, name_alt, name_latin, name_core, name_suffix,
  addr_norm, house_no, postcode, city_guess, state_guess, street_tokens, addr_missing
"""
import argparse
import json
import time
from collections import Counter, defaultdict

import polars as pl
from indic_transliteration import sanscript

from common import CACHE, FILES, log

INDIC = r"[ऀ-෿]"
SCRIPTS = [(0x0D00, sanscript.MALAYALAM), (0x0C80, sanscript.KANNADA), (0x0C00, sanscript.TELUGU),
           (0x0B80, sanscript.TAMIL), (0x0B00, sanscript.ORIYA), (0x0A80, sanscript.GUJARATI),
           (0x0A00, sanscript.GURMUKHI), (0x0980, sanscript.BENGALI), (0x0900, sanscript.DEVANAGARI)]
SEED_LEGAL = set("pvt private ltd limited llp inc incorporated llc corp corporation co company plc "
                 "sarl sas sasu sa sci eurl snc selarl scp ei gmbh".split())
ABBR = {"rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane",
        "ct": "court", "hwy": "highway", "r": "rue", "pl": "place", "chem": "chemin",
        "imp": "impasse", "opp": "near", "nr": "near", "rte": "route", "pkwy": "parkway",
        "cir": "circle", "sq": "square", "apt": "apartment", "ste": "suite", "fl": "floor",
        "flr": "floor", "bldg": "building", "ngr": "nagar", "n": "", "no": "", "null": ""}
LEARN_PAIRS = 1_500_000


# ---------------------------------------------------------------- stage A: base cleaning
def clean(e):
    """Accent-free lowercase text; '&' -> 'and'; dots/apostrophes joined (s.a.s -> sas)."""
    return (e.str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")
             .str.to_lowercase().str.replace_all("&", " and "))


def squash(e, keep=""):
    """Replace everything except letters/digits/marks (and `keep` chars) by one space."""
    return e.str.replace_all(rf"[^\p{{L}}\p{{N}}\p{{M}}{keep}]+", " ").str.strip_chars()


def base(df):
    """Name/address cleaning that needs no learned maps."""
    n = clean(pl.col("name"))
    n = (n.str.replace_all(r"(https?://)?www\.", " ")
          .str.replace_all(r"\.(com|net|org|co|in|fr|biz|info|us|io)(\.[a-z]{2})?\b", " ")
          .str.replace_all(r"#\s*\d+", " "))
    alt = n.str.extract(r"(?:\bd\s*/?\s*b\s*/?\s*a\b|\bt/a\b|\baka\b|\|)(.*)$", 1)
    a = clean(pl.col("address")).str.replace_all(r"[.'`]", "")
    a = squash(a, keep=",/-").str.replace_all(r"(\D)-", "${1} ").str.replace_all(r"-(\D)", " ${1}")
    for k, v in ABBR.items():
        a = a.str.replace_all(rf"\b{k}\b", v)
    comps = (a.str.split(",").list.eval(pl.element().str.replace_all(r"\s+", " ").str.strip_chars())
              .list.eval(pl.element().filter(pl.element() != "")))
    return df.with_columns(
        name_raw=pl.col("name"),
        name_norm=squash(n.str.replace_all(r"[.'`’]", "")).str.replace_all(r"\s+", " "),
        name_alt=squash(alt.fill_null("").str.replace_all(r"[.'`’]", "")).str.replace_all(r"\s+", " "),
        comps=comps)


# ---------------------------------------------------------------- transliteration
def rule_translit(tok):
    """Fallback: rule-based romanisation of one Indic token, reduced to a-z."""
    script = next((s for lo, s in SCRIPTS if ord(tok[0]) >= lo), sanscript.DEVANAGARI)
    out = sanscript.transliterate(tok, script, sanscript.ITRANS).lower()
    return "".join(c for c in out if "a" <= c <= "z") or tok


def translit_names(names, tdict, cache):
    """Map each Indic-script name to Latin token by token (learned dict, then rules)."""
    out = {}
    for s in names:
        toks = []
        for t in s.split():
            if t not in cache:
                cache[t] = tdict.get(t) or (rule_translit(t) if any("ऀ" <= c <= "෿" for c in t) else t)
            toks.append(cache[t])
        out[s] = " ".join(toks)
    return out


# ---------------------------------------------------------------- stage B: learn maps
def learn(n_pairs):
    """Learn translit dict, address alias map and noise tokens from FIT-split train pairs."""
    pairs = pl.read_parquet(CACHE / "gt_pairs.parquet")
    fit = pl.read_parquet(CACHE / "splits.parquet").filter(pl.col("rand") == "FIT").select("s1_id")
    pairs = pairs.join(fit, on="s1_id").sample(min(n_pairs, len(pairs)), seed=42)
    s1 = base(pl.read_parquet(CACHE / "train_source1.parquet").join(
        pairs.select(pl.col("s1_id").alias("id")).unique(), on="id"))
    m = []
    for src in (2, 3):
        ids = pairs.filter(pl.col("match_source") == src).select(pl.col("match_id").alias("id")).unique()
        m.append(base(pl.read_parquet(CACHE / f"train_source{src}.parquet").join(ids, on="id"))
                 .with_columns(match_source=pl.lit(src, pl.Int8)))
    m = pl.concat(m)
    j = (pairs.join(s1.select(pl.col("id").alias("s1_id"), pl.col("name_norm").alias("n1"),
                              pl.col("comps").alias("c1")), on="s1_id")
              .join(m.select(pl.col("id").alias("match_id"), "match_source", pl.col("name_norm").alias("n2"),
                             pl.col("comps").alias("c2")), on=["match_id", "match_source"]))
    log(f"learn: {len(j):,} FIT pairs")

    # 1) transliteration dict: Indic match name vs Latin S1 name with equal token counts
    votes = defaultdict(Counter)
    for n1, n2 in j.filter(pl.col("n2").str.contains(INDIC)).select("n1", "n2").iter_rows():
        a, b = n1.split(), n2.split()
        if len(a) == len(b):
            for x, y in zip(a, b):
                if any("ऀ" <= c <= "෿" for c in y):
                    votes[y][x] += 1
    tdict = {}
    for y, c in votes.items():
        x, k = c.most_common(1)[0]
        if k >= 2 and k / sum(c.values()) >= 0.5:
            tdict[y] = x

    # 2) address alias: the single differing component on each side, counted
    ali, tot = defaultdict(Counter), Counter()
    for c1, c2 in j.select("c1", "c2").iter_rows():
        A = {x for x in c1 if not any(ch.isdigit() for ch in x)}
        B = {x for x in c2 if not any(ch.isdigit() for ch in x)}
        oa, ob = A - B, B - A
        if len(oa) == 1 and len(ob) == 1:
            b = next(iter(ob))
            ali[b][next(iter(oa))] += 1
            tot[b] += 1
    alias = {}
    for b, c in ali.items():
        a, k = c.most_common(1)[0]
        if k >= 20 and k / tot[b] >= 0.6:
            alias[b] = a

    # 3) noise tokens: how often a token is present on only one side of a true pair
    cache = {}
    lat = translit_names(j.filter(pl.col("n2").str.contains(INDIC))["n2"].unique().to_list(), tdict, cache)
    one, both = Counter(), Counter()
    for n1, n2 in j.select("n1", "n2").iter_rows():
        A, B = set(n1.split()), set(lat.get(n2, n2).split())
        one.update(A ^ B)
        both.update(A & B)
    drop = {t: one[t] / (one[t] + both[t]) for t in set(one) | set(both)
            if one[t] + both[t] >= 300 and not t.isdigit()}
    noise = sorted({t for t, r in drop.items() if r >= 0.5} | SEED_LEGAL)
    log(f"learned: translit {len(tdict):,} tokens, alias {len(alias):,}, noise {len(noise)}")
    return {"translit": tdict, "alias": alias, "noise": noise,
            "noise_drop_rate": {t: round(drop[t], 3) for t in noise if t in drop}}


# ---------------------------------------------------------------- stage C: final columns
def finalize(df, maps, loc_freq, cache):
    """Apply learned maps and extract address parts. `df` is the output of base()."""
    ind = df.filter(pl.col("name_norm").str.contains(INDIC))["name_norm"].unique().to_list()
    lat = translit_names(ind, maps["translit"], cache)
    alt_ind = df.filter(pl.col("name_alt").str.contains(INDIC))["name_alt"].unique().to_list()
    lat.update(translit_names(alt_ind, maps["translit"], cache))
    words = "|".join(maps["noise"])
    lead, trail = rf"^(?:(?:{words})(?:\s+|$))+", rf"(?:(?:^|\s+)(?:{words}))+$"
    latin = pl.col("name_norm").replace(lat)
    df = df.with_columns(
        name_latin=latin,
        name_alt=pl.col("name_alt").replace(lat),
        comps=pl.col("comps").list.eval(pl.element().replace(maps["alias"])))
    core = pl.col("name_latin").str.replace(lead, "").str.replace(trail, "").str.strip_chars()
    suffix = pl.concat_str([pl.col("name_latin").str.extract(lead, 0),
                            pl.col("name_latin").str.extract(trail, 0)], ignore_nulls=True)
    df = df.with_columns(
        name_core=pl.when(core == "").then(pl.col("name_latin")).otherwise(core),
        name_suffix=suffix.str.replace_all(r"\s+", " ").str.strip_chars().fill_null(""),
        addr_norm=pl.col("comps").list.join(", "),
        street_tokens=(pl.col("comps").list.eval(pl.element().filter(pl.element().str.contains(r"\d")))
                       .list.first().str.replace_all(r"\S*\d\S*", "").str.replace_all(r"\s+", " ")
                       .str.strip_chars().fill_null("")))
    df = df.with_columns(
        house_no=pl.col("addr_norm").str.extract(r"(\d+[a-z]?(?:[/-]\d+[a-z]?)*)", 1)
                   .str.replace(r"^0+(\d)", "${1}"),  # "00547" -> "547"
        postcode=pl.coalesce(
            pl.col("addr_norm").str.extract(r"\b([1-9]\d{5})\b", 1),
            pl.col("comps").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{5}$"))).list.first()),
        addr_missing=pl.col("addr_norm") == "")
    # city/state: non-numeric components ranked by how common they are in the country
    loc = (df.select("id", "country", "comps").explode("comps", empty_as_null=True)
             .filter(pl.col("comps").is_not_null() & ~pl.col("comps").str.contains(r"\d"))
             .join(loc_freq, left_on=["country", "comps"], right_on=["country", "comp"], how="left")
             .with_columns(pl.col("freq").fill_null(0))
             .sort(["id", "freq"], descending=[False, True])
             .group_by("id", maintain_order=True)
             .agg(first=pl.col("comps").first(), f1=pl.col("freq").first(),
                  second=pl.col("comps").get(1, null_on_oob=True)))
    loc = loc.with_columns(
        state_guess=pl.when(pl.col("f1") >= 200).then(pl.col("first")),
        city_guess=pl.when(pl.col("f1") >= 200).then(pl.col("second")).otherwise(pl.col("first")))
    df = df.join(loc.select("id", "state_guess", "city_guess"), on="id", how="left")
    return df.select("id", "country", "name_raw", "name_norm", "name_alt", "name_latin", "name_core",
                     "name_suffix", "addr_norm", "house_no", "postcode", "city_guess", "state_guess",
                     "street_tokens", "addr_missing")


def location_freq(maps):
    """Count canonical non-numeric address components per country over S1 (train + test)."""
    parts = []
    for f in ("train_source1", "test_source1"):
        d = base(pl.read_parquet(CACHE / f"{f}.parquet"))
        parts.append(d.select("country", pl.col("comps").list.eval(pl.element().replace(maps["alias"])))
                      .explode("comps", empty_as_null=True).rename({"comps": "comp"}))
    return (pl.concat(parts).filter(pl.col("comp").is_not_null() & ~pl.col("comp").str.contains(r"\d"))
              .group_by("country", "comp").len("freq").with_columns(pl.col("freq").cast(pl.Int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="only first N rows per file")
    a = ap.parse_args()
    out = CACHE / ("sample_norm" if a.sample else "norm")
    out.mkdir(parents=True, exist_ok=True)

    maps = learn(min(a.sample or LEARN_PAIRS, LEARN_PAIRS))
    json.dump(maps, open(out / "norm_maps.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    loc_freq = location_freq(maps)
    log(f"location vocab: {len(loc_freq):,} components")
    cache, total_rows, t0 = {}, 0, time.time()
    for f in FILES:
        df = pl.read_parquet(CACHE / f"{f}.parquet", n_rows=a.sample)
        chunks = [finalize(base(df.slice(i, 1_000_000)), maps, loc_freq, cache)
                  for i in range(0, len(df), 1_000_000)]
        pl.concat(chunks).write_parquet(out / f"{f}.parquet")
        total_rows += len(df)
        log(f"{f}: {len(df):,} rows")
    dt = time.time() - t0
    log(f"normalised {total_rows:,} rows in {dt:.0f}s = {dt / total_rows * 1e6:.1f}s per 1M rows")


if __name__ == "__main__":
    main()
