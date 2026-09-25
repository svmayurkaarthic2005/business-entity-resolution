# Experiments
date | phase | config | blocking recall | avg candidates/S1 | val macro F0.5 | per-country F0.5 | runtime
2026-09-25 | P1 prepare | full data, seed 42, val 5% | - | - | - | - | 214s, peak 3.2GB
2026-09-25 | P2 normalize | learned translit 1312 / alias 296 / noise 74 from 1.5M FIT pairs | - | - | - | VAL pairs: name_core equal 58.7% (raw 26.8%), state eq 98.3%, house_no eq 79.5% | 275s (11.3s/1M rows), peak 4.7GB
2026-09-25 | P3 block v1 (Colab CPU) | char+word TF-IDF top30 + exact, score=char+word, max_df 0.01, VAL frac 0.2 (21,991 S1) | K5 0.781 / K10 0.823 / K20 0.855 / K30 0.872 | 10 / 20 / 40 / 60 | - | K30: US 0.907, India 0.820 | tfidf ~4-5 min/country, top-n ~10s per 10K S1 per source. GATE FAILED: misses are ranking losses (95%), not retrieval
