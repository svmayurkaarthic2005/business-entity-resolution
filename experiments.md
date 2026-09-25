# Experiments
date | phase | config | blocking recall | avg candidates/S1 | val macro F0.5 | per-country F0.5 | runtime
2026-09-25 | P1 prepare | full data, seed 42, val 5% | - | - | - | - | 214s, peak 3.2GB
2026-09-25 | P2 normalize | learned translit 1312 / alias 296 / noise 74 from 1.5M FIT pairs | - | - | - | VAL pairs: name_core equal 58.7% (raw 26.8%), state eq 98.3%, house_no eq 79.5% | 275s (11.3s/1M rows), peak 4.7GB
