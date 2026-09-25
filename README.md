# Business Entity Resolution — Amazon ML Challenge 2026

Matches every Source 1 business record to its Source 2 / Source 3 records
(noisy names + addresses, US / India / France). Metric: macro F0.5 per S1 entity.

Pipeline: prepare -> normalize -> block -> features -> LightGBM -> decision layer.

## Run
```
pip install -r requirements.txt
set PYTHONIOENCODING=utf-8
python src/score.py        # metric self-tests
python src/prepare.py      # TSV -> parquet, EDA facts, validation splits
python src/normalize.py    # name/address normalisation (maps learned from FIT pairs)
python src/check_norm.py   # normalisation gate report
```
Paths are in `config.yaml` (`data_dir` = folder with the 7 challenge TSVs).
No external data or APIs are used.
