# Training data for the forecast models

All 34 seasons are **synthetic** (platform generator, `wheatdt/synthetic.py`, faults on).

| Path | Contents | Size |
|---|---|---|
| `archives/s201.zip` … `s224.zip` | 24 training seasons (seeds 201–224): 8 CSV tables + `_truth/` each | ~1.4 MB each |
| `archives/s301.zip` … `s310.zip` | 10 held-out test seasons (seeds 301–310) | ~1.4 MB each |
| `stress_rows.csv.gz` | Plot-day feature table for the 7-day root-zone water outlook (target = RAW at t+7) | 8.3 MB |
| `yield_rows.csv` | Feature table for yield forecasts at Z30, Z65, Z65+21 | 0.4 MB |
| `clim_et0.json` | Daily ET0 climatology (366 values) from the training seasons only | 7 KB |

```bash
python -m wheatdt train                 # retrains from the feature tables (identical to models/)
```

If the feature tables are deleted, `train` unpacks each season from `archives/` (or regenerates it from its seed
when the archive is missing too) and rebuilds the tables. Unpacked season folders are git-ignored.
Test seasons are only used for the scores in `models/model_report.json`; never use them to tune anything.
