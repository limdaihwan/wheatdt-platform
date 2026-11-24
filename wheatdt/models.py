"""Forecasting layer: 7-day root-zone water outlook and in-season yield forecasts.

Models are trained by `python -m wheatdt train` and stored in models/forecast_models.joblib.
If no model file is present the twin runs without forecasts.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import features as F

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"


class Forecaster:
    def __init__(self, bundle, report=None):
        self.b = bundle
        self.report = report

    @classmethod
    def load(cls, models_dir=None):
        d = Path(models_dir) if models_dir else DEFAULT_DIR
        f = d / "forecast_models.joblib"
        if not f.exists():
            return None
        import joblib
        rep = d / "model_report.json"
        return cls(joblib.load(f), json.loads(rep.read_text()) if rep.exists() else None)

    @property
    def warn_threshold(self):
        return self.b["stress_interval"].get("warn_threshold", 0.45)

    def stress_outlook(self, twin):
        """(pred, lo, hi, warn) DataFrames (days × plots): root-zone water 7 days ahead with 80% interval, and early-warning flags."""
        X = F.stress_frame(twin, self.b["clim"], with_target=False)
        cols = list(twin.design.plots.plot_id)
        empty = pd.DataFrame(np.nan, index=twin.days, columns=cols)
        if X.empty:
            return empty, empty.copy(), empty.copy(), empty.copy()
        p = np.clip(self.b["stress"].predict(X[F.STRESS_FEATURES]), 0, 1)
        iv = self.b["stress_interval"]
        X = X.assign(pred=p, lo=np.clip(p + iv["q10"], 0, 1), hi=np.clip(p + iv["q90"], 0, 1))
        rule = iv.get("warn_rule", "hybrid")
        a, b = X.raw_proj < 0.45, X.pred < self.warn_threshold
        X["warn"] = ({"projection": a, "hybrid": b, "either": a | b}[rule] & (X.raw >= 0.45)).astype(float)
        out = [X.pivot(index="date", columns="plot_id", values=k).reindex(index=twin.days, columns=cols) for k in ("pred", "lo", "hi", "warn")]
        return tuple(out)

    def yield_forecasts(self, twin):
        """One row per plot and forecast point that has been reached: predicted yield with 80% interval."""
        X = F.yield_frame(twin, with_target=False)
        rows = []
        for point, g in X.groupby("point"):
            m = self.b["yield_models"][point]; iv = self.b["yield_interval"][point]
            p = m.predict(g[F.YIELD_FEATURES])
            rows.append(g[["plot_id", "point", "date"]].assign(pred=p, lo=p + iv["q10"], hi=p + iv["q90"]))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["plot_id", "point", "date", "pred", "lo", "hi"])
