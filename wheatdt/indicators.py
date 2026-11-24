"""Crop-indicator layer: root-zone available water, thermal time, Zadoks stage, CTD and alerts."""
import numpy as np
import pandas as pd
from .config import RAW_WEIGHTS, ALERTS, GDD_BASE_C


def daily_weather(weather):
    w = weather.set_index("timestamp").resample("D")
    return w.agg({"t_air": ["mean", "max", "min"], "rainfall": "sum", "vpd": "max", "et0": "first"}).set_axis(
        ["tmean", "tmax", "tmin", "rain", "vpdmax", "et0"], axis=1)


def root_zone_water(daily, design):
    w = np.array([RAW_WEIGHTS["vwc_10cm"], RAW_WEIGHTS["vwc_30cm"], RAW_WEIGHTS["vwc_60cm"]])
    out = []
    for pid, g in daily.groupby("plot_id"):
        fc, wp = design.fc_wp(pid)
        v = g[["v10", "v30", "v60"]].to_numpy()
        out.append(pd.Series(np.clip((v - wp) / (fc - wp), 0, 1) @ w, index=g.date, name=pid))
    return pd.concat(out, axis=1)            # NaN where a depth is missing


def thermal_time(wd, sowing):
    t = wd.tmean.where(wd.index >= pd.Timestamp(sowing), 0).clip(lower=GDD_BASE_C) - GDD_BASE_C
    return t.cumsum()


def zadoks(phenology, harvest, days, sowing):
    out = pd.DataFrame(index=days, dtype=float)
    for pid, g in phenology.groupby("plot_id"):
        pts = [(pd.Timestamp(sowing), 0.0)] + sorted((pd.Timestamp(r.date), float(r.zadoks_stage)) for r in g.itertuples())
        x = np.array([(d - days[0]).days for d, _ in pts]); y = np.array([z for _, z in pts])
        dd = np.array([(d - days[0]).days for d in days])
        z = np.interp(dd, x, y, left=np.nan, right=y[-1])   # mid-season: hold the last observed stage
        h = harvest[harvest.plot_id == pid]
        if len(h) and pd.notna(h.harvest_date.iloc[0]):
            z[days >= pd.Timestamp(h.harvest_date.iloc[0])] = 99.0
        out[pid] = z
    return out


def alerts(day, raw, zad, wd):
    """Rule-based alerts for one date. Returns a list of dicts (severity, title, plots)."""
    out = []
    z = zad.loc[day]; r = raw.loc[day]; w = wd.loc[day]
    active = z[(z >= 10) & (z < 92)].index
    sev = [p for p in active if r[p] < ALERTS["raw_severe"]]
    mod = [p for p in active if ALERTS["raw_severe"] <= r[p] < ALERTS["raw_moderate"]]
    if sev: out.append(dict(severity="critical", title="Severe water stress", plots=sev))
    if mod: out.append(dict(severity="warning", title="Water stress building", plots=mod))
    zm = z.median()
    if 60 <= zm < 92 and w.vpdmax > ALERTS["vpd_high"]: out.append(dict(severity="warning", title="High VPD in grain fill", plots=[]))
    if 60 <= zm < 92 and w.tmax >= ALERTS["heat_c"]: out.append(dict(severity="critical", title="Heat during grain fill", plots=[]))
    if 30 <= zm < 70 and w.tmin <= ALERTS["frost_c"]: out.append(dict(severity="critical", title="Frost after stem elongation", plots=[]))
    return out
