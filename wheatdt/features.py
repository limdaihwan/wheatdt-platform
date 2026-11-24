"""Leak-free features for the forecasting models.

Every feature for date t uses only what the twin knows on day t: QC-cleaned sensor data and weather up to t,
phenology stages already observed, and the irrigation plan (which is set in advance). No future weather is used.
"""
import numpy as np
import pandas as pd

HORIZON = 7                      # days ahead for the stress outlook
ROOT_DEPTH_MM = 700.0            # 150 + 200 + 350 mm represented by the 10/30/60 cm probes
TRT_CODE = {"FULL": 2, "DEFICIT": 1, "RAINFED": 0}
SOIL_CODE = {"loam": 0, "clay loam": 1}
CV_CODE = {"CV_A": 0, "CV_B": 1}

STRESS_FEATURES = ["raw", "raw_d3", "raw_min7", "stage", "gdd", "ndvi", "cover", "et0_7", "rain_7", "tmean_7",
                   "doy", "irr_next", "clim_et0_next", "raw_proj", "trt", "soil"]
YIELD_FEATURES = ["stage_point", "gdd", "raw_mean", "frac_lt45", "frac_lt25", "raw_min21", "ndvi_max", "ndvi_last",
                  "lai_last", "ctd_last3", "irr_to_date", "irr_planned_rest", "trt", "cv", "soil", "block"]
YIELD_POINTS = {"Z30": "stem elongation (Z30)", "Z65": "anthesis (Z65)", "Z65+21": "mid grain filling (Z65 + 21 d)"}


def climatology_et0(weather_daily_list):
    """Mean daily ET0 by day of year across seasons, smoothed over 15 days (366 values)."""
    s = pd.concat([w.et0.groupby(w.index.dayofyear).mean() for w in weather_daily_list], axis=1).mean(axis=1)
    s = s.reindex(range(1, 367)).interpolate(limit_direction="both")
    return np.convolve(np.r_[s.values[-7:], s.values, s.values[:7]], np.ones(15) / 15, mode="valid").tolist()


def _clim_sum(clim, day, n):
    doy = [(day + pd.Timedelta(days=k)).dayofyear for k in range(1, n + 1)]
    return float(sum(clim[d - 1] for d in doy))


def observed_stage(twin):
    """Last Zadoks stage observed on or before each day (step function; no interpolation into the future)."""
    out = pd.DataFrame(0.0, index=twin.days, columns=twin.design.plots.plot_id)
    for r in twin.tables["phenology"].itertuples():
        out.loc[out.index >= pd.Timestamp(r.date), r.plot_id] = np.maximum(out.loc[out.index >= pd.Timestamp(r.date), r.plot_id], r.zadoks_stage)
    return out


def canopy_asof(twin):
    """Daily last-known canopy values per plot (QC-rejected NDVI excluded)."""
    c = twin.qc.canopy.copy()
    c.loc[c.ndvi_flag, "ndvi"] = np.nan
    c["date"] = pd.to_datetime(c.date)
    res = {}
    for k in ("ndvi", "lai", "ctd"):
        piv = c.pivot_table(index="date", columns="plot_id", values=k).reindex(twin.days)
        res[k] = piv.ffill()
        res[k + "_obs"] = piv
    return res


def irrigation_plan(twin):
    m = twin.tables["management"]
    m = m[m.event_type == "IRRIGATION"].assign(date=lambda x: pd.to_datetime(x.date))
    return m.pivot_table(index="date", columns="plot_id", values="amount", aggfunc="sum").reindex(twin.days).fillna(0.0) \
            .reindex(columns=twin.design.plots.plot_id, fill_value=0.0)


def stress_frame(twin, clim, with_target=True):
    """One row per plot-day in season. Target: root-zone available water HORIZON days later."""
    stage = observed_stage(twin); can = canopy_asof(twin); irr = irrigation_plan(twin)
    raw = twin.raw.where(~twin.qc.drift.reindex(twin.days).fillna(False))     # drop drift-flagged days
    wd = twin.wd
    et7, rain7, t7 = wd.et0.rolling(7, min_periods=4).sum(), wd.rain.rolling(7, min_periods=4).sum(), wd.tmean.rolling(7, min_periods=4).mean()
    irr_next = irr[::-1].rolling(HORIZON, min_periods=1).sum()[::-1].shift(-1).fillna(0.0)
    z92 = twin.tables["phenology"].query("zadoks_stage == 92").set_index("plot_id").date.map(pd.Timestamp)
    last = twin.days[-1]
    rows = []
    for pid in twin.design.plots.plot_id:
        fc, wp = twin.design.fc_wp(pid); taw = (fc - wp) / 100 * ROOT_DEPTH_MM
        r = twin.design.plots.loc[pid]
        x = raw[pid]
        for i, d in enumerate(twin.days):
            if stage.at[d, pid] < 10 or np.isnan(x.iloc[i]):
                continue
            if with_target and d + pd.Timedelta(days=HORIZON) > z92.get(pid, last):
                continue                                   # training: target must fall before maturity
            if not with_target and stage.at[d, pid] >= 92:
                continue                                   # inference: only stages observed so far

            ndvi = can["ndvi"].at[d, pid]
            cover = float(np.clip((ndvi - 0.14) / 0.74, 0, 1) ** 1.25) if not np.isnan(ndvi) else 0.0
            ce = _clim_sum(clim, d, HORIZON)
            row = dict(plot_id=pid, date=d, raw=x.iloc[i],
                       raw_d3=(x.iloc[i] - x.iloc[i - 3]) / 3 if i >= 3 else np.nan,
                       raw_min7=x.iloc[max(0, i - 6):i + 1].min(), stage=stage.at[d, pid], gdd=twin.gdd.at[d],
                       ndvi=ndvi, cover=cover, et0_7=et7.at[d], rain_7=rain7.at[d], tmean_7=t7.at[d], doy=d.dayofyear,
                       irr_next=irr_next.at[d, pid], clim_et0_next=ce,
                       raw_proj=float(np.clip(x.iloc[i] + (irr_next.at[d, pid] - (0.3 + 0.85 * cover) * ce) / taw, 0, 1)),
                       trt=TRT_CODE[r.irrigation_trt], soil=SOIL_CODE[r.soil_texture])
            if with_target:
                j = i + HORIZON
                row["target"] = x.iloc[j] if j < len(x) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.dropna(subset=["target"]) if with_target else df


def forecast_dates(twin):
    """Date of each yield forecast point per plot (only points whose stage has been observed)."""
    ph = twin.tables["phenology"]
    out = {}
    for pid, g in ph.groupby("plot_id"):
        st = g.set_index("zadoks_stage").date.map(pd.Timestamp)
        out[pid] = {"Z30": st.get(30), "Z65": st.get(65),
                    "Z65+21": st.get(65) + pd.Timedelta(days=21) if 65 in st else None}
    return out


def yield_frame(twin, with_target=True):
    stage = observed_stage(twin); can = canopy_asof(twin); irr = irrigation_plan(twin)
    raw = twin.raw.where(~twin.qc.drift.reindex(twin.days).fillna(False))
    fdates = forecast_dates(twin)
    hv = twin.tables["harvest"].set_index("plot_id")
    rows = []
    for pid in twin.design.plots.plot_id:
        r = twin.design.plots.loc[pid]
        em = stage.index[stage[pid] >= 10]
        if len(em) == 0: continue
        for k, (point, d) in enumerate(fdates[pid].items()):
            if d is None or pd.isna(d) or d > twin.days[-1]: continue
            x = raw.loc[em[0]:d, pid].dropna()
            ctd = can["ctd_obs"].loc[:d, pid].dropna().tail(3)
            row = dict(plot_id=pid, point=point, date=d, stage_point=k, gdd=twin.gdd.at[d],
                       raw_mean=x.mean(), frac_lt45=(x < 0.45).mean(), frac_lt25=(x < 0.25).mean(),
                       raw_min21=x.tail(21).min(), ndvi_max=can["ndvi_obs"].loc[:d, pid].max(), ndvi_last=can["ndvi"].at[d, pid],
                       lai_last=can["lai"].at[d, pid], ctd_last3=ctd.mean() if len(ctd) else np.nan,
                       irr_to_date=irr.loc[:d, pid].sum(), irr_planned_rest=irr.loc[d + pd.Timedelta(days=1):, pid].sum(),
                       trt=TRT_CODE[r.irrigation_trt], cv=CV_CODE.get(r.cultivar, 0), soil=SOIL_CODE[r.soil_texture], block=int(r.block))
            if with_target:
                row["target"] = hv.at[pid, "grain_yield"]
            rows.append(row)
    return pd.DataFrame(rows)
