"""Trial-aware QC layer.

Spikes: range test (> FC + 10) and rolling-median test, per hourly reading.
Drift:  SaQC drift-from-norm on daily available water within treatment groups (default),
        plus the peer-trend rule on raw VWC within soil × treatment groups (secondary).
Outage: days with no readings. NDVI: dips against treatment median and neighbouring flights.
Flags are stored next to values; nothing is deleted.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import QC
from ._compat import load_saqc


@dataclass
class QCResult:
    soil: pd.DataFrame                 # hourly soil readings + spike flag
    daily: pd.DataFrame                # plot × day: v10/v30/v60 (spike-free means), n readings
    drift_dfn: pd.DataFrame            # day × plot booleans
    drift_pt: pd.DataFrame
    canopy: pd.DataFrame               # canopy + ndvi_flag
    engine: str
    groups: dict
    events: list = field(default_factory=list)

    @property
    def drift(self):
        """Primary drift flag: SaQC DFN on available water (peer-trend rule if SaQC is unavailable)."""
        return self.drift_dfn if self.engine == "saqc" else self.drift_pt

    @property
    def drift_check(self):
        """Secondary: peer-trend flags not confirmed by the primary test (shown as 'check probe')."""
        return self.drift_pt & ~self.drift if self.engine == "saqc" else self.drift_pt & False


def _runs(flags):
    out, s = [], None
    for d, f in flags.items():
        if f and s is None: s = d
        if not f and s is not None: out.append((s, prev)); s = None
        prev = d
    if s is not None: out.append((s, prev))
    return out


def detect_spikes(soil, design):
    s = soil.sort_values(["plot_id", "timestamp"]).copy()
    fc = s.plot_id.map(lambda p: design.fc_wp(p)[0])
    rm = s.groupby("plot_id").vwc_10cm.transform(
        lambda v: v.rolling(QC["spike_window_h"], center=True, min_periods=5).median())
    s["spike"] = (s.vwc_10cm > fc + QC["range_margin"]) | ((s.vwc_10cm - rm).abs() > QC["spike_k"])
    return s


def daily_means(s, design, days):
    c = s[~s.spike].assign(date=s.timestamp.dt.normalize())
    d = c.groupby(["plot_id", "date"]).agg(v10=("vwc_10cm", "mean"), v30=("vwc_30cm", "mean"),
                                          v60=("vwc_60cm", "mean"), n=("vwc_10cm", "size"))
    full = pd.MultiIndex.from_product([list(design.plots.plot_id), days], names=["plot_id", "date"])
    return d.reindex(full).reset_index()


def _trend(v):
    m = ~np.isnan(v)
    if m.sum() < 5: return 0.0
    x, y = np.arange(len(v))[m], v[m]
    if np.std(y) == 0: return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return np.polyfit(x, y, 1)[0] if abs(r) > QC["pt_r"] else 0.0


def _slope(v):
    m = ~np.isnan(v)
    return np.polyfit(np.arange(len(v))[m], v[m], 1)[0] if m.sum() >= 3 else 0.0


def _expand(flag):
    return flag.astype(float)[::-1].rolling(7, min_periods=1).max()[::-1].astype(bool)


def drift_peer_trend(D, design):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    for pid in D.columns:
        pe = design.peers_of(pid)
        if not pe: continue
        pm = D[pe].median(axis=1)
        fl = (D[pid] - pm).rolling(7, min_periods=5).apply(_trend, raw=True).abs() > QC["pt_slope"]
        fl &= pm.rolling(7, min_periods=5).apply(_slope, raw=True).abs() < QC["pt_gate"]
        out[pid] = _expand(fl.fillna(False))
    # common-mode suppression: when half or more of a soil × treatment group trends together,
    # the field is changing (rain, irrigation, dry-down), not one probe
    keep = out.copy()
    for pid in D.columns:
        grp = [pid] + design.peers_of(pid)
        keep[pid] = out[pid] & (out[grp].sum(axis=1) < len(grp) / 2)
    return keep


def drift_dfn(D, design, SaQC):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    Z = pd.DataFrame({p: design.available_water(p, D[p]) for p in D.columns})
    groups = design.peer_groups(("irrigation_trt",))
    for _, g in groups.items():
        if len(g) < 3: continue           # DFN needs a majority
        df = Z[g].interpolate(limit=3); df.columns = [f"f_{c}" for c in g]
        qc = SaQC(df).flagDriftFromNorm(list(df.columns), window=QC["dfn_window"], spread=QC["dfn_spread"], frac=QC["dfn_frac"])
        for c in g:
            out[c] = (qc.flags[f"f_{c}"] > 0).reindex(D.index).fillna(False).values
    return out, groups


def ndvi_flags(canopy, design, phenology):
    c = canopy.sort_values(["plot_id", "date"]).copy()
    trt = c.plot_id.map(design.plots.irrigation_trt)
    med = c.assign(trt=trt).groupby(["date", "trt"]).ndvi.transform("median")
    nb = c.groupby("plot_id").ndvi.transform(lambda v: pd.concat([v.shift(1), v.shift(-1)], axis=1).mean(axis=1))
    z92 = phenology[phenology.zadoks_stage == 92].set_index("plot_id").date
    late = pd.to_datetime(c.date) >= pd.to_datetime(c.plot_id.map(z92)) - pd.Timedelta(days=QC["ndvi_exclude_before_maturity_d"])
    k = QC["ndvi_ratio"]
    c["ndvi_flag"] = (c.ndvi < k * med) & (c.ndvi < k * nb) & ~late
    return c


def run_qc(tables, design, days, use_saqc=True):
    s = detect_spikes(tables["soil_sensor"], design)
    daily = daily_means(s, design, days)
    D = daily.pivot(index="date", columns="plot_id", values="v10")
    pt = drift_peer_trend(D, design)
    SaQC = load_saqc() if use_saqc else None
    if SaQC is not None:
        dfn, groups = drift_dfn(D, design, SaQC); engine = "saqc"
    else:
        dfn, groups = pd.DataFrame(False, index=D.index, columns=D.columns), design.peer_groups(("irrigation_trt", "soil_texture")); engine = "peer-trend only"
    can = ndvi_flags(tables["canopy"], design, tables["phenology"])
    res = QCResult(s, daily, dfn, pt, can, engine, groups)
    sow = pd.Timestamp(design.plots.sowing_date.iloc[0])
    for pid in D.columns:
        for a, b in _runs(res.drift[pid]):
            agree = engine == "saqc" and bool(pt[pid].loc[a:b].any())
            res.events.append(dict(type="drift", plot=pid, start=a, end=b,
                                   by=("SaQC drift-from-norm, treatment group, available water" if engine == "saqc" else "peer-trend rule")
                                   + ("; peer-trend rule agrees" if agree else "")))
        for a, b in _runs(res.drift_check[pid]):
            if (b - a).days < 2: continue   # ignore 1–2 day secondary blips
            res.events.append(dict(type="drift_check", plot=pid, start=a, end=b, by="peer-trend rule only"))
        n = daily[daily.plot_id == pid].set_index("date").n
        for a, b in _runs((n.isna() | (n == 0)) & (n.index >= sow)):
            res.events.append(dict(type="outage", plot=pid, start=a, end=b, by="no readings"))
    for r in can[can.ndvi_flag].itertuples():
        res.events.append(dict(type="ndvi", plot=r.plot_id, start=pd.Timestamp(r.date), end=pd.Timestamp(r.date), by=f"NDVI {r.ndvi:.2f}"))
    res.events.sort(key=lambda e: e["start"])
    return res
