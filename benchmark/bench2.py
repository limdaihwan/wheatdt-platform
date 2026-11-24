"""Additional analyses for revision:
A) plausibility of synthetic seasons vs. a real Valladolid trial
B) peer-group construction and group size
C) robustness to generator / fault assumptions
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import bench as B
from saqc import SaQC

TP = json.load(open(f"{B.BENCH}/tuned_params.json"))
SPREAD = TP["drift"]["SaQC drift-from-norm"]; SLOPE = TP["drift"]["Proposed"]
ALL = B.TUNE + B.TEST

# ------------------------------------------------------------------ A
def plausibility():
    rows = []
    for s in ALL:
        d = f"{B.BENCH}/s{s}"
        w = pd.read_csv(f"{d}/weather.csv", parse_dates=["timestamp"])
        p = pd.read_csv(f"{d}/plots.csv"); h = pd.read_csv(f"{d}/harvest.csv").merge(p)
        ph = pd.read_csv(f"{d}/phenology.csv", parse_dates=["date"]).merge(p)
        c = pd.read_csv(f"{d}/canopy.csv", parse_dates=["date"]).merge(p)
        mg = pd.read_csv(f"{d}/management.csv")
        hv = pd.to_datetime(h.harvest_date).median()
        seas = w[(w.timestamp >= "2024-11-15") & (w.timestamp <= hv)]
        daily = seas.set_index("timestamp").resample("D").agg({"t_air": ["mean", "max", "min"], "rainfall": "sum"})
        z65 = ph[ph.zadoks_stage == 65].date
        gf = c[(c.date >= z65.median()) & (c.date <= z65.median() + pd.Timedelta(days=21))]
        ctd = gf.groupby("irrigation_trt").ctd.mean()
        rows.append(dict(seed=s, t_mean=daily[("t_air", "mean")].mean(), t_max=daily[("t_air", "max")].mean(), t_min=daily[("t_air", "min")].mean(),
                         rain=daily[("rainfall", "sum")].sum(),
                         y_full=h[h.irrigation_trt == "FULL"].grain_yield.mean(), y_def=h[h.irrigation_trt == "DEFICIT"].grain_yield.mean(),
                         y_rf=h[h.irrigation_trt == "RAINFED"].grain_yield.mean(),
                         drop=1 - h[h.irrigation_trt == "RAINFED"].grain_yield.mean() / h[h.irrigation_trt == "FULL"].grain_yield.mean(),
                         anthesis_doy=z65.dt.dayofyear.median(), harvest_doy=hv.dayofyear,
                         irr_full=mg[(mg.event_type == "IRRIGATION")].groupby("plot_id").amount.sum().max(),
                         ctd_diff=ctd.get("RAINFED") - ctd.get("FULL"), hi_full=h[h.irrigation_trt == "FULL"].harvest_index.mean()))
    df = pd.DataFrame(rows); df.to_csv(f"{B.BENCH}/plausibility.csv", index=False)
    print(df.describe().round(2).T[["mean", "std", "min", "max"]])

# ------------------------------------------------------------------ B
def make_groups(p, scheme, rng):
    if scheme == "soil x treatment (4)":
        return [list(g) for _, g in p.groupby(["soil_texture", "irrigation_trt"]).plot_id]
    if scheme == "treatment x cultivar (4)":
        return [list(g) for _, g in p.groupby(["irrigation_trt", "cultivar"]).plot_id]
    if scheme == "treatment (8)":
        return [list(g) for _, g in p.groupby("irrigation_trt").plot_id]
    if scheme == "block, mixed treatments (6)":
        return [list(g) for _, g in p.groupby("block").plot_id]
    if scheme == "random (4)":
        ids = rng.permutation(p.plot_id.values); return [list(ids[i:i + 4]) for i in range(0, 24, 4)]
    if scheme == "all plots (24)":
        return [list(p.plot_id)]
    if scheme.startswith("treatment, size "):
        k = int(scheme.split()[-1]); out = []
        for _, g in p.groupby("irrigation_trt").plot_id:
            ids = rng.permutation(g.values); out += [list(ids[i:i + k]) for i in range(0, len(ids), k)]
        return out
    raise ValueError(scheme)

def saqc_groups(D, groups, spread=SPREAD):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    for g in groups:
        if len(g) < 2: continue
        df = D[g].copy(); df.columns = [f"f_{c}" for c in g]
        qc = SaQC(df.interpolate(limit=3)).flagDriftFromNorm([f"f_{c}" for c in g], window="7D", spread=spread, frac=0.5)
        for c in g: out[c] = (qc.flags[f"f_{c}"] > 0).reindex(D.index).fillna(False).values
    return out

def peer_groups(D, groups, s=SLOPE, gate=0.15):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    for g in groups:
        for pid in g:
            pe = [x for x in g if x != pid]
            if not pe: continue
            pm = D[pe].median(axis=1); res = D[pid] - pm
            fl = res.rolling(7, min_periods=5).apply(B.trend, raw=True).abs() > s
            fl &= B.slope_series(pm).abs() < gate
            out[pid] = B.expand(fl)
    return out

SCHEMES = ["soil x treatment (4)", "treatment x cultivar (4)", "treatment (8)", "block, mixed treatments (6)", "random (4)", "all plots (24)",
           "treatment, size 2", "treatment, size 3", "treatment, size 4", "treatment, size 8"]

def grouping(seeds):
    rows = []
    for s in seeds:
        p, wide = B.load(s); X, sp, dh, ev = B.inject(p, wide, s); D = B.daily(X)
        rng = np.random.default_rng(s + 7)
        for sch in SCHEMES:
            G = make_groups(p, sch, rng)
            for name, fn in [("SaQC DFN", saqc_groups), ("Peer trend + gate", peer_groups)]:
                rows.append(dict(seed=s, scheme=sch, method=name, **B.drift_scores(fn(D, G), ev, D)))
        print("grouping seed", s, flush=True)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ C
def inject_var(p, wide, seed, scen):
    rng = np.random.default_rng(seed + 5000)
    X = wide.copy()
    if scen == "noise x2":
        X = X + rng.normal(0, 0.4, X.shape)  # doubles noise variance-equivalent (0.4 -> ~0.57 SD)
    spike_rate = 0.01 if scen == "spikes x3.3 (p = 0.01)" else 0.003
    lo, hi = (0.08, 0.20) if scen == "small drift (0.08-0.20)" else (0.15, 0.40)
    spike = pd.DataFrame(False, index=X.index, columns=X.columns); events = []
    plots = list(rng.choice(X.columns, 3, replace=False))
    if scen == "two drifts in one group":
        grp = p[p.plot_id == plots[0]].iloc[0]
        mates = p[(p.soil_texture == grp.soil_texture) & (p.irrigation_trt == grp.irrigation_trt) & (p.plot_id != plots[0])].plot_id.values
        plots[1] = rng.choice(mates)
    shared = None
    for k, pid in enumerate(plots):
        start = pd.Timestamp("2024-12-15") + pd.Timedelta(days=int(rng.integers(0, 150)))
        days = int(rng.integers(10, 22)); slope = float(rng.uniform(lo, hi)) * rng.choice([-1, 1])
        if scen == "two drifts in one group" and k == 1: start, days, slope = shared
        if k == 0: shared = (start, days, slope)
        idx = (X.index >= start) & (X.index < start + pd.Timedelta(days=days))
        X.loc[idx, pid] = X.loc[idx, pid] + slope * (X.index[idx] - start).total_seconds() / 86400
        events.append(dict(plot=pid, start=start.normalize(), days=days, slope=slope))
    obs = X.notna().values
    m = (rng.random(X.shape) < spike_rate) & obs
    pos = rng.random(X.shape) < 0.7
    fac = np.where(pos, rng.uniform(1.25, 1.9, X.shape), rng.uniform(0.45, 0.75, X.shape))
    X = X.where(~m, X * fac); spike = pd.DataFrame(m, index=X.index, columns=X.columns)
    return X, spike, events

SCENS = ["baseline", "noise x2", "small drift (0.08-0.20)", "two drifts in one group", "spikes x3.3 (p = 0.01)"]

def robustness(seeds):
    rows = []
    for s in seeds:
        p, wide = B.load(s)
        for scen in SCENS:
            X, sp, ev = inject_var(p, wide, s, scen); D = B.daily(X); valid = X.notna().values
            trial = make_groups(p, "soil x treatment (4)", None)
            res = {"SaQC DFN, trial groups": saqc_groups(D, trial), "SaQC DFN, all plots": B.dr_saqc_all(p, D, TP["drift"]["SaQC drift-from-norm, all plots"]),
                   "Proposed peer trend + gate": B.dr_ours(p, D, SLOPE), "Single-series trend": B.dr_self(p, D, TP["drift"]["Single-series trend"])}
            for k, fl in res.items(): rows.append(dict(seed=s, scenario=scen, task="drift", method=k, **B.drift_scores(fl, ev, D)))
            if scen in ("baseline", "noise x2", "spikes x3.3 (p = 0.01)"):
                for k, (fn, prm) in {"Proposed": (B.sp_ours, TP["spike"]["Proposed"]), "SaQC (range + offset)": (B.sp_saqc, TP["spike"]["SaQC (range + offset)"]),
                                     "Hampel filter": (B.sp_hampel, TP["spike"]["Hampel filter"])}.items():
                    rows.append(dict(seed=s, scenario=scen, task="spike", method=k, **B.prf(fn(p, X, prm), sp, valid)))
        print("robust seed", s, flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    what = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]] or B.TEST
    if what == "A": plausibility()
    elif what == "B": grouping(seeds).to_csv(f"{B.BENCH}/grp_{seeds[0]}_{seeds[-1]}.csv", index=False)
    elif what == "C": robustness(seeds).to_csv(f"{B.BENCH}/rob_{seeds[0]}_{seeds[-1]}.csv", index=False)
