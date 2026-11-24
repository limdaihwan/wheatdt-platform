"""
QC benchmark: proposed trial-aware QC vs. standard methods and SaQC.
Fault-free synthetic seasons (seed dirs) -> inject labelled faults -> run methods -> score.
Tuning seeds: 1-5. Test seeds: 101-120. Parameters are chosen on tuning seeds only.
"""
import os, sys, json, warnings, itertools, time
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import saqc
from saqc import SaQC

BENCH = "/home/claude/bench"
HYD = {"clay loam": (36.0, 18.0), "loam": (30.0, 12.0)}
TUNE = [1, 2, 3, 4, 5]
TEST = list(range(101, 121))

# ---------------------------------------------------------------- data + faults
def load(seed):
    d = f"{BENCH}/s{seed}"
    p = pd.read_csv(f"{d}/plots.csv")
    s = pd.read_csv(f"{d}/soil_sensor.csv", usecols=["plot_id", "timestamp", "vwc_10cm"], parse_dates=["timestamp"])
    wide = s.pivot(index="timestamp", columns="plot_id", values="vwc_10cm")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return p, wide.reindex(full)

def inject(p, wide, seed, spike_rate=0.003, n_drift=3):
    rng = np.random.default_rng(seed + 1000)
    X = wide.copy()
    spike = pd.DataFrame(False, index=X.index, columns=X.columns)
    drift_h = pd.DataFrame(False, index=X.index, columns=X.columns)
    events = []
    # drift: gradual ramp on 10 cm probe, then probe re-seated (returns to normal)
    for pid in rng.choice(X.columns, n_drift, replace=False):
        start = pd.Timestamp("2024-12-15") + pd.Timedelta(days=int(rng.integers(0, 150)))
        days = int(rng.integers(10, 22)); slope = float(rng.uniform(0.15, 0.40)) * rng.choice([-1, 1])
        idx = (X.index >= start) & (X.index < start + pd.Timedelta(days=days))
        ramp = slope * (X.index[idx] - start).total_seconds() / 86400
        X.loc[idx, pid] = X.loc[idx, pid] + ramp
        drift_h.loc[idx, pid] = True
        events.append(dict(plot=pid, start=start.normalize(), days=days, slope=slope))
    # spikes: positive (x1.25-1.9) and negative (x0.45-0.75) single-hour errors
    obs = X.notna().values
    m = (rng.random(X.shape) < spike_rate) & obs
    pos = rng.random(X.shape) < 0.7
    fac = np.where(pos, rng.uniform(1.25, 1.9, X.shape), rng.uniform(0.45, 0.75, X.shape))
    X = X.where(~m, X * fac)
    spike = pd.DataFrame(m, index=X.index, columns=X.columns)
    return X, spike, drift_h, events

# ---------------------------------------------------------------- spike methods (hourly)
def fcwp(p, pid):
    return HYD[p.set_index("plot_id").loc[pid, "soil_texture"]]

def roll_med(x, w=13):
    return x.rolling(w, center=True, min_periods=5).median()

def sp_range(p, X, k):
    out = pd.DataFrame(False, index=X.index, columns=X.columns)
    for pid in X.columns:
        fc, wp = fcwp(p, pid); x = X[pid]
        out[pid] = (x > fc + k) | (x < wp - k)
    return out

def sp_hampel(p, X, t):
    out = pd.DataFrame(False, index=X.index, columns=X.columns)
    for pid in X.columns:
        x = X[pid]; med = roll_med(x)
        mad = (x - med).abs().rolling(13, center=True, min_periods=5).median() * 1.4826
        out[pid] = (x - med).abs() > t * mad.clip(lower=0.05)
    return out

def sp_saqc(p, X, thresh):
    out = pd.DataFrame(False, index=X.index, columns=X.columns)
    for pid in X.columns:
        fc, wp = fcwp(p, pid)
        x = X[pid].dropna()
        qc = SaQC(pd.DataFrame({"v": x}))
        qc = qc.flagRange("v", min=wp - 10, max=fc + 10)
        qc = qc.flagOffset("v", window="3h", thresh=thresh, tolerance=thresh / 2)
        f = qc.flags["v"] > 0  # BAD
        out.loc[f.index, pid] = f.values
    return out

def sp_ours(p, X, k):
    out = pd.DataFrame(False, index=X.index, columns=X.columns)
    for pid in X.columns:
        fc, wp = fcwp(p, pid); x = X[pid]
        out[pid] = (x > fc + 10) | ((x - roll_med(x)).abs() > k)
    return out

SPIKE_METHODS = {
    "Range check": (sp_range, [0, 2, 4, 6, 8, 10, 15]),
    "Hampel filter": (sp_hampel, [3, 4, 5, 6, 8, 10, 15, 20]),
    "SaQC (range + offset)": (sp_saqc, [1, 2, 3, 4, 6, 8]),
    "Proposed": (sp_ours, [2, 3, 4, 5, 6, 8]),
}

def prf(flag, truth, valid):
    f = flag.values & valid; t = truth.values & valid
    tp = (f & t).sum(); fp = (f & ~t).sum(); fn = (~f & t).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)

# ---------------------------------------------------------------- drift methods (daily)
def daily(X):
    return X.resample("D").median()

def peers_of(p, pid):
    r = p.set_index("plot_id").loc[pid]
    q = p[(p.soil_texture == r.soil_texture) & (p.irrigation_trt == r.irrigation_trt) & (p.plot_id != pid)]
    return list(q.plot_id)

def trend(v):
    m = ~np.isnan(v)
    if m.sum() < 5: return 0.0
    x = np.arange(len(v))[m]; y = v[m]
    if np.std(y) == 0: return 0.0
    sl = np.polyfit(x, y, 1)[0]; r = np.corrcoef(x, y)[0, 1]
    return sl if abs(r) > 0.85 else 0.0

def slope_series(s):
    return s.rolling(7, min_periods=5).apply(lambda v: np.polyfit(np.arange(len(v))[~np.isnan(v)], v[~np.isnan(v)], 1)[0]
                                            if (~np.isnan(v)).sum() >= 3 else 0.0, raw=True)

def expand(flag):  # mark the whole 7-day window that triggered
    return flag.astype(float)[::-1].rolling(7, min_periods=1).max()[::-1].astype(bool)

SOW = pd.Timestamp("2024-11-22")  # start evaluation one week after sowing

def dr_self(p, D, s):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    for pid in D.columns:
        sl = D[pid].rolling(7, min_periods=5).apply(trend, raw=True)
        out[pid] = expand(sl.abs() > s)
    return out

def dr_peer(p, D, s, gate=None):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    for pid in D.columns:
        pe = peers_of(p, pid); pm = D[pe].median(axis=1)
        res = D[pid] - pm
        sl = res.rolling(7, min_periods=5).apply(trend, raw=True)
        fl = sl.abs() > s
        if gate is not None:
            fl &= slope_series(pm).abs() < gate
        out[pid] = expand(fl)
    return out

def dr_ours(p, D, s):  return dr_peer(p, D, s, gate=0.15)
def dr_nogate(p, D, s): return dr_peer(p, D, s, gate=None)

def dr_saqc(p, D, spread):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    groups = p.groupby(["soil_texture", "irrigation_trt"]).plot_id.apply(list)
    for g in groups:
        df = D[g].copy(); df.columns = [f"f_{c}" for c in g]
        qc = SaQC(df.interpolate(limit=3))
        qc = qc.flagDriftFromNorm([f"f_{c}" for c in g], window="7D", spread=spread, frac=0.5)
        for c in g:
            out[c] = (qc.flags[f"f_{c}"] > 0).reindex(D.index).fillna(False).values
    return out

def dr_saqc_all(p, D, spread):
    out = pd.DataFrame(False, index=D.index, columns=D.columns)
    g = list(p.plot_id)
    df = D[g].copy(); df.columns = [f"f_{c}" for c in g]
    qc = SaQC(df.interpolate(limit=3))
    qc = qc.flagDriftFromNorm([f"f_{c}" for c in g], window="7D", spread=spread, frac=0.5)
    for c in g:
        out[c] = (qc.flags[f"f_{c}"] > 0).reindex(D.index).fillna(False).values
    return out

DRIFT_METHODS = {
    "Single-series trend": (dr_self, [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]),
    "SaQC drift-from-norm": (dr_saqc, [0.25, 0.5, 0.75, 1, 1.5, 2, 3]),
    "SaQC drift-from-norm, all plots": (dr_saqc_all, [0.5, 1, 2, 3, 5, 8, 12]),
    "Peer trend, no gating": (dr_nogate, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
    "Proposed": (dr_ours, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
}

def drift_scores(flag, events, D):
    ev_mask = pd.DataFrame(False, index=D.index, columns=D.columns)
    slack = pd.DataFrame(False, index=D.index, columns=D.columns)
    for e in events:
        win = (D.index >= e["start"]) & (D.index < e["start"] + pd.Timedelta(days=e["days"]))
        ev_mask.loc[win, e["plot"]] = True
        sl = (D.index >= e["start"] - pd.Timedelta(days=6)) & (D.index < e["start"] + pd.Timedelta(days=e["days"] + 4))
        slack.loc[sl, e["plot"]] = True
    valid = (D.index >= SOW)[:, None] & np.ones(D.shape, bool)
    f = flag.values & valid
    tp_days = (f & ev_mask.values).sum(); fp_days = (f & ~slack.values).sum(); fn_days = (~f & ev_mask.values & valid).sum()
    pr = tp_days / (tp_days + fp_days) if tp_days + fp_days else 0.0
    rc = tp_days / (tp_days + fn_days) if tp_days + fn_days else 0.0
    det, delays = 0, []
    for e in events:
        win = (D.index >= e["start"]) & (D.index < e["start"] + pd.Timedelta(days=e["days"] + 3))
        hit = flag.loc[win, e["plot"]]
        if hit.any():
            det += 1; delays.append((hit[hit].index[0] - e["start"]).days)
    # false-alarm episodes (runs outside slack)
    fa = 0
    for pid in D.columns:
        x = (flag[pid].values & valid[:, 0] & ~slack[pid].values).astype(int)
        fa += int(((np.diff(np.r_[0, x]) == 1)).sum())
    return dict(day_precision=pr, day_recall=rc, day_f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                event_recall=det / len(events), false_alarms=fa, delay=np.mean(delays) if delays else np.nan)

# ---------------------------------------------------------------- run
def run(seeds, spike_params=None, drift_params=None):
    rows = []
    for seed in seeds:
        p, wide = load(seed)
        X, spike, drift_h, events = inject(p, wide, seed)
        valid = X.notna().values
        # spikes
        for name, (fn, grid) in SPIKE_METHODS.items():
            for prm in ([spike_params[name]] if spike_params else grid):
                t = time.time(); fl = fn(p, X, prm)
                rows.append(dict(task="spike", seed=seed, method=name, param=prm, sec=time.time() - t, **prf(fl, spike, valid)))
        # drift on daily medians (robust to single spikes)
        D = daily(X)
        for name, (fn, grid) in DRIFT_METHODS.items():
            for prm in ([drift_params[name]] if drift_params else grid):
                t = time.time(); fl = fn(p, D, prm)
                rows.append(dict(task="drift", seed=seed, method=name, param=prm, sec=time.time() - t, **drift_scores(fl, events, D)))
        print("seed", seed, "done", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    tune = run(TUNE)
    tune.to_csv(f"{BENCH}/tuning_results.csv", index=False)
    best_sp = tune[tune.task == "spike"].groupby(["method", "param"]).f1.mean().reset_index().sort_values("f1").groupby("method").tail(1)
    best_dr = tune[tune.task == "drift"].groupby(["method", "param"]).day_f1.mean().reset_index().sort_values("day_f1").groupby("method").tail(1)
    sp_params = dict(zip(best_sp.method, best_sp.param)); dr_params = dict(zip(best_dr.method, best_dr.param))
    json.dump(dict(spike=sp_params, drift=dr_params), open(f"{BENCH}/tuned_params.json", "w"), indent=1, default=float)
    print("tuned", sp_params, dr_params)
    test = run(TEST, sp_params, dr_params)
    test.to_csv(f"{BENCH}/test_results.csv", index=False)
    print(test.groupby(["task", "method"]).mean(numeric_only=True).round(3))
