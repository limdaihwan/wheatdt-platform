"""Train and evaluate the forecasting models on synthetic seasons.

Training seasons and test seasons use different seeds and never mix. The ET0 climatology used as a
stand-in for a weather forecast is computed from the training seasons only.
"""
import json, os, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from .synthetic import generate
from . import features as F

warnings.filterwarnings("ignore")


def _gen(args):
    out, seed = args
    if not (Path(out) / "harvest.csv").exists():
        arc = Path(out).parent / "archives" / f"{Path(out).name}.zip"
        if arc.exists():                       # season shipped with the repo: unpack instead of regenerating
            import zipfile
            with zipfile.ZipFile(arc) as z:
                z.extractall(Path(out).parent)
        else:
            generate(out, seed=seed, faults=True)
    return out


def _cached(path):
    """Return the cached table, accepting a gzip copy (as shipped in the repo)."""
    p = Path(path)
    return p if p.exists() else (p.with_name(p.name + ".gz") if p.with_name(p.name + ".gz").exists() else None)


def _daily_et0(d):
    w = pd.read_csv(Path(d) / "weather.csv", parse_dates=["timestamp"])
    return w.set_index("timestamp").resample("D").et0.first().to_frame()


def _frames(args):
    d, clim, seed = args
    from .twin import DigitalTwin
    t = DigitalTwin(d).update()
    s = F.stress_frame(t, clim).assign(seed=seed)
    y = F.yield_frame(t).assign(seed=seed)
    return s, y


def build_dataset(train_seeds, test_seeds, workdir, procs=2, reuse=True):
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    cached = [work / "stress_rows.csv", work / "yield_rows.csv", work / "clim_et0.json"]
    found = [_cached(c) for c in cached]
    if reuse and all(found):
        st = pd.read_csv(found[0], parse_dates=["date"]); yl = pd.read_csv(found[1], parse_dates=["date"])
        if set(st.seed) >= set(train_seeds) | set(test_seeds):
            return st, yl, json.load(open(found[2]))
    jobs = [(str(work / f"s{s}"), s) for s in list(train_seeds) + list(test_seeds)]
    with Pool(procs) as p:
        p.map(_gen, jobs)
    clim = F.climatology_et0([_daily_et0(work / f"s{s}") for s in train_seeds])
    with Pool(procs) as p:
        res = p.map(_frames, [(str(work / f"s{s}"), clim, s) for s in list(train_seeds) + list(test_seeds)])
    stress = pd.concat([r[0] for r in res], ignore_index=True)
    yld = pd.concat([r[1] for r in res], ignore_index=True)
    stress.to_csv(work / "stress_rows.csv", index=False); yld.to_csv(work / "yield_rows.csv", index=False)
    json.dump(clim, open(work / "clim_et0.json", "w"))
    return stress, yld, clim


# ---------------------------------------------------------------- models
def _gbr(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0, **kw)


def _alert_scores(y_true, y_pred, thr):
    t, p = y_true < thr, y_pred < thr
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, events=int(t.sum()))


def train_stress(stress, train_seeds, test_seeds):
    tr, te = stress[stress.seed.isin(train_seeds)], stress[stress.seed.isin(test_seeds)]
    m = _gbr().fit(tr[F.STRESS_FEATURES], tr.target)
    preds = {"Persistence (no change)": te.raw.values,
             "Water-balance projection": te.raw_proj.values,
             "Hybrid model (water balance + gradient boosting)": np.clip(m.predict(te[F.STRESS_FEATURES]), 0, 1)}
    # warning threshold on the forecast, tuned on out-of-fold training predictions only
    oof = np.clip(_oof_pred(tr, F.STRESS_FEATURES, train_seeds, _gbr), 0, 1)
    fr_tr = (tr.raw >= 0.45).values
    grid = np.round(np.arange(0.40, 0.601, 0.005), 3)
    f1s = [_alert_scores(tr.target.values[fr_tr], oof[fr_tr] - (t - 0.45), 0.45)["f1"] for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    # only days on which stress is not already present: this is where an early warning has value
    fresh = (te.raw >= 0.45).values
    rep = {}
    for k, p in preds.items():
        err = p - te.target.values
        rep[k] = dict(mae=round(float(np.abs(err).mean()), 4), rmse=round(float(np.sqrt((err ** 2).mean())), 4),
                      new_stress_warning=_alert_scores(te.target.values[fresh], p[fresh], 0.45))
    hk = "Hybrid model (water balance + gradient boosting)"
    p = preds[hk]
    rep[hk]["new_stress_warning"] = _alert_scores(te.target.values[fresh], p[fresh] - (thr - 0.45), 0.45)
    # choose the warning rule on training seasons: projection, tuned hybrid, or either
    def rule_flags(raw_proj, hyb, rule):
        a, b = raw_proj < 0.45, hyb < thr
        return {"projection": a, "hybrid": b, "either": a | b}[rule]
    yt = tr.target.values[fr_tr] < 0.45
    def f1(flag, truth):
        tp = (flag & truth).sum(); pr = tp / max(flag.sum(), 1); rc = tp / max(truth.sum(), 1)
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    rule_cv = {r: round(float(f1(rule_flags(tr.raw_proj.values[fr_tr], oof[fr_tr], r), yt)), 3) for r in ("projection", "hybrid", "either")}
    rule = max(rule_cv, key=rule_cv.get)
    w = rule_flags(te.raw_proj.values[fresh], p[fresh], rule); truth = te.target.values[fresh] < 0.45
    tp = int((w & truth).sum()); pr = tp / max(int(w.sum()), 1); rc = tp / max(int(truth.sum()), 1)
    rep["Early warning rule used by the platform"] = dict(rule=rule, rule_f1_train=rule_cv, mae=None, rmse=None,
        new_stress_warning=dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, events=int(truth.sum())))
    resid = tr.target.values - oof
    return m, rep, dict(q10=float(np.quantile(resid, 0.10)), q90=float(np.quantile(resid, 0.90)), n_train=len(tr), n_test=len(te),
                        warn_threshold=thr, warn_rule=rule)


def _oof_pred(tr, feats, seeds, make, k=4):
    """Out-of-fold predictions by season (positional, so duplicate row indexes are safe)."""
    seeds = list(seeds); out = np.full(len(tr), np.nan)
    for f in [seeds[i::k] for i in range(k)]:
        m = tr.seed.isin(f).to_numpy()
        a, b = tr[~m], tr[m]
        out[m] = make().fit(a[feats], a.target).predict(b[feats])
    return out


def _oof_residuals(tr, feats, seeds, k=4):
    seeds = list(seeds); folds = [seeds[i::k] for i in range(k)]; res = []
    for f in folds:
        a, b = tr[~tr.seed.isin(f)], tr[tr.seed.isin(f)]
        res.append(b.target.values - _gbr().fit(a[feats], a.target).predict(b[feats]))
    return np.concatenate(res)


class GroupMean:
    """Baseline: mean yield of the plot's treatment × cultivar group in the training seasons."""
    def fit(self, X, y):
        d = X[["trt", "cv"]].assign(y=np.asarray(y)); self.m = d.groupby(["trt", "cv"]).y.mean(); self.g = float(np.mean(y)); return self

    def predict(self, X):
        return np.array([self.m.get((t, c), self.g) for t, c in zip(X.trt, X.cv)])


def _ridge():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def train_yield(yld, train_seeds, test_seeds):
    models, rep, unc = {}, {}, {}
    for point in F.YIELD_POINTS:
        d = yld[yld.point == point]
        tr, te = d[d.seed.isin(train_seeds)], d[d.seed.isin(test_seeds)]
        trt_mean = tr.groupby(["trt", "cv"]).target.mean()
        base = te.apply(lambda r: trt_mean.get((r.trt, r.cv), tr.target.mean()), axis=1).values
        ridge = _ridge().fit(tr[F.YIELD_FEATURES], tr.target)
        gbr = _gbr().fit(tr[F.YIELD_FEATURES], tr.target)
        preds = {"Treatment × cultivar mean": base, "Linear (ridge)": ridge.predict(te[F.YIELD_FEATURES]), "Gradient boosting": gbr.predict(te[F.YIELD_FEATURES])}
        oof = {"Treatment × cultivar mean": _oof_pred(tr, F.YIELD_FEATURES, train_seeds, GroupMean),
               "Linear (ridge)": _oof_pred(tr, F.YIELD_FEATURES, train_seeds, _ridge),
               "Gradient boosting": _oof_pred(tr, F.YIELD_FEATURES, train_seeds, _gbr)}
        cv_rmse = {k: float(np.sqrt(((v - tr.target.values) ** 2).mean())) for k, v in oof.items()}
        chosen = min(cv_rmse, key=cv_rmse.get)
        r = {}
        for k, p in preds.items():
            e = p - te.target.values
            ss = ((te.target - te.target.mean()) ** 2).sum()
            r[k] = dict(rmse=round(float(np.sqrt((e ** 2).mean())), 3), mae=round(float(np.abs(e).mean()), 3),
                        r2=round(float(1 - (e ** 2).sum() / ss), 3), bias=round(float(e.mean()), 3))
        r["Selected model"] = dict(r[chosen], name=chosen, cv_rmse_train=round(cv_rmse[chosen], 3))
        rep[point] = r
        models[point] = {"Linear (ridge)": ridge, "Gradient boosting": gbr}.get(chosen) or GroupMean().fit(tr[F.YIELD_FEATURES], tr.target)
        res = tr.target.values - oof[chosen]
        unc[point] = dict(q10=float(np.quantile(res, 0.10)), q90=float(np.quantile(res, 0.90)), n_train=len(tr), n_test=len(te), model=chosen)
    return models, rep, unc


def train_all(workdir, out_dir, train_seeds=range(201, 225), test_seeds=range(301, 311), procs=2):
    import joblib
    from . import __version__
    train_seeds, test_seeds = list(train_seeds), list(test_seeds)
    stress, yld, clim = build_dataset(train_seeds, test_seeds, workdir, procs)
    sm, srep, sunc = train_stress(stress, train_seeds, test_seeds)
    ym, yrep, yunc = train_yield(yld, train_seeds, test_seeds)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(stress=sm, yield_models=ym, clim=clim, stress_interval=sunc, yield_interval=yunc,
                     horizon=F.HORIZON, features=dict(stress=F.STRESS_FEATURES, yield_=F.YIELD_FEATURES),
                     version=__version__), out / "forecast_models.joblib")
    report = dict(trained_on=f"{len(train_seeds)} synthetic seasons (seeds {train_seeds[0]}–{train_seeds[-1]})",
                  tested_on=f"{len(test_seeds)} held-out synthetic seasons (seeds {test_seeds[0]}–{test_seeds[-1]})",
                  stress_outlook=dict(horizon_days=F.HORIZON, models=srep, interval=sunc),
                  yield_forecast=dict(points=F.YIELD_POINTS, models=yrep, interval=yunc),
                  caveat="Synthetic data come from the platform's own crop–soil generator, so accuracy here verifies the "
                         "pipeline and is an upper bound; it is not agronomic validation.")
    json.dump(report, open(out / "model_report.json", "w"), indent=1)
    return report
