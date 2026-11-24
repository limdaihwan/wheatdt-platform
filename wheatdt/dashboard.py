"""Dashboard layer: serialise the twin's state and inject it into the single-file web dashboard."""
import json
from pathlib import Path
import pandas as pd
from .config import QC, ALERTS
from . import __version__

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


def _r(x, n=2):
    return None if pd.isna(x) else round(float(x), n)


def build_payload(twin):
    t, q, des = twin.tables, twin.qc, twin.design
    days = twin.days
    dates = [d.strftime("%Y-%m-%d") for d in days]
    spk = q.soil.assign(date=q.soil.timestamp.dt.normalize()).groupby(["plot_id", "date"]).spike.sum()
    nspk = q.soil.groupby("plot_id").spike.sum()
    can = q.canopy
    plots = []
    for pid, r in des.plots.iterrows():
        g = q.daily[q.daily.plot_id == pid].set_index("date").reindex(days)
        cc = can[can.plot_id == pid]
        hr = t["harvest"][t["harvest"].plot_id == pid]
        h = hr.drop(columns="plot_id").iloc[0].to_dict() if len(hr) else {c: None for c in t["harvest"].columns if c != "plot_id"}
        h = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in h.items()}   # not harvested yet → nulls
        plots.append(dict(
            id=pid, block=int(r.block), trt=r.irrigation_trt, cv=r.cultivar, soil=r.soil_texture, area=float(r.plot_area),
            v10=[_r(x, 1) for x in g.v10], v30=[_r(x, 1) for x in g.v30], v60=[_r(x, 1) for x in g.v60],
            raw=[_r(x, 3) for x in twin.raw[pid]], nobs=[0 if pd.isna(x) else int(x) for x in g.n],
            drift=[bool(x) for x in q.drift[pid].reindex(days).fillna(False)],
            check=[bool(x) for x in q.drift_check[pid].reindex(days).fillna(False)],
            spikes=int(nspk.get(pid, 0)), spk=[int(spk.get((pid, d), 0)) for d in days],
            canopy=[dict(d=x.date, ndvi=x.ndvi, ndre=x.ndre, ctd=x.ctd, lai=x.lai, h=x.plant_height, spad=x.spad, flag=bool(x.ndvi_flag)) for x in cc.itertuples()],
            pheno={int(x.zadoks_stage): x.date for x in t["phenology"][t["phenology"].plot_id == pid].itertuples()},
            biomass=[dict(d=x.date, z=int(x.zadoks_stage), bm=x.biomass_dry, n=x.leaf_n) for x in t["biomass"][t["biomass"].plot_id == pid].itertuples()],
            harvest=h,
            outlook=[_r(x, 3) for x in twin.outlook[0][pid]] if twin.outlook is not None else None,
            warn=[bool(x == 1) for x in twin.outlook[3][pid]] if twin.outlook is not None else None,
            yfc=[dict(point=x.point, d=pd.Timestamp(x.date).strftime("%Y-%m-%d"), pred=_r(x.pred), lo=_r(x.lo), hi=_r(x.hi))
                 for x in twin.yield_fc[twin.yield_fc.plot_id == pid].sort_values("date").itertuples()] if twin.yield_fc is not None else [],
            irrigation=[dict(d=x.date, mm=x.amount) for x in t["management"][(t["management"].plot_id == pid) & (t["management"].event_type == "IRRIGATION")].itertuples()]))
    mg = t["management"]; first = des.plots.plot_id.iloc[0]
    order = {"FULL": 0, "DEFICIT": 1, "RAINFED": 2}
    groups = [dict(name=k, plots=[dict(id=p, soil=des.plots.at[p, "soil_texture"]) for p in v]) for k, v in sorted(q.groups.items(), key=lambda kv: order.get(kv[0], 9))]
    events = [dict(type=e["type"], plot=e["plot"], start=e["start"].strftime("%Y-%m-%d"), end=e["end"].strftime("%Y-%m-%d"), by=e["by"]) for e in q.events]
    return dict(
        version=__version__, synthetic=twin.ingest_report["synthetic"], dates=dates,
        weather={k: [_r(x) for x in twin.wd[k]] for k in twin.wd.columns}, plots=plots,
        soilParams=des.soil,
        mgmt=[dict(d=x.date, type=x.event_type, amount=x.amount, product=x.product) for x in mg[mg.plot_id == first].itertuples() if x.event_type != "IRRIGATION"],
        models=_model_summary(twin), alerts=ALERTS,
        qc=dict(engine=q.engine, groups=groups, params=QC, events=events, validation=twin.evaluate(),
                warnings=twin.ingest_report["range_warnings"]))


def write_dashboard(twin, out_path):
    payload = json.dumps(build_payload(twin), separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA_JSON__", payload)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


def _model_summary(twin):
    f = twin.forecaster
    if f is None or not f.report:
        return None
    r = f.report
    return dict(warn=round(f.warn_threshold, 3), rule=f.b["stress_interval"].get("warn_rule"), selected={k: v["Selected model"]["name"] for k, v in r["yield_forecast"]["models"].items()},
                trained=r["trained_on"], tested=r["tested_on"], caveat=r["caveat"], horizon=r["stress_outlook"]["horizon_days"],
                stress=r["stress_outlook"]["models"], yld=r["yield_forecast"]["models"], points=r["yield_forecast"]["points"],
                yinterval={k: [round(v["q10"], 2), round(v["q90"], 2)] for k, v in r["yield_forecast"]["interval"].items()})
