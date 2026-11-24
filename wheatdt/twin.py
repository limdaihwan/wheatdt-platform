"""The virtual twin: per-plot state, design metadata, QC flags and indicators, kept together."""
from pathlib import Path
import json
import pandas as pd
from .ingest import load_tables
from .metadata import TrialDesign
from .qc import run_qc
from . import indicators as ind
from .models import Forecaster


class DigitalTwin:
    def __init__(self, data_dir, soil=None, use_saqc=True, models_dir=None, forecasts=True):
        self.data_dir = Path(data_dir)
        self.tables, self.ingest_report = load_tables(data_dir)
        self.design = TrialDesign(self.tables["plots"], **({"soil": soil} if soil else {}))
        self.sowing = self.design.plots.sowing_date.iloc[0]
        self.wd = ind.daily_weather(self.tables["weather"])
        self.days = self.wd.index
        self.use_saqc = use_saqc
        self.qc = None
        self.forecaster = Forecaster.load(models_dir) if forecasts else None
        self.outlook = self.yield_fc = None

    # ------------------------------------------------------------------ update cycle
    def update(self):
        """Run QC and recompute indicators from the current tables (one sync of physical → virtual)."""
        self.qc = run_qc(self.tables, self.design, self.days, self.use_saqc)
        self.raw = ind.root_zone_water(self.qc.daily, self.design).reindex(self.days)
        self.gdd = ind.thermal_time(self.wd, self.sowing)
        self.zadoks = ind.zadoks(self.tables["phenology"], self.tables["harvest"], self.days, self.sowing)
        if self.forecaster is not None:
            self.outlook = self.forecaster.stress_outlook(self)        # (pred, lo, hi)
            self.yield_fc = self.forecaster.yield_forecasts(self)
        return self

    # ------------------------------------------------------------------ queries
    def snapshot(self, date):
        d = pd.Timestamp(date).normalize()
        dly = self.qc.daily[self.qc.daily.date == d].set_index("plot_id")
        state = {}
        for pid in self.design.plots.plot_id:
            fc, wp = self.design.fc_wp(pid)
            state[pid] = dict(
                treatment=self.design.plots.at[pid, "irrigation_trt"], soil=self.design.plots.at[pid, "soil_texture"],
                vwc={k: _r(dly.at[pid, k]) for k in ("v10", "v30", "v60")}, root_zone_water=_r(self.raw.at[d, pid], 3),
                zadoks=_r(self.zadoks.at[d, pid], 0), readings=int(dly.at[pid, "n"]) if pd.notna(dly.at[pid, "n"]) else 0,
                drift_flag=bool(self.qc.drift.at[d, pid]), fc=fc, wp=wp)
        al = ind.alerts(d, self.raw, self.zadoks, self.wd)
        if self.outlook is not None:
            pred = self.outlook[0].loc[d]
            wf = self.outlook[3].loc[d]
            soon = [p for p in wf.index if wf[p] == 1]
            if soon:
                al.append(dict(severity="warning", title="Water stress expected within 7 days", plots=soon))
            for p in state:
                state[p]["root_zone_water_in_7_days"] = _r(pred.get(p), 3)
                if self.yield_fc is not None:
                    f = self.yield_fc[(self.yield_fc.plot_id == p) & (pd.to_datetime(self.yield_fc.date) <= d)]
                    if len(f):
                        r = f.sort_values("date").iloc[-1]
                        state[p]["yield_forecast"] = dict(point=r.point, t_ha=_r(r.pred, 2), interval_80=[_r(r.lo, 2), _r(r.hi, 2)])
        return dict(date=str(d.date()), thermal_time=_r(self.gdd.at[d], 0), alerts=al, plots=state)

    def evaluate(self):
        """Compare QC output with the synthetic ground truth, when _truth/ exists."""
        t = self.data_dir / "_truth" / "data_quality_events.csv"
        if not t.exists():
            return None
        ev = pd.read_csv(t)
        res = {}
        for r in ev.itertuples():
            if str(r.type).startswith("sensor_drift"):
                win = self.qc.drift.loc[r.start:r.end, r.plot_id]
                others = self.qc.drift.drop(columns=[r.plot_id]).to_numpy().sum()
                res["drift"] = dict(plot=r.plot_id, window=f"{r.start}–{r.end}", detected=bool(win.any()),
                                    days_flagged=int(win.sum()), false_positive_plot_days=int(others))
            elif r.type == "logger_offline":
                n = self.qc.daily[(self.qc.daily.plot_id == r.plot_id) & (self.qc.daily.date.between(r.start, r.end))].n
                res["outage"] = dict(plot=r.plot_id, detected=bool((n.isna() | (n == 0)).all()))
            elif str(r.type).startswith("random_spike"):
                n_true = int(str(r.type).split("(")[1].split()[0])
                res["spikes"] = dict(injected=n_true, flagged=int(self.qc.soil.spike.sum()))
            elif str(r.type).startswith("reflectance_error"):
                rows = [int(x) for x in str(r.type).split("rows ")[1].split(",")]
                can = pd.read_csv(self.data_dir / "canopy.csv").iloc[rows][["plot_id", "date"]]
                f = self.qc.canopy.set_index(["plot_id", "date"]).ndvi_flag
                hit = sum(bool(f.get((a, b), False)) for a, b in can.itertuples(index=False))
                res["ndvi"] = dict(injected=len(rows), detected=hit, flagged=int(f.sum()))
        return res

    def summary(self):
        return dict(plots=len(self.design.plots), days=len(self.days), qc_engine=self.qc.engine,
                    forecasts=self.forecaster is not None,
                    peer_groups=self.qc.groups, events=len(self.qc.events), ingest=self.ingest_report)

    def save_state(self, path):
        Path(path).write_text(json.dumps(self.summary() | {"evaluation": self.evaluate()}, indent=1, default=str))


def _r(x, n=1):
    return None if pd.isna(x) else round(float(x), n)
