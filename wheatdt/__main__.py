"""Command line: python -m wheatdt generate|run|snapshot ..."""
import argparse, json, sys
from .synthetic import generate
from .twin import DigitalTwin
from .dashboard import write_dashboard


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wheatdt", description="Trial-aware wheat field-trial digital twin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write one synthetic season (8 CSV tables + _truth/)")
    g.add_argument("--out", default="data"); g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-faults", action="store_true")
    r = sub.add_parser("run", help="ingest → QC → indicators → dashboard")
    r.add_argument("--data", default="data"); r.add_argument("--out", default="dashboard.html")
    r.add_argument("--state", default="twin_state.json"); r.add_argument("--no-saqc", action="store_true")
    t = sub.add_parser("train", help="train the 7-day water outlook and yield forecast models on synthetic seasons")
    t.add_argument("--work", default="training_seasons"); t.add_argument("--out", default="models")
    t.add_argument("--train-seeds", default="201-224"); t.add_argument("--test-seeds", default="301-310")
    t.add_argument("--procs", type=int, default=2)
    s = sub.add_parser("snapshot", help="print the twin state for one date as JSON")
    s.add_argument("date"); s.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    rng = lambda x: range(int(x.split("-")[0]), int(x.split("-")[1]) + 1)
    if a.cmd == "generate":
        print("wrote", generate(a.out, a.seed, not a.no_faults))
    elif a.cmd == "run":
        tw = DigitalTwin(a.data, use_saqc=not a.no_saqc).update()
        write_dashboard(tw, a.out); tw.save_state(a.state)
        print(f"forecasts: {'on' if tw.forecaster else 'off (run: python -m wheatdt train)'}")
        print(f"QC engine: {tw.qc.engine} | events: {len(tw.qc.events)} | dashboard: {a.out} | state: {a.state}")
        ev = tw.evaluate()
        if ev: print("validation against synthetic ground truth:", json.dumps(ev, default=str))
    elif a.cmd == "train":
        from .train import train_all
        rep = train_all(a.work, a.out, rng(a.train_seeds), rng(a.test_seeds), a.procs)
        print(json.dumps(rep, indent=1))
    elif a.cmd == "snapshot":
        print(json.dumps(DigitalTwin(a.data).update().snapshot(a.date), indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
