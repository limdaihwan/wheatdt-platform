"""Ingestion layer: the eight-table schema of the platform, loading and validation."""
from pathlib import Path
import pandas as pd

SCHEMA = {
    "plots": ["plot_id", "block", "irrigation_trt", "cultivar", "plot_area", "lat", "lon", "soil_texture", "sowing_date", "season"],
    "weather": ["timestamp", "t_air", "rh", "vpd", "rainfall", "solar_rad", "wind_speed", "et0"],
    "soil_sensor": ["plot_id", "timestamp", "vwc_10cm", "vwc_30cm", "vwc_60cm", "soil_temp_10cm", "soil_ec"],
    "canopy": ["plot_id", "date", "ndvi", "ndre", "canopy_temp", "ctd", "lai", "plant_height", "spad", "source"],
    "phenology": ["plot_id", "date", "zadoks_stage", "gdd_cum"],
    "biomass": ["plot_id", "date", "zadoks_stage", "biomass_dry", "leaf_n"],
    "harvest": ["plot_id", "harvest_date", "grain_yield", "tkw", "spikes_m2", "grain_protein", "harvest_index"],
    "management": ["plot_id", "date", "event_type", "amount", "product"],
}
RANGES = {  # physically plausible ranges: values outside are reported, not dropped
    ("weather", "t_air"): (-25, 50), ("weather", "rh"): (0, 100), ("weather", "vpd"): (0, 8),
    ("soil_sensor", "vwc_10cm"): (0, 70), ("soil_sensor", "vwc_30cm"): (0, 70), ("soil_sensor", "vwc_60cm"): (0, 70),
    ("canopy", "ndvi"): (-1, 1), ("harvest", "grain_yield"): (0, 20),
}
TIME_COLS = {"weather": "timestamp", "soil_sensor": "timestamp"}


class SchemaError(ValueError):
    pass


def load_tables(data_dir):
    d = Path(data_dir)
    tables, report = {}, []
    for name, cols in SCHEMA.items():
        f = d / f"{name}.csv"
        if not f.exists():
            raise SchemaError(f"missing table {f}")
        df = pd.read_csv(f)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SchemaError(f"{name}.csv lacks columns {missing}")
        if name in TIME_COLS:
            df[TIME_COLS[name]] = pd.to_datetime(df[TIME_COLS[name]])
        for (t, c), (lo, hi) in RANGES.items():
            if t == name:
                n = int(((df[c] < lo) | (df[c] > hi)).sum())
                if n:
                    report.append(f"{name}.{c}: {n} values outside [{lo}, {hi}]")
        tables[name] = df
    unknown = set(tables["soil_sensor"].plot_id) - set(tables["plots"].plot_id)
    if unknown:
        raise SchemaError(f"soil_sensor has plots not in plots.csv: {sorted(unknown)[:5]}")
    synthetic = "source" in tables["canopy"] and (tables["canopy"].source == "synthetic").any()
    return tables, {"range_warnings": report, "synthetic": bool(synthetic)}
