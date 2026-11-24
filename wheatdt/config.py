"""Platform constants. Replace soil hydraulic parameters with measured values when field data arrive."""
SOIL_HYDRAULICS = {            # field capacity / wilting point, % vol.
    "clay loam": {"fc": 36.0, "wp": 18.0},
    "loam": {"fc": 30.0, "wp": 12.0},
}
RAW_WEIGHTS = {"vwc_10cm": 0.40, "vwc_30cm": 0.35, "vwc_60cm": 0.25}
GDD_BASE_C = 0.0               # base temperature for thermal time; sowing date comes from plots.csv

QC = {
    "spike_k": 4.0,              # % vol. from 13 h rolling median (tuned on 5 seasons)
    "spike_window_h": 13,
    "range_margin": 10.0,        # flag above FC + margin
    "dfn_window": "7D",          # SaQC drift-from-norm
    "dfn_spread": 3.0,           # on available water, treatment groups
    "dfn_frac": 0.5,
    "pt_slope": 0.15,            # peer-trend rule, % vol. per day
    "pt_gate": 0.15,
    "pt_r": 0.85,
    "ndvi_ratio": 0.75,
    "ndvi_exclude_before_maturity_d": 7,
}
ALERTS = {                     # used by twin.snapshot() and by the dashboard (passed in the payload)
    "raw_severe": 0.25,        # root-zone available water, fraction
    "raw_moderate": 0.45,
    "vpd_high": 3.0,           # kPa, grain filling
    "heat_c": 32.0,            # °C, grain filling
    "frost_c": -2.0,           # °C, stem elongation to anthesis
}
