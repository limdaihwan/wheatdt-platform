"""Trial-design metadata held by the virtual twin and passed to QC (the core idea of the paper)."""
from dataclasses import dataclass, field
import pandas as pd
from .config import SOIL_HYDRAULICS


@dataclass
class TrialDesign:
    plots: pd.DataFrame
    soil: dict = field(default_factory=lambda: dict(SOIL_HYDRAULICS))

    def __post_init__(self):
        self.plots = self.plots.set_index("plot_id", drop=False)
        bad = set(self.plots.soil_texture) - set(self.soil)
        if bad:
            raise ValueError(f"no FC/WP for soil textures {bad}; add them to the trial metadata")

    def fc_wp(self, pid):
        s = self.soil[self.plots.at[pid, "soil_texture"]]
        return s["fc"], s["wp"]

    def available_water(self, pid, vwc):
        fc, wp = self.fc_wp(pid)
        return (vwc - wp) / (fc - wp) * 100.0

    def peer_groups(self, by=("irrigation_trt",)):
        """Groups of plots that should behave alike. Default: treatment groups (used on available water)."""
        return {k if isinstance(k, str) else " × ".join(k): list(g.plot_id)
                for k, g in self.plots.groupby(list(by))}

    def peers_of(self, pid, by=("irrigation_trt", "soil_texture")):
        r = self.plots.loc[pid]
        m = (self.plots[list(by)] == r[list(by)]).all(axis=1) & (self.plots.plot_id != pid)
        return list(self.plots[m].plot_id)
