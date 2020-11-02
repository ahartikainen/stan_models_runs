import gzip
import os
import pickle
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.resources import INLINE


def get_results(root_path):
    draws = 0
    res = []
    i = 0
    root_path = Path(root_path)
    for path in glob(str(root_path / "fit_results" / "*.pickle.gz")):
        with gzip.open(path, mode="rb") as f:
            data = pickle.load(f)

        for name, info in data.items():
            if name == "FAIL_INFO":
                continue

            timing = info.pop("stan_timing", None)
            if timing is not None:
                info["timing_warmup"] = timing["warm-up"].tolist()
                info["timing_sampling"] = timing["sampling"].tolist()

            if not draws:
                draws = info.get("draws", 0)
                warmup_draws = info.get("warmup_draws", 0)
                chains = info.get("chains", 0)

            m_sec, f_sec = info.pop("cmdstanpy_model_duration", None), info.pop(
                "cmdstanpy_fit_duration", None
            )
            res.append(
                {
                    "duration_model": m_sec if m_sec is not None else 1e-2,
                    "duration_fit": f_sec if f_sec is not None else 1e-2,
                    "name": name,
                    "i": i,
                    "fit_color": "lime" if f_sec is not None else "darkgrey",
                    **info,
                }
            )
            i += 1

    results = pd.DataFrame(res)
    cds = ColumnDataSource(results)

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("Model name", "@name"),
        ("Index", "@i"),
        ("Model Duration", "@duration_model{1.1}"),
        ("Fit Duration", "@duration_fit{1.1}"),
        ("Timing_warmup", "@timing_warmup"),
        ("Timing_sampling", "@timing_sampling"),
        ("Gradient timing reported", "@stan_gradient_timing"),
        ("Gradient timing calculated", "@stan_gradient_timing_sampling"),
        ("Divergent (sum)", "@n_divergent"),
        ("Tree Depth (sum)", "@n_tree_depth"),
        ("Max tree-depth (sum)", "@n_max_tree"),
        ("Leapfrogs (sum)", "@n_leapfrogs"),
        ("Min ESS_bulk/draw", "@min_ess_bulk_per_draw"),
        ("Min ESS_bulk/second", "@min_ess_bulk_per_second"),
        ("Min ESS_tail/draw", "@min_ess_tail_per_draw"),
        ("Min ESS_tail/second", "@min_ess_tail_per_second"),
        ("Stepsize", "@stepsize"),
        ("Warmup draws", "@warmup_draws"),
        ("Draws", "@draws"),
        ("Chains", "@chains"),
    ]

    p_model_time = figure(
        sizing_mode="stretch_both",
        max_height=400,
        output_backend="webgl",
        title=f"Model compilation comparison: Updated {datetime.utcnow().date()} {datetime.utcnow():%H:%M} UTC",
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset",
    )
    p_model_time.add_tools(hover_tool)

    p_model_time.circle(
        x="i", y="duration_model", size=10, color="dodgerblue", source=cds
    )

    p_model_time.yaxis.axis_label = "Compilation time (seconds)"
    p_model_time.xaxis.axis_label = "Model names (see hover)"

    p_fit_time = figure(
        sizing_mode="stretch_both",
        max_height=400,
        output_backend="webgl",
        title=f"Sampling comparison (warmup: {warmup_draws}, draws: {draws}, chains: {chains})",
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset",
        y_axis_type="log",
    )
    p_fit_time.add_tools(hover_tool)

    p_fit_time.circle(x="i", y="duration_fit", size=10, source=cds, color="fit_color")

    p_fit_time.yaxis.axis_label = "Sampling time (seconds)"
    p_fit_time.xaxis.axis_label = "Models (see model info with hover)"

    panel = pn.Column(p_model_time, p_fit_time, sizing_mode="stretch_both")

    os.makedirs("./results")
    panel.save("results/posteriordb_sampling_stan.html", resources=INLINE)


if __name__ == "__main__":
    print(datetime.utcnow())
    get_results("..")
    print(datetime.utcnow())
