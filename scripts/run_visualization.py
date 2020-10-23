import gzip
import os
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.resources import INLINE


def get_results(root_path):
    res = []
    i = 0
    root_path = Path(root_path)
    for path in glob(str(root_path / "fit_results" / "*.pickle.gz")):
        with gzip.open(path, mode="rb") as f:
            data = pickle.load(f)

        for name, info in data.items():
            if name == "FAIL_INFO":
                continue

            m_sec, f_sec = info["duration_model_seconds"], info["duration_fit_seconds"]
            res.append(
                {"duration_model": m_sec, "duration_fit": f_sec, "name": name, "i": i}
            )
            i += 1

    results = pd.DataFrame(res)
    cds = ColumnDataSource(results)

    hover_tool = HoverTool()
    hover_tool.tooltips = [
        ("Model name", "@name"),
        ("Index", "@i"),
        ("Model Duration", "@duration_model"),
        ("Fit Duration", "@duration_fit"),
    ]

    p = figure(
        width=400,
        height=300,
        output_backend="webgl",
        title="Model compilation comparison",
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset",
    )
    p.add_tools(hover_tool)

    p.circle(x="i", y="duration_model", size=10, source=cds)

    p.yaxis.axis_label = "Compilation time (seconds)"
    p.xaxis.axis_label = "Model names (see hover)"

    pn.pane.Bokeh(p)

    p1 = figure(
        width=400,
        height=300,
        output_backend="webgl",
        title="Fit sampling comparison",
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset",
    )
    p1.add_tools(hover_tool)

    p1.circle(x="i", y="duration_fit", size=10, source=cds, color="red")

    p1.yaxis.axis_label = "Sampling time (seconds)"
    p1.xaxis.axis_label = "Model names (see hover)"

    panel = pn.Row(p, p1)

    os.makedirs("./results")
    panel.save("results/test.html", resources=INLINE)


if __name__ == "__main__":
    import datetime

    print(datetime.datetime.now())
    get_results("..")
    print(datetime.datetime.now())
