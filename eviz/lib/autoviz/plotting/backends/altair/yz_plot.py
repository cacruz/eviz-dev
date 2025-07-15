import numpy as np
import pandas as pd
import altair as alt
import logging

from eviz.lib.autoviz.plotting.base import YZPlotter


class AltairYZPlotter(YZPlotter):
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
        alt.renderers.enable("default")
        alt.data_transformers.disable_max_rows()

    def plot(self, config, plot_data):
        data2d, x, y, field_name, plot_type, file_index, figure = plot_data
        if data2d is None:
            return None

        ax_opts = config.ax_opts

        if "fill_value" in config.spec_data[field_name]["xyplot"]:
            fill_value = config.spec_data[field_name]["xyplot"]["fill_value"]
            data2d = data2d.where(data2d != fill_value, np.nan)

        # Colormap mapping
        cmap = ax_opts.get("use_cmap", "viridis")
        cmap_mapping = {
            "viridis": "viridis",
            "plasma": "plasma",
            "inferno": "inferno",
            "magma": "magma",
            "cividis": "cividis",
            "rainbow": "rainbow",
            "jet": "rainbow",
            "Blues": "blues",
            "Reds": "reds",
            "Greens": "greens",
            "YlOrRd": "yelloworangered",
            "RdBu": "redblue",
            "coolwarm": "redblue",
        }
        vega_scheme = cmap_mapping.get(cmap, "viridis")

        title = config.spec_data[field_name].get("name", field_name)
        units = config.spec_data[field_name].get(
            "units", getattr(data2d, "units", "n.a.")
        )

        # DataFrame construction
        df = self._convert_to_dataframe(data2d, x, y)
        if df.empty or df["value"].isna().all():
            self.logger.error("DataFrame is empty or all values are NaN")
            return None

        x_name = x.name if hasattr(x, "name") else "Latitude"
        y_name = y.name if hasattr(y, "name") else "Pressure (hPa)"

        # Invert y-axis for pressure
        y_is_pressure = (
            "pressure" in y_name.lower()
            or "lev" in y_name.lower()
            or "pres" in y_name.lower()
        )
        if y_is_pressure:
            y_domain = [df["y"].max(), df["y"].min()]  # High to low
            y_sort = "descending"
        else:
            y_domain = [df["y"].min(), df["y"].max()]
            y_sort = "ascending"

        x_ticks = [-90, -60, -30, 0, 30, 60, 90]
        x_labels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]

        if ax_opts.get("clevs") is not None:
            color_domain = ax_opts["clevs"]
        else:
            color_domain = [df["value"].min(), df["value"].max()]

        chart = (
            alt.Chart(df)
            .mark_square(size=100)
            .encode(
                x=alt.X(
                    "x:Q",
                    title=x_name,
                    scale=alt.Scale(domain=[df["x"].min(), df["x"].max()], nice=False),
                    axis=alt.Axis(
                        values=x_ticks,
                        labelExpr="{'-90':'90S','-60':'60S','-30':'30S','0':'EQ','30':'30N','60':'60N','90':'90N'}[datum.value] || datum.value",
                    ),
                ),
                y=alt.Y(
                    "y:Q",
                    title=y_name,
                    scale=alt.Scale(domain=[df["y"].max(), df["y"].min()], nice=False),
                ),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(scheme=vega_scheme, domain=color_domain),
                    title=f"{field_name} ({units})",
                ),
                tooltip=[
                    alt.Tooltip("x:Q", title=x_name),
                    alt.Tooltip("y:Q", title=y_name),
                    alt.Tooltip("value:Q", title=field_name, format=".3f"),
                ],
            )
            .properties(width=800, height=500, title=title)
            .interactive()
        )

        self.plot_object = chart
        return chart

    def _convert_to_dataframe(self, data2d, x, y):
        x_vals = x.values if hasattr(x, "values") else x
        y_vals = y.values if hasattr(y, "values") else y
        data_values = data2d.values if hasattr(data2d, "values") else data2d

        rows = []
        for i, x_val in enumerate(x_vals):  # lat (horizontal)
            for j, y_val in enumerate(y_vals):  # lev/pressure (vertical)
                if j < data_values.shape[0] and i < data_values.shape[1]:
                    value = data_values[j, i]
                    if not np.isnan(value):
                        rows.append({"x": x_val, "y": y_val, "value": value})
        return pd.DataFrame(rows)

    def show(self):
        pass

    def save(self, filename, **kwargs):
        pass
