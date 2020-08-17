#!/usr/bin/env python3
# thoth-report-visualization
# Copyright(C) 2020 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Amun Inspection results visualization methods."""

import os
import logging

from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import iplot

pd.options.plotting.backend = "plotly"

# set up logging
DEBUG_LEVEL = bool(int(os.getenv("DEBUG_LEVEL", 0)))

if DEBUG_LEVEL:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

_LOGGER = logging.getLogger(__name__)

_PERFORMANCE_QUANTITY = ["elapsed_time", "rate"]

_PERFORMANCE_QUANTITY_MAP = {"elapsed_time": "Elapsed Time [ms]", "rate": "Rate [GFLOPS]"}


class AmunInspectionsVisualization:
    """Class of methods used to create statistics from Amun Inspections Runs."""

    @staticmethod
    def create_inspection_3d_plot(plot_df: pd.DataFrame, quantity: str, identifiers_inspections: List[str]):
        """Create inspection performance parameters plot in 3D.

        :param plot_df dataframe for plot of inspections results
        """
        if quantity not in _PERFORMANCE_QUANTITY:
            _LOGGER.info(f"Only {_PERFORMANCE_QUANTITY} are accepted as quantity")
            return

        x_vector = [x[0] for x in plot_df[["re_string"]].values]

        integer_y_encoded = [y[0] for y in plot_df[["sws_hash_id_encoded"]].values]

        z_vector = [z[0] for z in plot_df[[quantity]].values]

        trace1 = go.Scatter3d(
            x=x_vector,
            y=integer_y_encoded,
            z=z_vector,
            mode="markers",
            hovertext=[yc[0] for yc in plot_df[["sws_string"]].values],
            hoverinfo="text",
            marker=dict(
                size=12,
                color=z_vector,  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
                showscale=True,
            ),
            name=f"PI=Conv2D-tensorflow-{identifiers_inspections}",
        )

        data = [trace1]

        annotations = []
        c = 0

        for (x, y, z) in zip(x_vector, integer_y_encoded, z_vector):
            annotations.append(
                dict(
                    showarrow=False,
                    x=x,
                    y=y,
                    z=z,
                    text="".join(plot_df["tensorflow"].values[c]),
                    xanchor="left",
                    xshift=15,
                    opacity=0.7,
                )
            )
            c += 1

        margin = {"l": 0, "r": 0, "b": 0, "t": 0}

        layout = go.Layout(
            title="PI=Conv2D",
            margin=margin,
            scene=dict(
                xaxis=dict(title="Runtime Environment"),
                yaxis=dict(title="Software Stack ID integer encoded"),
                zaxis=dict(title=_PERFORMANCE_QUANTITY_MAP[quantity]),
                #         annotations=annotations,
            ),
            showlegend=True,
            legend=dict(orientation="h"),
        )
        fig = go.Figure(data=data, layout=layout)

        iplot(fig, filename="3d-scatter-colorscale")

    @staticmethod
    def create_inspection_2d_plot(
        plot_df: pd.DataFrame,
        quantity: str,
        components: List[str],
        color_scales: List[str],
        identifiers_inspections: List[str],
        have_annotations: bool = False,
    ):
        """Create inspection performance parameters plot in 2D.

        :param plot_df dataframe for plot of inspections results
        """
        integer_y_encoded = [y[0] for y in plot_df[["sws_hash_id_encoded"]].values]

        data = []
        annotations = []
        colour_counter = 0
        distance: float = 0
        for component in components:
            name_component = component
            if component == "pytorch":
                name_component = "torch"
            subset_df = plot_df[plot_df["pi_component"] == component]
            z_vector = [z[0] for z in subset_df[[quantity]].values]

            trace = go.Scatter(
                x=integer_y_encoded,
                y=z_vector,
                mode="markers",
                hovertext=[y[0] for y in subset_df[["sws_string"]].values],
                hoverinfo="text",
                marker=dict(
                    size=12,
                    color=z_vector,  # set color to an array/list of desired values
                    colorscale=color_scales[colour_counter],  # choose a colorscale
                    opacity=0.8,
                    showscale=True,
                    colorbar={"x": 1 + distance},
                ),
                name=f"{component}==version",
            )

            data.append(trace)
            colour_counter += 1
            distance += 0.2

            if have_annotations:
                for (yr, zr) in zip(integer_y_encoded, z_vector):
                    annotations.append(
                        dict(
                            showarrow=False,
                            x=yr,
                            y=zr,
                            text=f"{name_component}{subset_df[name_component].values[colour_counter][1]}, "
                            + f"np{subset_df[name_component].values[colour_counter][1]}",
                            xanchor="left",
                            xshift=15,
                            opacity=0.7,
                        )
                    )

        layout = go.Layout(
            title=f"PI=Conv2D-{components}-{identifiers_inspections}-2Dplot",
            xaxis=dict(title="Software Stack ID integer encoded"),
            yaxis=dict(title=_PERFORMANCE_QUANTITY_MAP[quantity]),
            annotations=annotations,
            showlegend=True,
            legend=dict(orientation="h", y=-0.3, yanchor="top"),
        )
        fig = go.Figure(data=data, layout=layout)

        iplot(fig, filename="scatter-colorscale")
