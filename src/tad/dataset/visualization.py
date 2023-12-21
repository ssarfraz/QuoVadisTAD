"""
this module contains functions for plotting.
"""
import numpy as np
from typing import List
from datetime import datetime
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _convert_label_dim(labels: np.array) -> np.array:
    """Some datasets offer anomaly labels on both timestamp and sensor level (2D np.array).
    For evaluation, we only consider the anomalies over timestamp. This function ignores the anomalies labels at the sensor level.
    :param labels: the anomaly labels
    :return: the anomaly labels at timestamp level (1D np.array)
    """
    if len(labels.shape) > 1:  # if there is also info. about sensor level anomaly
        # take the max value of each row for evaluation - if a model can identify the anomaly timestamp
        test_labels = labels.max(1)
    else:
        test_labels = labels
    return test_labels


def convert_array_to_dataframe_for_plotting(sensor_arr: np.array, label_arr: np.array) -> pd.DataFrame:
    """inputs are two numpy arrays from the test sets (labeled data). This function converts numpy arrays to pandas.DataFrame.
    :param sensor_arr: sensor data in 2D np.array - rows: timestamp; cols: sensors
    :param label_arr: label data in 1D np.array. We only consider anomalies at certain timestamp.
        rows: timestamp; values: 0 (normal) or 1 (anomaly).
    :return: pandas.DataFrame[
        "startTime_date_time": datetime,
        "label": (0,1),
        "sensor_k": measurements
    ]
    """
    
    if sensor_arr.ndim>1:
        # define the dimensionality
        nrows, ncols = sensor_arr.shape
        
        # create pd.dataframe with sensor data
        sensor_df = pd.DataFrame(
            data=sensor_arr,
            columns=[f"sensor_{i+1}" for i in range(ncols)]
        )
    else:
        nrows = len(sensor_arr)
        # create pd.dataframe with sensor data
        sensor_df = pd.DataFrame(
            data=sensor_arr,
            columns=["sensor"]
        )

    # ensure the dimensionality of the label_arr
    label_arr = _convert_label_dim(label_arr)

    assert label_arr.ndim == 1
    assert label_arr.shape[0] == nrows

    # create pd.dataframe with labels
    label_df = pd.DataFrame(data={
        "startTime_date_time": pd.to_datetime(pd.date_range(datetime.today(), freq="S", periods=nrows)).astype(str),
        "label": label_arr,
    })

    # concat and return
    return pd.concat([label_df, sensor_df], axis=1)


def plot_time_series_with_pandas_dataframe(
        df: pd.DataFrame,
        cols_to_plot: List[str],
        title_text: str,
        flag_col_name: str = "isEvent",
        threshold_col_name: str = None) -> None:
    """
    this plots all the input column as subplot over timestamp. Input: pandas.DataFrame
    :param df: input pandas data frame.
        It must contain a `startTime_date_time` column (data type: datetime) and the column names in `cols_to_plot`.
    :param cols_to_plot: the column name of measurements
    :param title_text: the title of the plot
    :param flag_col_name: the column of the event identifier (1: event; 0: not event).
        It will be shown as a boxcar function on the plot.
    :param threshold_col_name: the column name of `threshold`. It will be shown as a dash reference line on the plot.
    x: startTime, y: measurement
    """

    # high_light_region
    high_light_df = df[df[flag_col_name] == 1]

    # set up the config
    shapes = []
    nrows = len(cols_to_plot)
    fig = make_subplots(
        rows=nrows, cols=1,
        specs=[[{"rowspan": 1}]] * nrows,
        shared_xaxes=True,
        vertical_spacing=0.05,
        x_title='Start Time',
        subplot_titles=cols_to_plot,
    )
    # plot
    for i in range(nrows):
        y_col_name = cols_to_plot[i]
        fig.add_trace(
            go.Scatter(x=df["startTime_date_time"], y=df[y_col_name], name=y_col_name, showlegend=False,),
            row=i + 1, col=1,
        )

        # Add the dashline indicating threshold
        if threshold_col_name:
            # add dashline
            fig.add_trace(
                go.Scatter(
                    x=df["startTime_date_time"],
                    y=df[threshold_col_name],
                    mode='lines',
                    line={
                        'dash': 'dash',
                        'color': 'gray'},
                    showlegend=False),
                row=i + 1,
                col=1,
            )

            # add annotation text `threshold` on the first subplot
            if i == 0:
                # set the location of start on the plot
                x_annotation = df["startTime_date_time"].values[-1000]
                y_annotation = df[threshold_col_name].values[0] * 1.10
                fig.add_annotation(x=x_annotation, y=y_annotation,
                                   text="Threshold",
                                   showarrow=False,
                                   )

        # mark fault event with colored area
        dt_format = "%Y-%m-%d %H:%M:%S.%f"
        shapes.append({
            'type': 'rect',
            'xref': f'x1',
            'yref': f'y{i+1}',
            'x0': datetime.strptime(high_light_df["startTime_date_time"].min(), dt_format),
            'y0': df[y_col_name].min(),
            'x1': datetime.strptime(high_light_df["startTime_date_time"].max(), dt_format),
            'y1': df[y_col_name].max(),
            'opacity': 0.5,
            'line_color': 'LightSkyBlue',
            'fillcolor': 'LightSkyBlue',
        })

    # add the shade
    fig['layout'].update(shapes=shapes)

    # Change the range for the pa, if "adjusted_prediction" is in the `cols_to_plot`
    if "adjusted_prediction" in cols_to_plot:
        fig.update_yaxes(title_text="adjusted_prediction",
                         range=[-.05, 1.05], row=cols_to_plot.index("adjusted_prediction") + 1, col=1)

    fig.update_layout(height=500, width=1000, title_text=title_text)
    fig.show()