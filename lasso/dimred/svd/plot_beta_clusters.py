import os
import re
import time
import webbrowser
from typing import Sequence, Union

import numpy as np

from lasso.dimred.svd.html_str_eles import (
    CONST_STRING,
    OVERHEAD_STRING,
    SCRIPT_STRING,
    TRACE_STRING,
)
from lasso.plotting.plot_shell_mesh import _read_file


def timestamp() -> str:
    """
    Creates a timestamp string of format yymmdd_hhmmss_
    """

    def add_zero(in_str) -> str:
        if len(in_str) == 1:
            return "0" + in_str
        return in_str

    year, month, day, hour, minute, second, _, _, _ = time.localtime()
    y_str = str(year)[2:]
    mo_str = add_zero(str(month))
    d_str = add_zero(str(day))
    h_str = add_zero(str(hour))
    mi_str = add_zero(str(minute))
    s_str = add_zero(str(second))
    t_str = y_str + mo_str + d_str + "_" + h_str + mi_str + s_str + "_"
    return t_str


# pylint: disable = inconsistent-return-statements
def plot_clusters_js(
    beta_cluster: Sequence,
    id_cluster: Union[np.ndarray, Sequence],
    save_path: str,
    img_path: Union[None, str] = None,
    mark_outliers: bool = False,
    mark_timestamp: bool = True,
    filename: str = "3d_beta_plot",
    write: bool = True,
    show_res: bool = True,
) -> Union[None, str]:
    """
    Creates a .html visualization of input data

    Parameters
    ----------
    beta_cluster: np.ndarray
        Numpy array containing beta clusters
    id_cluster: Union[np.ndarray, Sequence]
        Numpy array or sequence containing the ids samples in clusters.
        Must be of same structure as beta_clusters
    save_path: str
        Where to save the .html visualization
    img_path: Union[None, str], default: None
        Path to images of samples
    mark_outliers: bool, default: False
        Set to True if first entry in beta_cluster are outliers
    mark_timestamp: bool, default: True
        Set to True if name of visualization shall contain time of creation.
        If set to False, visualization will override previous file
    filename: str, default "3d_beta_plot"
        Name of .hmtl file
    write: bool, default: True
        Set to False to not write .html file and return as string instead
    show_res: bool, default: True
        Set to False to not open resulting page in webbrowser

    Returns
    -------
    html_str_formatted: str
        If **write=False** returns .hmtl file as string, else None
    """

    # pylint: disable = too-many-arguments, too-many-locals

    if not isinstance(img_path, str):
        img_path = ""

    colorlist = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    tracelist = []

    # rescaling betas to better fit in viz
    scale_multiplier = 300
    max_val = (
        max(*[max(np.max(cluster), abs(np.min(cluster))) for cluster in beta_cluster])
        if len(beta_cluster) > 1
        else max(np.max(beta_cluster[0]), abs(np.min(beta_cluster[0])))
    )

    beta_cluster = [cluster / max_val * scale_multiplier for cluster in beta_cluster]

    id_nr = []
    for group in id_cluster:
        id_group = []
        for entry in group:
            nr = re.findall(r"\d+", entry)[0]
            id_group.append(nr)
        id_nr.append(id_group)

    # pylint: disable = consider-using-f-string
    _three_min_ = '<script type="text/javascript">%s</script>' % _read_file(
        os.path.join(
            # move path to "~/lasso/"
            os.path.split(os.path.split(os.path.dirname(__file__))[0])[0],
            "plotting/resources/three_latest.min.js",
        )
    )

    html_str_formatted = OVERHEAD_STRING + CONST_STRING.format(
        _three_min_=_three_min_, _path_str_=img_path, _runIdEntries_=id_nr
    )
    for index, cluster in enumerate(beta_cluster):
        name = "Error, my bad"
        color = "pink"
        if (index == 0) and mark_outliers:
            name = "outliers"
            color = "black"
        else:
            name = "cluster {i}".format(i=index)
            color = colorlist[(index - 1) % 10]
        formated_trace = TRACE_STRING.format(
            _traceNr_="trace{i}".format(i=index),
            _name_=name,
            _color_=color,
            _runIDs_=id_cluster[index].tolist(),
            _x_=np.around(cluster[:, 0], decimals=5).tolist(),
            _y_=np.around(cluster[:, 1], decimals=5).tolist(),
            _z_=np.around(cluster[:, 2], decimals=5).tolist(),
        )
        tracelist.append(f"trace{index}")
        html_str_formatted += formated_trace
    trace_list_string = "    traceList = ["
    for trace in tracelist:
        trace_list_string += trace + ", "
    trace_list_string += "]"
    html_str_formatted += trace_list_string
    html_str_formatted += SCRIPT_STRING

    if write:
        os.makedirs(save_path, exist_ok=True)

        # Timestamp for differentiating different viz / not override previous viz
        stamp = timestamp() if mark_timestamp else ""

        output_filepath = os.path.join(save_path, stamp + filename + ".html")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_str_formatted)
        if show_res:
            webbrowser.open("file://" + os.path.realpath(output_filepath))
    else:
        # only needed for testcases
        return html_str_formatted
