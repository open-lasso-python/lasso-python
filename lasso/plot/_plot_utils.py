import numpy as np
from tempfile import NamedTemporaryFile
import webbrowser


_PLOTLY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{TITLE}</title>
    <script type="text/javascript" src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
  <body>
    <script type="text/javascript">
    'use strict';
    {{
      let plot = document.createElement('div');
      plot.style = 'width: 100vw; height: 100vh';
      plot.innerHTML = 'Creating Plot...';
      document.body.appendChild(plot);

      let
        data = [
          {{
            name: 'Scatter', type: 'scatter3d', mode: 'markers',
            x: [{X_PTS}],
            y: [{Y_PTS}],
            z: [{Z_PTS}],
            mode: 'markers',
            text: {_scatter_fringe_text},
            marker: {{
              color: {_scatter_fringe_color},
              size: 5.0,
              //line: {{ width: 0.5, color: 'black' }},
              line:{{'width': 0}},
              showscale: true,
              autocolorscale: true,
              cauto: true
            }},
          }}
        ],
        layout = {{
          scene: {{
            xaxis: {{
                title: 'X-Axis',
                {_x_range}
            }},
            yaxis: {{
                title: 'Y-Axis',
                {_y_range}
            }},
            zaxis: {{
                title: 'Z-Axis',
                {_z_range}
            }}
          }},
          title: '{TITLE}',
          {_scatter_updatemenu}
        }};
      plot.innerHTML = '';
      Plotly.plot(
        plot, data, layout,
        {{ showLink: false, modeBarButtonsToRemove: ['sendDataToCloud'] }}
      );
      window.onresize = () => Plotly.Plots.resize(plot);
    }}
    </script>
  </body>
</html>;
"""

_PLOTLY_UPDATEMENU = """updatemenus: [{{
            active: 0,
            yanchor: 'top',
            buttons: [
{_buttons}            ],
        }}],
"""

_PLOTLY_BUTTON = """            {{
                args: [{{
                  text: [{_button_fringe_text}],
                  marker: [{{color: {_button_marker_color}}}],
                  showscale: true,
                  autocolorscale: true,
                  cauto: true
                }}],
                method: 'restyle',
                label: {_button_text}
            }},
"""


def array2str(arr: np.ndarray) -> str:
    entries_per_row = 100
    return ",\n            ".join(
        ", ".join(map(str, arr[i: i + entries_per_row]))
        for i in range(0, len(arr), entries_per_row)
    )


def plotly_3d(x, y, z, fringe=None, fringe_names=None, filepath=None, title=""):

    assert x.shape == y.shape
    assert x.shape == z.shape

    # axis equalizing
    xmax = x.max()
    xmin = x.min()
    dx = xmax - xmin
    xmid = (xmax + xmin) / 2

    ymax = y.max()
    ymin = y.min()
    dy = ymax - ymin
    ymid = (ymax + ymin) / 2

    zmax = z.max()
    zmin = z.min()
    dz = zmax - zmin
    zmid = (zmax + zmin) / 2

    dmax = max(dx, dy, dz) * 1.1
    x_range = [xmid - dmax / 2, xmid + dmax / 2]
    x_range = "range: %s" % str(x_range)
    y_range = [ymid - dmax / 2, ymid + dmax / 2]
    y_range = "range: %s" % str(y_range)
    z_range = [zmid - dmax / 2, zmid + dmax / 2]
    z_range = "range: %s" % str(z_range)

    if not isinstance(fringe, (np.ndarray, list, tuple)):
        fringe = np.zeros(len(x))
    scatter_hover_labels = []
    color_range = "'red'"

    # multiple fringes
    fringe = np.asarray(fringe)
    if fringe.ndim > 1:

        # fringe names?
        if not fringe_names:
            fringe_names = ["fringe " + str(ii) for ii in range(fringe.ndim)]
        else:
            fringe_names = ["'%s'" % name for name in fringe_names]

        # coloring
        color_range = "[" + array2str(fringe[0]) + "]"
        scatter_hover_labels = color_range

        # buttons
        buttons_str = "".join(
            [
                _PLOTLY_BUTTON.format(
                    _button_fringe_text="[" + array2str(fringe[ii]) + "]",
                    # _button_fringe_text="[]",
                    _button_marker_color="[" + array2str(fringe[ii]) + "]",
                    # _button_marker_color="",
                    _button_text=fringe_names[ii],
                )
                for ii in range(len(fringe))
            ]
        )

        button_menu_str = _PLOTLY_UPDATEMENU.format(_buttons=buttons_str)

        scatter_hover_labels = fringe[0]
    # single fringe only, no menu needed
    else:
        button_menu_str = ""

    if len(scatter_hover_labels) == 0:
        scatter_hover_labels = array2str(fringe)

    # export as file if filepath is given
    if filepath:
        with open(filepath, "w") as plot_file:
            plot_file.write(
                _PLOTLY_TEMPLATE.format(
                    TITLE=title,
                    X_PTS=array2str(x),
                    Y_PTS=array2str(y),
                    Z_PTS=array2str(z),
                    _scatter_updatemenu=button_menu_str,
                    _scatter_fringe_text=color_range,
                    _scatter_fringe_color=color_range,
                    _x_range=x_range,
                    _y_range=y_range,
                    _z_range=z_range,
                )
            )

    # plot temporary file if no filepath
    else:
        with NamedTemporaryFile(prefix="plot", suffix=".html", mode="w", delete=False) as plot_file:
            plot_file.write(
                _PLOTLY_TEMPLATE.format(
                    TITLE=title,
                    X_PTS=array2str(x),
                    Y_PTS=array2str(y),
                    Z_PTS=array2str(z),
                    _scatter_updatemenu=button_menu_str,
                    _scatter_fringe_text=color_range,
                    _scatter_fringe_color=color_range,
                    _x_range=x_range,
                    _y_range=y_range,
                    _z_range=z_range,
                )
            )

            webbrowser.open("file://" + plot_file.name)
