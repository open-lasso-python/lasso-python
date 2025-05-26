import io
import json
import os
import uuid
from base64 import b64encode
from typing import Union
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def _read_file(filepath: str):
    """This function reads file as str

    Parameters
    ----------
    filepath : str
        filepath of the file to read as string

    Returns
    -------
    file_content : str
    """

    with open(filepath, encoding="utf-8") as fp_filepath:
        return fp_filepath.read()


def plot_shell_mesh(
    node_coordinates: np.ndarray,
    shell_node_indexes: np.ndarray,
    field: Union[np.ndarray, None] = None,
    is_element_field: bool = True,
    fringe_limits: Union[tuple[float, float], None] = None,
):
    """Plot a mesh

    Parameters
    ----------
    node_coordinates : np.ndarray
        array of node coordinates for elements
    shell_node_indexes : np.ndarray
        node indexes of shells
    field : Union[np.ndarray, None]
        Array containing a field value for every element or node
    is_element_field : bool
        if the specified field is for elements or nodes
    fringe_limits : Union[Tuple[float, float], None]
        limits for the fringe bar. Set by default to min and max.

    Returns
    -------
    html : str
        html code for plotting as string
    """

    # pylint: disable = too-many-locals, too-many-statements

    if getattr(node_coordinates, "ndim", None) != 2:
        raise ValueError(
            f"node_coordinates must be 2-dimensional, "
            f"got ndim={getattr(node_coordinates, 'ndim', 'unknown')}"
        )

    if getattr(node_coordinates, "shape", (None, None))[1] != 3:
        raise ValueError(
            f"node_coordinates must have shape[1] == 3, "
            f"got shape={getattr(node_coordinates, 'shape', 'unknown')}"
        )

    if getattr(shell_node_indexes, "ndim", None) != 2:
        raise ValueError(
            f"shell_node_indexes must be 2-dimensional, "
            f"got ndim={getattr(shell_node_indexes, 'ndim', 'unknown')}"
        )

    shape_1 = getattr(shell_node_indexes, "shape", (None, None))[1]
    if shape_1 not in (3, 4):
        raise ValueError(f"shell_node_indexes must have shape[1] of 3 or 4, got shape[1]={shape_1}")

    if isinstance(field, np.ndarray):
        if getattr(field, "ndim", None) != 1:
            raise ValueError(
                f"field must be 1-dimensional, got ndim={getattr(field, 'ndim', 'unknown')}"
            )

        if is_element_field:
            if (
                getattr(field, "shape", (None,))[0]
                != getattr(shell_node_indexes, "shape", (None,))[0]
            ):
                raise ValueError(
                    f"field length {getattr(field, 'shape', (None,))[0]} does not match "
                    f"shell_node_indexes length {getattr(shell_node_indexes, 'shape', (None,))[0]}"
                )
        elif getattr(field, "shape", (None,))[0] != getattr(node_coordinates, "shape", (None,))[0]:
            raise ValueError(
                f"field length {getattr(field, 'shape', (None,))[0]} does not match "
                f"node_coordinates length {getattr(node_coordinates, 'shape', (None,))[0]}"
            )

    # cast types correctly
    # the types MUST be float32
    node_coordinates = node_coordinates.astype(np.float32)
    if isinstance(field, np.ndarray):
        field = field.astype(np.float32)

    # distinguish tria and quads
    is_quad = shell_node_indexes[:, 2] != shell_node_indexes[:, 3]
    is_tria = np.logical_not(is_quad)

    # separate tria and quads ... I know its sad :(
    tria_node_indexes = shell_node_indexes[is_tria][:, :3]
    quad_node_indexes = shell_node_indexes[is_quad]

    # we can only plot tria, therefore we need to split quads
    # into two trias
    quad_node_indexes_tria1 = quad_node_indexes[:, :3]
    # quad_node_indexes_tria2 = quad_node_indexes[:, [True, False, True, True]]
    quad_node_indexes_tria2 = quad_node_indexes[:, [0, 2, 3]]

    # assemble elements for plotting
    # This seems to take a lot of memory, and you are right, thinking this,
    # the issue is just in order to plot fringe values, we need to output
    # the element values at the 3 corner nodes. Since elements share nodes
    # we can not use the same nodes, thus we need to create multiple nodes
    # at the same position but with different fringe.
    nodes_xyz = np.concatenate([
        node_coordinates[tria_node_indexes].reshape((-1, 3)),
        node_coordinates[quad_node_indexes_tria1].reshape((-1, 3)),
        node_coordinates[quad_node_indexes_tria2].reshape((-1, 3)),
    ])

    # fringe value and hover title
    if isinstance(field, np.ndarray):
        if is_element_field:
            n_shells = len(shell_node_indexes)
            n_tria = np.sum(is_tria)
            n_quads = n_shells - n_tria

            # split field according to elements
            field_tria = field[is_tria]
            field_quad = field[is_quad]

            # allocate fringe array
            node_fringe = np.zeros((len(field_tria) + 2 * len(field_quad), 3), dtype=np.float32)

            # set fringe values
            node_fringe[:n_tria, 0] = field_tria
            node_fringe[:n_tria, 1] = field_tria
            node_fringe[:n_tria, 2] = field_tria

            node_fringe[n_tria : n_tria + n_quads, 0] = field_quad
            node_fringe[n_tria : n_tria + n_quads, 1] = field_quad
            node_fringe[n_tria : n_tria + n_quads, 2] = field_quad

            node_fringe[n_tria + n_quads : n_tria + 2 * n_quads, 0] = field_quad
            node_fringe[n_tria + n_quads : n_tria + 2 * n_quads, 1] = field_quad
            node_fringe[n_tria + n_quads : n_tria + 2 * n_quads, 2] = field_quad

            # flatty paddy
            node_fringe = node_fringe.flatten()
        else:
            # copy & paste ftw
            node_fringe = np.concatenate([
                field[tria_node_indexes].reshape((-1, 3)),
                field[quad_node_indexes_tria1].reshape((-1, 3)),
                field[quad_node_indexes_tria2].reshape((-1, 3)),
            ])
            node_fringe = node_fringe.flatten()

        # element text
        node_txt = [str(entry) for entry in node_fringe.flatten()]
    else:
        node_fringe = np.zeros(len(nodes_xyz), dtype=np.float32)
        node_txt = [""] * len(nodes_xyz)

    # zip compression of data for HTML (reduces size)
    zip_data = io.BytesIO()
    with ZipFile(zip_data, "w", compression=ZIP_DEFLATED) as zipfile:
        zipfile.writestr("/intensities", node_fringe.tostring())
        zipfile.writestr("/positions", nodes_xyz.tostring())
        zipfile.writestr("/text", json.dumps(node_txt))
    zip_data = b64encode(zip_data.getvalue()).decode("utf-8")

    # read html template
    _html_template = _read_file(
        os.path.join(os.path.dirname(__file__), "resources", "template.html")
    )

    # format html template file
    min_value = 0
    max_value = 0
    if fringe_limits:
        min_value = fringe_limits[0]
        max_value = fringe_limits[1]
    elif isinstance(field, np.ndarray):
        min_value = field.min()
        max_value = field.max()

    _html_div = _html_template.format(
        div_id=uuid.uuid4(), lowIntensity=min_value, highIntensity=max_value, zdata=zip_data
    )

    # wrap it up with all needed js libraries

    script_string_js = '<script type="text/javascript">{0}</script>'
    jszip_js_format = _read_file(
        os.path.join(os.path.dirname(__file__), "resources", "jszip.min.js")
    )
    jszip_three_format = _read_file(
        os.path.join(os.path.dirname(__file__), "resources", "three.min.js")
    )
    jszip_chroma_format = _read_file(
        os.path.join(os.path.dirname(__file__), "resources", "chroma.min.js")
    )
    jszip_jquery_format = _read_file(
        os.path.join(os.path.dirname(__file__), "resources", "jquery.min.js")
    )
    _html_jszip_js = script_string_js.format(jszip_js_format)
    _html_three_js = script_string_js.format(jszip_three_format)
    _html_chroma_js = script_string_js.format(jszip_chroma_format)
    _html_jquery_js = script_string_js.format(jszip_jquery_format)

    # pylint: disable = consider-using-f-string
    return f"""
<!DOCTYPE html>
<html lang="en">
    <head>
    <meta charset="utf-8" />
        {_html_jquery_js}
        {_html_jszip_js}
        {_html_three_js}
        {_html_chroma_js}
    </head>
    <body>
        {_html_div}
    </body>
</html>"""
