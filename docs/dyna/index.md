# lasso.dyna

The dyna module contains classes to read, write and display LS-Dyna result
files.
For a detailed list of features, see the following list:

- D3plot
    - Read & Write
        - Beam, Shell, Solid, TShell Results
        - Element/Node deletion
        - Part Metadata
        - Material Type Data
        - Rigid Body Data
        - Rigid Road Data
        - SPH
        - Particles
    - Plot shell-based parts as 3D HTML
- Binout
    - Read & Write

!!! note

    The D3plot reader can process almost every structural result or results
    connected to it such as airbag results. Multi-Solver data though and
    connected results are not supported.

## D3plot Example

``` python
>>> from lasso.dyna import D3plot, ArrayType, FilterType

>>> # read a file (zero-copy reading of everything)
>>> d3plot = D3plot("path/to/d3plot")

>>> # read file 
>>> # - buffered (less memory usage)
>>> # - only node displacements (safes memory)
>>> # - read only first and last state
>>> d3plot = D3plot("path/to/d3plot",
>>>                 state_array_filter=["node_displacement"],
>>>                 buffered_reading=True,
>>>                 state_filter={0, -1})

>>> # and of course femzipped files
>>> d3plot = D3plot("path/to/d3plot.fz")

>>> # get arrays (see docs of ArrayType for shape info)
>>> disp = d3plot.arrays["node_displacement"]
>>> disp.shape
(34, 51723, 3)
>>> # this is safer and has auto-completion
>>> disp = d3plot.arrays[ArrayType.node_displacement]

>>> # filter elements for certain parts
>>> pstrain = d3plot.arrays[ArrayType.element_shell_effective_plastic_strain]
>>> pstrain.shape
(34, 56372, 3)
>>> mask = d3plot.get_part_filter(FilterType.SHELL, [44, 45])
>>> # filter elements with mask
>>> pstrain[:, mask].shape
(34, 17392, 3)

>>> # create a standalone html plot
>>> d3plot.plot()
```
