# DimredRun

This class can be used in order to comfortably represent multiple D3plot samples
in a 3D graph through dimensionality reduction.
In the 3D graph every simulation is a data point and the closeness of the data
points represents the similarity of the result field.
It offers functions to:

- Subsample simulations to reduce the computational effort and account for
  different meshing.
- Reduce order which is the black magic reducing complex field results to
  a few numbers which can then be plotted in 3D.
- Clustering with sklearn to group simulations semi-automatically
  (mostly just a visual feature).
- Output 3D similarity plot as a webpage

For ease of use, check out the `Tool` section, which explains the command line
tool for this dimensionality reduction feature.

::: lasso.dimred.dimred_run.DimredRun
    options:
      members:
        - __init__
        - process_reference_run
        - subsample_to_reference_run
        - dimension_reduction_svd
        - clustering_results
        - visualize_results
