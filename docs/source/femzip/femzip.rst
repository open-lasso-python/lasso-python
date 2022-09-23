
lasso.femzip (beta)
===================

Documentation of the Femzip module. 

.. warning::
   The module is still in a beta phase, thus the API might change in the future.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   FemzipAPI
   FemzipError
   FemzipFileMetadata
   VariableInfo
   FemzipBufferInfo
   FemzipAPIStatus

Example
-------

   .. code-block:: python

      >>> from lasso.femzip import FemzipAPI
      >>> from ctypes import *
      >>> filepath = "path/to/d3plot.fz"

      >>> # Initialize API
      >>> api = FemzipAPI()

   Here we check if we can use the extended FEMZIP-API. The extended FEMZIP-API
   allows reading selected arrays, but reqires a license with the
   feature "FEMUNZIPLIB-DYNA", which can be attained from SIDACT or femzip
   distributors.

   .. code-block:: python

      >>> api.has_femunziplib_license()
      True

   Check if a file is femzipped

   .. code-block:: python

      >>> # check if file is femzipped
      >>> api.is_sidact_file(filepath)
      True

   Check the file and library version. This is usually not neccessary.

   .. code-block:: python

      >>> api.is_femunzip_version_ok(filepath)
      True

   It's efficient to get the memory demand for arrays beforehand and hand this memory info to other functions. It is often not mandatory though and a mere speedup.

   .. code-block:: python

      >>> # read memory demand info first
      >>> buffer_info = api.get_buffer_info(filepath)
      >>> # buffer info is a c struct, but we can print it
      >>> api.struct_to_dict(buffer_info)
      {'n_timesteps': 12, 'timesteps': <lasso.femzip.femzip_api.LP_c_float object at 0x0000028A8F6B21C0>, 'size_geometry': 537125, 'size_state': 1462902, 'size_displacement': 147716, 'size_activity': 47385, 'size_post': 1266356, 'size_titles': 1448}
      >>> for i_timestep in range(buffer_info.n_timesteps):
      >>>     print(buffer_info.timesteps[i_timestep])
      0.0
      0.9998100399971008
      1.9998900890350342
      2.9999701976776123
      3.9997801780700684

   Here we read the geometry buffer. The file is kept open so that we can 
   read states afterwards.

   .. code-block:: python

      >>> mview = api.read_geometry(filepath, buffer_info, close_file=False)
      
   Femzip can handle only one file per process. In case of issues close the current file (shown later). We can check the API status as follows

   .. code-block:: python

      >>> print(api.struct_to_dict(api.get_femzip_status()))
      {'is_file_open': 1, 'is_geometry_read': 1, 'is_states_open': 0, 'i_timestep_state': -1, 'i_timestep_activity': -1}

   Get the memory of a single state. Must start at 0. Femzip does not allow reading arbitrary states inbetween.

   .. code-block:: python
      
      >>> mview = api.read_single_state(i_timestep=0, buffer_info=buffer_info)

   It is also possible to read the state memory directly into an already
   allocated buffer.

   .. code-block:: python

      >>> BufferType = c_float * (buffer_info.size_state)
      >>> mview = memoryview(BufferType())
      >>> api.read_single_state(1, buffer_info=buffer_info, state_buffer=mview)

   Let's close the file manually. This ensures that femzip resets its internal state.

   .. code-block:: python

      >>> api.close_current_file()